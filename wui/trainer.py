from __future__ import annotations
import os
import sys
import json
import math
import random
import datetime
import shutil
import re
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
import gradio as gr

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextTokenizer

from core import core
from core.core import _
from core.spice import GenericSpiceTokenizer
from core.normalizer import MultilingualNormalizer, MultilingualWordifier

# --- OPTIMIZATION: ENV VARS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

# --- GLOBAL CONTROL FLAG ---
STOP_TRAINING = False

def stop_training_fn():
    global STOP_TRAINING
    STOP_TRAINING = True
    return _("TRAINER_MSG_STOPPING")
    
# ==========================================
# Helpers
# ==========================================

def get_checkpoint_list(run_name):
    """Lists all .pth files in the training directory."""
    if not run_name:
        return gr.Dropdown(choices=[])
    
    run_dir = os.path.join(core.path_base, "trains", run_name)
    if not os.path.exists(run_dir):
        return gr.Dropdown(choices=[])
    
    # List .pth files, excluding the output name 'gpt.pth' to avoid confusion
    files = [f for f in os.listdir(run_dir) if f.endswith(".pth") and f != "gpt.pth"]
    return gr.Dropdown(choices=sorted(files))

def unwrap_and_save_handler(run_name, selected_file, apply_conformer_map):
    """Unwraps the selected checkpoint and saves it as gpt.pth in the same folder."""
    if not run_name or not selected_file:
        return "❌ Error: Selection missing."
    
    run_dir = os.path.join(core.path_base, "trains", run_name)
    ckpt_path = os.path.join(run_dir, selected_file)
    output_path = os.path.join(run_dir, "gpt.pth")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        
        new_state = {}
        for k, v in state_dict.items():
            # 1. Clean 'module.' prefixes
            k = k.replace("module.", "")
            k = k.replace("_orig_mod.", "")
            
            # 2. Conformer Naming Fix: Maps flat 'conv_' names back to nested 'conv_module' structure
            if apply_conformer_map:
                k = k.replace(".conv_pointwise_conv1", ".conv_module.pointwise_conv1")
                k = k.replace(".conv_pointwise_conv2", ".conv_module.pointwise_conv2")
                k = k.replace(".conv_depthwise_conv",  ".conv_module.depthwise_conv")
                k = k.replace(".conv_norm",           ".conv_module.norm")
                
            new_state[k] = v
        
        torch.save(new_state, output_path)
        return f"✅ Successfully unwrapped {selected_file} ➔ gpt.pth"
    except Exception as e:
        return f"❌ Error: {str(e)}"
        
# ==========================================
# 1. DATASET LOGIC
# ==========================================

@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    sample_type: str = "single"

class IndexTTSDataset(Dataset):
    def __init__(self, manifest_path: str, vector_root: Optional[Path] = None):
        self.samples: List[Sample] = []
        self.max_text_found = 0
        self.max_code_found = 0
        
        path = Path(manifest_path)
        if not path.exists():
            print(f"❌ Manifest not found: {path}")
            return

        base_dir = path.parent
        
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing JSON at line {line_no}: {e}")
                    continue
                
                t_len = int(rec["text_len"])
                c_len = int(rec["code_len"])
                
                if t_len > self.max_text_found: self.max_text_found = t_len
                if c_len > self.max_code_found: self.max_code_found = c_len

                # --- ROBUST RESOLVER ---
                def resolve(p_str, subfolder=None):
                    p_obj = Path(p_str)
                    
                    # 1. Try Re-linking (Override via UI Selection)
                    if vector_root:
                        if subfolder:
                            candidate = vector_root / subfolder / p_obj.name
                            if candidate.exists(): return candidate
                        
                        candidate_flat = vector_root / p_obj.name
                        if candidate_flat.exists(): return candidate_flat

                    # 2. Try Absolute Path (Legacy / Standard)
                    if p_obj.is_absolute():
                        if p_obj.exists(): return p_obj
                        
                    # 3. Try Relative to Manifest (Standard JSONL behavior)
                    candidate_rel = base_dir / p_obj
                    if candidate_rel.exists(): return candidate_rel

                    # 4. Try Relative to Global Projects Directory
                    # Note: Assumes core.path_base points to 'wui/' and projects are in 'wui/projects/' 
                    # consistent with uploaded project.py logic.
                    candidate_global = Path(core.path_base) / "projects" / p_obj
                    if candidate_global.exists(): return candidate_global
                    
                    return p_obj

                self.samples.append(Sample(
                    id=rec["id"],
                    text_ids_path=resolve(rec["text_ids_path"], "text_ids"),
                    codes_path=resolve(rec["codes_path"], "codes"),
                    condition_path=resolve(rec["condition_path"], "condition"),
                    emo_vec_path=resolve(rec["emo_vec_path"], "emo_vec"),
                    text_len=t_len,
                    code_len=c_len,
                    condition_len=32 
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            text = np.load(sample.text_ids_path).astype(np.int64)
            codes = np.load(sample.codes_path).astype(np.int64)
            cond = np.load(sample.condition_path).astype(np.float32)
            emo = np.load(sample.emo_vec_path).astype(np.float32)

            return {
                "id": sample.id,
                "text_ids": torch.from_numpy(text),
                "codes": torch.from_numpy(codes),
                "condition": torch.from_numpy(cond),
                "emo_vec": torch.from_numpy(emo),
                "text_len": torch.tensor(sample.text_len, dtype=torch.long),
                "code_len": torch.tensor(sample.code_len, dtype=torch.long)
            }
        except Exception as e:
            print(f"⚠️ Error loading sample {sample.id}: {e}. Path used: {sample.codes_path}")
            return None

def collate_batch(batch, max_text_val=11999, max_code_val=8191):
    batch = [b for b in batch if b is not None]
    if not batch: return None

    # Dynamic clamping
    text_list = [b["text_ids"].clamp(0, max_text_val) for b in batch]
    code_list = [b["codes"].clamp(0, max_code_val) for b in batch]

    # Padding value 0
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_list, batch_first=True, padding_value=0)
    
    conds = [b["condition"] for b in batch]
    emos = [b["emo_vec"] for b in batch]
    
    cond_stacked = torch.stack([c if c.dim() > 0 else c.unsqueeze(0) for c in conds])
    emo_stacked = torch.stack([e if e.dim() > 0 else e.unsqueeze(0) for e in emos])

    return {
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": cond_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": torch.stack([b["text_len"] for b in batch]),
        "code_lengths": torch.stack([b["code_len"] for b in batch])
    }

# ==========================================
# 2. MODEL BUILDING
# ==========================================

def build_model_smart(cfg_path: str, tokenizer_inst, base_ckpt: str, device: torch.device, 
                      case_format="lowercase", max_text_req=0, max_code_req=0,
                      resize_vocab=False, expand_text=False, expand_audio=False):
                          
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer_inst.vocab_size
    
    # 1. Vocab Resizing
    if resize_vocab and cfg.gpt.number_text_tokens != vocab_size:
        print(f"📐 Resizing Vocab: {cfg.gpt.number_text_tokens} -> {vocab_size}")
        cfg.gpt.number_text_tokens = vocab_size

    # 2. Text Context Expansion
    if expand_text:
        needed_text = max_text_req + 100
        if needed_text > cfg.gpt.max_text_tokens:
            print(f"📈 Expanding Text Context: {cfg.gpt.max_text_tokens} -> {needed_text}")
            cfg.gpt.max_text_tokens = needed_text
        
    # 3. Audio Context Expansion
    if expand_audio:
        needed_code = max_code_req + 200
        if needed_code > cfg.gpt.max_mel_tokens:
            print(f"📈 Expanding Audio Context: {cfg.gpt.max_mel_tokens} -> {needed_code}")
            cfg.gpt.max_mel_tokens = needed_code

    model = UnifiedVoice(**cfg.gpt)

    if os.path.exists(base_ckpt):
        print(f"📥 Loading base checkpoint: {base_ckpt}")
        try:
            ckpt = torch.load(base_ckpt, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: Failed to load base checkpoint at {base_ckpt}. File may be corrupted. Error: {e}")
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: Base checkpoint not found at {base_ckpt}. Cannot perform fine-tuning.")
            
    new_state_dict = model.state_dict()
    filtered = {}
        
    for k, v in state_dict.items():
        new_k = k.replace(".base_layer.", ".")
        if new_k not in new_state_dict: continue

        target_shape = new_state_dict[new_k].shape
        
        if v.shape == target_shape:
            filtered[new_k] = v
            continue
        
        try:
            if len(v.shape) == len(target_shape):
                slices = tuple(slice(0, min(ds, ts)) for ds, ts in zip(v.shape, target_shape))
                expanded_param = new_state_dict[new_k].clone()
                expanded_param[slices] = v[slices]
                filtered[new_k] = expanded_param
                print(f"   ✨ Resized layer {new_k}: {v.shape} -> {target_shape}")
            else:
                print(f"   ⚠️ Skipping {new_k} (Dimensions changed)")
        except Exception as e:
            print(f"   ⚠️ Error resizing {new_k}: {e}")

    model.load_state_dict(filtered, strict=False)
    
    return model.to(device), cfg

# ==========================================
# 3. LOSS & METRIC COMPUTATION
# ==========================================

def compute_losses_official(model, batch, device, use_duration=True, duration_dropout=0.0):
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)
    duration_free = model.speed_emb(torch.zeros_like(use_speed))
    
    if use_duration:
        duration_ctrl = model.get_duration_embeddings(code_lengths)
        if duration_dropout > 0.0:
            drop_mask = torch.rand(code_lengths.size(0), device=device) < duration_dropout
            if drop_mask.any():
                duration_ctrl = torch.where(drop_mask.unsqueeze(1), duration_free, duration_ctrl)
    else:
        duration_ctrl = model.speed_emb(torch.ones_like(use_speed))

    if emo_vec.dim() == 2: emo_vec = emo_vec.unsqueeze(1)
    
    conds = torch.cat(
        (condition + emo_vec, duration_ctrl.unsqueeze(1), duration_free.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)
    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    text_mask = torch.arange(text_targets.size(1), device=device).unsqueeze(0) < (text_lengths + 1).unsqueeze(1)
    mel_mask = torch.arange(mel_targets.size(1), device=device).unsqueeze(0) < (code_lengths + 1).unsqueeze(1)

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    return text_loss, mel_loss, metrics

def evaluate(model, loader, device, use_duration, duration_dropout):
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            # FIX: Force duration_dropout to 0.0 during evaluation for deterministic results
            text_loss, mel_loss, metrics = compute_losses_official(
                model, batch, device, use_duration=use_duration, duration_dropout=0.0
            )
            bsz = batch["text_ids"].size(0)
            totals["text_loss"] += text_loss.item() * bsz
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
            
    model.train()
    if count == 0: return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_official_ui(
    selected_project, config_path, tokenizer_path, train_manifest, val_manifest, vector_folder_name, 
    run_name, base_model_type, resume_ckpt, epochs, batch_size, lr, grad_accum, num_workers, 
    use_duration, duration_dropout, save_every, use_compile, case_format, tok_type, use_merged,
    resize_vocab, expand_text, expand_audio,
    lang, wordify, abbrev, extract
):
    global STOP_TRAINING
    STOP_TRAINING = False

    logs = []
    current_piece = _("TRAINER_VALUE_PIECE")
    
    def update_ui(msg=None, status_msg="Initializing..."):
        if msg: logs.append(msg)
        return "\n".join(logs), status_msg, current_piece

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yield update_ui(f"🚀 Starting Hybrid Trainer (Visible Accuracy)", "Startup")

    if not selected_project: 
        yield update_ui("❌ Error: No Project Selected!", "Error")
        return
    if not run_name: run_name = selected_project

    trains_root = os.path.join(core.path_base, "trains")
    run_dir = os.path.join(trains_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    yield update_ui(f"📂 Global Logs Folder: {run_dir}")
    
    # --- CHECK RESUME STATUS ---
    is_resuming = (resume_ckpt is not None and os.path.exists(resume_ckpt))
    if is_resuming:
         yield update_ui(f"♻️ RESUME MODE DETECTED. Preserving existing frozen environment.")

    vector_root_path = None
    if vector_folder_name:
        vector_root_path = os.path.join(core.path_base, "projects", selected_project, "extractions", vector_folder_name)
        if os.path.exists(vector_root_path):
            yield update_ui(f"🔗 Vectors linked to: {vector_root_path}")
            vector_root_path = Path(vector_root_path)
        else:
            yield update_ui(f"⚠️ Vector folder not found at expected path: {vector_root_path}")
            vector_root_path = None

    # --- FILE COPY LOGIC (SAFE VERSION) ---
    try:
        # 1. Config
        target_config = os.path.join(run_dir, "config_original.yaml")
        if not (is_resuming and os.path.exists(target_config)):
            shutil.copy2(config_path, target_config)
        
        # 2. Tokenizer
        frozen_tokenizer_path = os.path.join(run_dir, "bpe.model")
        
        if use_merged and not is_resuming:
            tok_dir = os.path.dirname(tokenizer_path)
            merged_path = os.path.join(tok_dir, f"{selected_project}_m_bpe.model")
            
            if os.path.exists(merged_path):
                tokenizer_path = merged_path
                yield update_ui(f"🔄 Switched target tokenizer to merged model: {merged_path}")
            else:
                yield update_ui(f"⚠️ Merged model not found at {merged_path}. Falling back to standard model.")
                
        if is_resuming and os.path.exists(frozen_tokenizer_path):
            yield update_ui(f"🔒 Resume: Using EXISTING frozen tokenizer: {frozen_tokenizer_path}")
            tokenizer_path = frozen_tokenizer_path 
        else:
            shutil.copy2(tokenizer_path, frozen_tokenizer_path) 
            tokenizer_path = frozen_tokenizer_path 
            yield update_ui(f"🔒 New Run: Frozen tokenizer copied to: {frozen_tokenizer_path}")

        # 3. Manifests (Switch to internal copies)
        target_train = os.path.join(run_dir, "train.jsonl")
        if is_resuming and os.path.exists(target_train):
             train_manifest = target_train
             yield update_ui(f"📜 Resume: Using frozen train manifest")
        else:
             shutil.copy2(train_manifest, target_train)
             train_manifest = target_train

        target_val = os.path.join(run_dir, "val.jsonl")
        if is_resuming and os.path.exists(target_val):
             val_manifest = target_val
             yield update_ui(f"📜 Resume: Using frozen val manifest")
        else:
             shutil.copy2(val_manifest, target_val)
             val_manifest = target_val
        
        # 4. Extraction Config
        if vector_root_path:
            ext_conf_src = vector_root_path / "extraction.config"
            if ext_conf_src.exists():
                shutil.copy2(ext_conf_src, os.path.join(run_dir, "extraction.config"))
        
    except Exception as e: 
        yield update_ui(f"⚠️ Warning during file copy: {e}")

    writer = SummaryWriter(log_dir=run_dir)
    yield update_ui("⏳ Parsing Datasets...")
    
    if tok_type == "indextts":
        yield update_ui(f"⚙️ Initializing IndexTTS Tokenizer [Lang: {lang} | Wordify: {wordify} | Abbrev: {abbrev} | Extract: {extract}]")
        norm = MultilingualNormalizer(lang=lang, wordify=wordify, abbreviations=abbrev, extract=extract)
        tokenizer_inst = TextTokenizer(vocab_file=tokenizer_path, normalizer=norm)
    else:
        yield update_ui(f"⚙️ Initializing ITTS-TR Tokenizer (GenericSpiceTokenizer) [Case: {case_format} | Wordify: {wordify} | Abbrev: {abbrev} | Extract: {extract}]")
        norm = MultilingualNormalizer(lang=lang, wordify=wordify, abbreviations=abbrev, extract=extract)
        tokenizer_inst = GenericSpiceTokenizer(vocab_file=tokenizer_path, normalizer=norm)
    
    train_ds = IndexTTSDataset(train_manifest, vector_root=vector_root_path)
    val_ds = IndexTTSDataset(val_manifest, vector_root=vector_root_path)
    
    if len(train_ds) == 0: 
        yield update_ui("❌ Train dataset is empty.", "Error")
        return

    yield update_ui(f"✅ Datasets Ready | Train: {len(train_ds)} | Val: {len(val_ds)}")

    max_text = max(train_ds.max_text_found, val_ds.max_text_found)
    max_code = max(train_ds.max_code_found, val_ds.max_code_found)
    
    yield update_ui("🏗️ Building Model & Resizing Vocabulary...")
    
    # --- MODEL ROUTING LOGIC ---
    models_dir = core.models_directory()
    if base_model_type == "blank":
        resolved_base_ckpt = os.path.join(models_dir, "blank_model.pth")
        yield update_ui(f"🧠 Using blank untrained model: {resolved_base_ckpt}")
    else:
        resolved_base_ckpt = os.path.join(models_dir, "official_model.pth")
        if not os.path.exists(resolved_base_ckpt):
            resolved_base_ckpt = os.path.join(core.path_base, "indextts", "checkpoints", "gpt.pth")
            yield update_ui(f"⚠️ Notice: Resized official model not found. Falling back to base checkpoint: {resolved_base_ckpt}")
        else:
            yield update_ui(f"🧠 Using resized official model: {resolved_base_ckpt}")
            
    try:
        model, updated_cfg = build_model_smart(
            config_path, tokenizer_inst, resolved_base_ckpt, device, 
            max_text_req=max_text, max_code_req=max_code,
            resize_vocab=resize_vocab, expand_text=expand_text, expand_audio=expand_audio
        )
        if use_compile:
            yield update_ui("⚙️ Compiling model with torch.compile (this may take a few minutes)...")
            model = torch.compile(model)
        yield update_ui("✅ Model Built Successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield update_ui(f"❌ Model Build Failed: {e}", "Error")
        return

    new_config_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(updated_cfg, new_config_path)

    # --- DYNAMIC CLAMPING ---
    vocab_limit = updated_cfg.gpt.number_text_tokens - 1
    #code_limit = getattr(updated_cfg.gpt, "number_mel_tokens", 8192) - 1
    code_limit = 8191
    
    collate_fn = partial(collate_batch, max_text_val=vocab_limit, max_code_val=code_limit)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=int(num_workers),
        pin_memory=True,
        prefetch_factor=4 if int(num_workers) > 0 else None,
        persistent_workers=True if int(num_workers) > 0 else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=int(num_workers),
        pin_memory=True,
        prefetch_factor=4 if int(num_workers) > 0 else None,
        persistent_workers=True if int(num_workers) > 0 else False
    )

    yield update_ui("⚙️ Initializing Optimizer & Scheduler...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader) // grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    # Resume Logic
    start_epoch = 0
    global_step = 0
    if is_resuming:
        yield update_ui(f"♻️ Resuming from: {resume_ckpt}")
        try:
            ckpt = torch.load(resume_ckpt, map_location=device)
            if "model" in ckpt: model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and scaler: scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("step", 0)
            yield update_ui(f"✅ Resumed at Epoch {start_epoch+1}")
        except Exception as e:
            yield update_ui(f"⚠️ Resume Failed: {e}")
            return

    model.train()
    start_time = time.time()
    yield update_ui("🎬 Training Loop Starting...")
    
    total_batches = len(train_loader)
    
    # Track stats
    current_acc = 0.0
    current_mel_loss = 0.0

    for epoch in range(start_epoch, epochs):
        if STOP_TRAINING: break
        yield update_ui(f"\n🌀 Epoch {epoch + 1}/{epochs}")
        
        # Reset accumulation for the new epoch to avoid phantom gradients from previous epoch
        optimizer.zero_grad() 
        epoch_valid_steps = 0 
        
        for i, batch in enumerate(train_loader):
            # --- UPDATED STATUS TEXT ---
            steps_left_in_epoch = (total_batches - i) // grad_accum
            status_text = (
                f"🔄 Ep {epoch + 1}/{epochs} ({steps_left_in_epoch} steps left) | "
                f"Step {global_step} | Acc: {current_acc:.3f} | "
                f"Val in {int(save_every) - (global_step % int(save_every))} steps"
            )
            
            if i % 2 == 0:
                # --- DECODE TOKENS FOR UI ---
                if batch is not None:
                    # Get the text IDs of the first item in the batch, ignoring padding (0)
                    valid_ids = [tid for tid in batch["text_ids"][0].cpu().tolist() if tid != 0]
                    # Convert to string tokens
                    tokens = tokenizer_inst.convert_ids_to_tokens(valid_ids)
                    # Format as "ID Token"
                    current_piece = " ".join([f"{tid} {tok}" for tid, tok in zip(valid_ids, tokens)])
                    if len(current_piece) > 200: 
                        current_piece = current_piece[:200] + "..."
                        
                yield update_ui(None, status_text)

            if STOP_TRAINING:
                ckpt_path = os.path.join(run_dir, "gpt_interrupted.pth")
                torch.save({
                    "model": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": global_step,
                    "epoch": epoch
                }, ckpt_path)
                yield update_ui(f"🛑 Training Stopped by User. Saved: {ckpt_path}", "Stopped")
                return

            if batch is None: continue
            
            with torch.cuda.amp.autocast():
                text_loss, mel_loss, metrics = compute_losses_official(
                    model, batch, device, use_duration=use_duration, duration_dropout=duration_dropout
                )
                loss = (0.2 * text_loss) + (0.8 * mel_loss)
                # Normalize by accumulation steps
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            epoch_valid_steps += 1
            
            # Update UI Stats
            current_acc = metrics["mel_top1"]
            current_mel_loss = mel_loss.item()

            if epoch_valid_steps % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 20 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                current_lr = scheduler.get_last_lr()[0]
                
                # Tensorboard
                writer.add_scalar("Train/Loss_Total", loss.item() * grad_accum, global_step)
                writer.add_scalar("Train/Loss_Text", text_loss.item(), global_step)
                writer.add_scalar("Train/Loss_Mel", mel_loss.item(), global_step)
                writer.add_scalar("Train/Mel_Top1", metrics["mel_top1"], global_step)
                writer.add_scalar("Train/LR", current_lr, global_step)

                # Log to scrolling text
                if global_step % 10 == 0:
                    elapsed = int(time.time() - start_time)
                    elapsed_str = str(datetime.timedelta(seconds=elapsed))
                    log_msg = f"[{elapsed_str}] Ep {epoch+1} | Step {global_step} | T-Loss: {text_loss.item():.3f} | M-Loss: {mel_loss.item():.3f} | Acc: {metrics['mel_top1']:.3f} | LR: {current_lr:.2e}"
                    yield update_ui(log_msg, status_text)

                if global_step > 0 and global_step % int(save_every) == 0:
                    # Validate
                    yield update_ui(f"🧪 Running Validation...", status_text)
                    val_metrics = evaluate(model, val_loader, device, use_duration, duration_dropout)
                    writer.add_scalar("Val/Loss_Text", val_metrics["text_loss"], global_step)
                    writer.add_scalar("Val/Loss_Mel", val_metrics["mel_loss"], global_step)
                    writer.add_scalar("Val/Mel_Top1", val_metrics["mel_top1"], global_step)
                    
                    val_msg = f"   🧪 VAL RESULT: T-Loss: {val_metrics['text_loss']:.3f} | M-Loss: {val_metrics['mel_loss']:.3f} | Acc: {val_metrics['mel_top1']:.3f}"
                    yield update_ui(val_msg, status_text)
                    
                    val_log_path = os.path.join(run_dir, "val.log")
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{current_time}] Epoch: {epoch+1:02d} | Step: {global_step:05d} | T-Loss: {val_metrics['text_loss']:.4f} | M-Loss: {val_metrics['mel_loss']:.4f} | M-TopL: {val_metrics['mel_top1']:.4f}\n"
                    with open(val_log_path, "a", encoding="utf-8") as f:
                        f.write(log_entry)

                    # Save with gpt_ prefix
                    ckpt_path = os.path.join(run_dir, f"gpt_step_{global_step}.pth")
                    torch.save({
                        "model": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": global_step,
                        "epoch": epoch
                    }, ckpt_path)
                    yield update_ui(f"💾 Saved Step: gpt_step_{global_step}.pth", status_text)
            
        latest_path = os.path.join(run_dir, "gpt_latest.pth")
        torch.save({
            "model": model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": global_step,
            "epoch": epoch + 1
        }, latest_path)
        yield update_ui(f"💾 Saved Epoch {epoch+1}", status_text)

    writer.close()
    yield update_ui("🎉 Training Complete!", "Done")

# ==========================================
# 5. UI CONSTRUCTION
# ==========================================

def list_available_projects():
    p_root = os.path.join(core.path_base, "projects")
    if not os.path.exists(p_root): return []
    return sorted([d for d in os.listdir(p_root) if os.path.isdir(os.path.join(p_root, d))])

def auto_discover_project_files(project_name):
    if not project_name:
        return "", "", "", "", "", "❌ No project selected", False, "itts-tr", "lowercase", "tr", True, False, False

    proj_dir = os.path.join(core.path_base, "projects", project_name)
    config_path = os.path.join(proj_dir, "configs", "config.yaml")
    
    if not os.path.exists(config_path):
        return "", "", "", "", "", f"❌ Config not found in {config_path}", False, "itts-tr", "lowercase", "tr", True, False, False

    ext_dir = os.path.join(proj_dir, "extractions")
    train_manifest = ""
    val_manifest = ""
    dataset_name = ""
    
    if os.path.exists(ext_dir):
        train_files = [f for f in os.listdir(ext_dir) if f.endswith("_train.jsonl")]
        if train_files:
            train_filename = train_files[0]
            dataset_name = train_filename.replace("_train.jsonl", "")
            train_manifest = os.path.join(ext_dir, train_filename)
            val_manifest = os.path.join(ext_dir, f"{dataset_name}_val.jsonl")
    
    if not train_manifest:
        return config_path, "", "", "", "", "❌ No *_train.jsonl found in extractions/", False, "itts-tr", "lowercase", "tr", True, False, False

    # --- SMART DISCOVERY: CHECK MAIN config.yaml ---
    tok_path = ""
    vector_folder = dataset_name 
    
    # Defaults
    is_merged = False
    tok_type = "itts-tr"
    c_format = "lowercase"
    lang = "tr"
    wordify = True
    abbrev = False
    extract = False
    
    try:
        from omegaconf import OmegaConf
        conf_data = OmegaConf.load(config_path)
        conf_dict = OmegaConf.to_container(conf_data, resolve=True)
        
        # 1. Fetch Normalizer settings from config.yaml
        if "tokenizer" in conf_dict:
            tok_cfg = conf_dict["tokenizer"]
            tok_type = tok_cfg.get("tokenizer_type", "itts-tr")
            c_format = tok_cfg.get("case_format", "lowercase")
            lang = tok_cfg.get("language", "tr")
            wordify = tok_cfg.get("wordify", True)
            abbrev = tok_cfg.get("abbreviations", False)
            extract = tok_cfg.get("extract", False)

        # 2. Fetch EXACT Tokenizer Path from extraction.config
        ext_config_path = os.path.join(ext_dir, dataset_name, "extraction.config")
        if os.path.exists(ext_config_path):
            with open(ext_config_path, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
                
            raw_tok_path = ext_data.get("tokenizer_path", "")
            is_merged = ext_data.get("is_merged_model", False)
            
            if raw_tok_path:
                abs_tok_path = os.path.join(core.path_base, raw_tok_path)
                if os.path.exists(abs_tok_path):
                    tok_path = abs_tok_path
                elif os.path.exists(raw_tok_path):
                    tok_path = raw_tok_path
                
    except Exception as e:
        print(f"[Smart Discovery] ⚠️ Error reading configs: {e}")
        
    if not tok_path:
        return config_path, "", "", "", "", "❌ No .model file found", False, tok_type, c_format, lang, wordify, abbrev, extract

    status_msg = (
        f"✅ Found Config\n"
        f"✅ Found Tokenizer: {os.path.basename(tok_path)}\n"
        f"✅ Found Dataset: {dataset_name}\n"
        f"ℹ️ Settings: {tok_type} | {c_format} | lang: {lang} | wordify: {wordify} | abbrev: {abbrev}"
    )

    return config_path, tok_path, train_manifest, val_manifest, vector_folder, status_msg, is_merged, tok_type, c_format, lang, wordify, abbrev, extract

def resume_official_ui(proj, config, tokenizer, train, val, vec, name, base_model_type, _, epochs, bs, lr, accum, workers, dur, drop, save, use_compile, case_format, tok_type, use_merged, resize_vocab, expand_text, expand_audio, lang, wordify, abbrev, extract):
    if not name.strip():
        yield "❌ Run Name is required to resume. Please enter the folder name.", "Error", "Piece:"
        return
    
    trains_root = os.path.join(core.path_base, "trains")
    run_dir = os.path.join(trains_root, name)

    interrupted_path = os.path.join(run_dir, "gpt_interrupted.pth")
    latest_path = os.path.join(run_dir, "gpt_latest.pth")
    
    # Fallback checks for old naming
    if not os.path.exists(interrupted_path) and os.path.exists(os.path.join(run_dir, "interrupted.pth")):
         interrupted_path = os.path.join(run_dir, "interrupted.pth")
    
    if not os.path.exists(latest_path) and os.path.exists(os.path.join(run_dir, "latest.pth")):
         latest_path = os.path.join(run_dir, "latest.pth")
    
    if os.path.exists(interrupted_path):
        resume_path = interrupted_path
        yield from train_official_ui(proj, config, tokenizer, train, val, vec, name, base_model_type, resume_path, epochs, bs, lr, accum, workers, dur, drop, save, use_compile, case_format, tok_type, use_merged, resize_vocab, expand_text, expand_audio, lang, wordify, abbrev, extract)
    elif os.path.exists(latest_path):
        resume_path = latest_path
        yield from train_official_ui(proj, config, tokenizer, train, val, vec, name, base_model_type, resume_path, epochs, bs, lr, accum, workers, dur, drop, save, use_compile, case_format, tok_type, use_merged, resize_vocab, expand_text, expand_audio, lang, wordify, abbrev, extract)
    else:
        yield f"❌ Could not find any checkpoint (gpt_interrupted.pth or gpt_latest.pth) in: {run_dir}", "Error"
    
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    with gr.Blocks() as demo:
        gr.Markdown(_("TRAINER_HEADER"))
        gr.Markdown(_("TRAINER_DESC"))

        with gr.Row():
            with gr.Column(scale=1):
                current_projects = list_available_projects()
                project_dd = gr.Dropdown(
                    label=_("TRAINER_LABEL_SELECT_PROJECT"), 
                    choices=list_available_projects(), 
                    value=current_projects[0] if current_projects else None,
                    interactive=True
                )
                refresh_btn = gr.Button(_("TRAINER_BTN_SCAN_PROJECTS"), size="lg")
                with gr.Row():
                    use_duration = gr.Checkbox(label=_("TRAINER_CHK_USE_DURATION"), value=True)
                    use_merged = gr.Checkbox(label=_("TRAINER_CHK_USE_MERGED"), value=False)
                    use_compile = gr.Checkbox(label=_("TRAINER_CHK_USE_COMPILE"), value=False)
            with gr.Column(scale=1):
                with gr.Row():
                    tok_type_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_TTYPE"),
                        choices=["itts-tr", "indextts"],
                        value="itts-tr",
                        interactive=True
                    )
                with gr.Row(visible=True) as case_row:
                    case_format = gr.Dropdown(
                        label=_("TRAINER_LABEL_CASE_FORMAT"),
                        choices=["lowercase", "uppercase"],
                        value="lowercase",
                        interactive=True
                    )          
        with gr.Row():
            run_name = gr.Textbox(label=_("TRAINER_LABEL_RUN_NAME"), value="", placeholder=_("TRAINER_PLACEHOLDER_RUN_NAME"))
            base_model_dd = gr.Dropdown(label=_("TRAINER_LABEL_BASE_MODEL"), choices=["official", "blank"], value="official", interactive=True)

        with gr.Row(visible=True): 
            config = gr.Textbox(label=_("TRAINER_LABEL_CONFIG"), interactive=False)
            tokenizer = gr.Textbox(label=_("TRAINER_LABEL_TOKENIZER"), interactive=False)
            
        
        with gr.Row(visible=True):
            vector_folder = gr.Textbox(label=_("TRAINER_LABEL_VECTOR_FOLDER"), interactive=False)
            train_manifest = gr.Textbox(label=_("TRAINER_LABEL_TRAIN_MANIFEST"), interactive=False)
            val_manifest = gr.Textbox(label=_("TRAINER_LABEL_VAL_MANIFEST"), interactive=False)
             
        tok_type_dd.change(
            fn=lambda t: gr.update(visible=(t == "itts-tr")),
            inputs=[tok_type_dd],
            outputs=[case_row]
        )
        
        with gr.Row():
            epochs = gr.Slider(1, 100, 10, label=_("TRAINER_SLIDER_EPOCHS"))
            bs = gr.Slider(1, 32, 8, label=_("TRAINER_SLIDER_BATCH_SIZE"))
            grad_accum = gr.Slider(1, 16, 4, label=_("TRAINER_SLIDER_GRAD_ACCUM"))
            lr = gr.Slider(1e-6, 1e-3, 2e-5, label=_("TRAINER_SLIDER_LR"))
        
        with gr.Row():
            save_every = gr.Slider(100, 5000, 1000, step=100, label=_("TRAINER_SLIDER_SAVE_EVERY"))
            num_workers_slider = gr.Slider(0, 32, 8, step=1, label=_("TRAINER_SLIDER_WORKERS"))
            duration_dropout = gr.Slider(0.0, 1.0, 0.3, label=_("TRAINER_SLIDER_DROPOUT"))
            
            resume_ckpt = gr.Textbox(visible=False)
            
        # --- 3. ADVANCED OPTIONS (Hidden to reduce clutter) ---
        with gr.Accordion(_("TRAINER_ACC_ADVANCED"), open=False, elem_classes="wui-accordion"):
            with gr.Row():
                resize_vocab = gr.Checkbox(label=_("TRAINER_CHK_RESIZE_VOCAB"), value=False)
                expand_text = gr.Checkbox(label=_("TRAINER_CHK_EXPAND_TEXT"), value=True)
                expand_audio = gr.Checkbox(label=_("TRAINER_CHK_EXPAND_AUDIO"), value=True)
                
        # --- HIDDEN STATES FOR CONFIG VARIABLES ---
        state_lang = gr.State("tr")
        state_wordify = gr.State(True)
        state_abbrev = gr.State(False)
        state_extract = gr.State(False)
    
        start_btn = gr.Button(_("COMMON_BTN_START"), variant="primary")
        
        status_box = gr.Textbox(label=_("TRAINER_LABEL_STATUS"), value=_("COMMON_STATUS_READY"), lines=1, interactive=False)
        piece_box = gr.Textbox(label=_("TRAINER_LABEL_PIECE"), value=_("TRAINER_VALUE_PIECE"), lines=2, interactive=False) 
        logs = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=15, autoscroll=True)
               
        with gr.Row():
            stop_btn = gr.Button(_("COMMON_BTN_STOP"))
            resume_btn = gr.Button(_("COMMON_BTN_RESUME"))
            
        gr.HTML("<div style='height:16px'></div>")
        gr.Markdown(_("TRAINER_HEADER_EXPORT"))
        with gr.Row():
            ckpt_selector = gr.Dropdown(label=_("TRAINER_LABEL_CKPT_SELECT"), choices=[])
            refresh_ckpt_btn = gr.Button(_("TRAINER_BTN_REFRESH_CKPT"))
            conformer_map_check = gr.Checkbox(label=_("TRAINER_CHK_CONFORMER"), value=True)
            unwrap_btn = gr.Button(_("TRAINER_BTN_UNWRAP"), variant="stop")
        
        unwrap_status = gr.Textbox(label=_("TRAINER_LABEL_UNWRAP_STATUS"), placeholder=_("TRAINER_PLACEHOLDER_UNWRAP"), interactive=False)

        refresh_ckpt_btn.click(
            get_checkpoint_list, 
            inputs=[run_name], 
            outputs=[ckpt_selector]
        )
        
        unwrap_btn.click(
            unwrap_and_save_handler, 
            inputs=[run_name, ckpt_selector, conformer_map_check],
            outputs=[unwrap_status]
        )
        
        def toggle_merged_ui(is_merged, current_path, proj_name):
            if not current_path or not proj_name: 
                return current_path
            
            tok_dir = os.path.dirname(current_path)
            if is_merged:
                # Switch to merged
                return os.path.join(tok_dir, f"{proj_name}_m_bpe.model")
            else:
                # Switch back to standard
                return os.path.join(tok_dir, f"{proj_name}_bpe.model")
                
        use_merged.change(
            fn=toggle_merged_ui,
            inputs=[use_merged, tokenizer, project_dd],
            outputs=[tokenizer]
        )

        def on_project_select(p_name):
            conf, tok, tr, val, vec, stat, is_merged, tok_type, c_format, lang, wordify, abbrev, extract = auto_discover_project_files(p_name)
            
            if is_merged and tok:
                tok_dir = os.path.dirname(tok)
                tok = os.path.join(tok_dir, f"{p_name}_m_bpe.model")
            
            return (
                conf, tok, tr, val, vec, stat, p_name, 
                gr.update(value=is_merged), 
                gr.update(value=tok_type), 
                gr.update(value=c_format), 
                lang, wordify, abbrev, extract
            )

        project_dd.change(
            on_project_select, 
            inputs=[project_dd], 
            outputs=[config, tokenizer, train_manifest, val_manifest, vector_folder, logs, run_name, use_merged, tok_type_dd, case_format, state_lang, state_wordify, state_abbrev, state_extract]
        )
        
        demo.load(
            fn=on_project_select,
            inputs=[project_dd],
            outputs=[config, tokenizer, train_manifest, val_manifest, vector_folder, logs, run_name, use_merged, tok_type_dd, case_format, state_lang, state_wordify, state_abbrev, state_extract]
        )
        
        refresh_btn.click(lambda: gr.Dropdown(choices=list_available_projects()), outputs=[project_dd])
        
        inputs = [
            project_dd, config, tokenizer, train_manifest, val_manifest, vector_folder, 
            run_name, base_model_dd, resume_ckpt, epochs, bs, lr, grad_accum, num_workers_slider, 
            use_duration, duration_dropout, save_every, use_compile, case_format, tok_type_dd, 
            use_merged, resize_vocab, expand_text, expand_audio,
            state_lang, state_wordify, state_abbrev, state_extract
        ]
        
        # Added piece_box to outputs
        start_btn.click(train_official_ui, inputs=inputs, outputs=[logs, status_box, piece_box])
        stop_btn.click(stop_training_fn, outputs=logs)
        resume_btn.click(resume_official_ui, inputs=inputs, outputs=[logs, status_box, piece_box])

        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("trainer"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")

    return demo