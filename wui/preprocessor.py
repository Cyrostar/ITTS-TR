import gradio as gr
import os
import json
import time
import torch
import torchaudio
import numpy as np
import pandas as pd
import safetensors.torch
from huggingface_hub import hf_hub_download
from transformers import SeamlessM4TFeatureExtractor
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from indextts.utils.front import TextTokenizer
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model

from core import core
from core.core import _
from core.spice import GenericSpiceTokenizer
from core.normalizer import MultilingualNormalizer, MultilingualWordifier

# --- OPTIMIZATIONS: TF32 + Memory Management ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- GLOBAL CONTROL STATE ---
PROCESS_CONTROL = {
    "stop": False
}

def stop_process():
    PROCESS_CONTROL["stop"] = True
    return "🛑 Stopping... (Waiting for current batch to finish)"

# --- HELPER FUNCTIONS ---

def list_datasets(lang):
    if not lang:
        return []
    ds_path = os.path.join(core.path_base, "datasets", lang)
    if not os.path.exists(ds_path): 
        return []
    return sorted([d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))])

def ensure_config_exists():
    project_config_path = os.path.join(core.configs_directory(), "config.yaml")
    if os.path.exists(project_config_path):
        return project_config_path, f"✅ Found Project Config: {project_config_path}"
    return None, f"❌ Error: Project-specific config.yaml not found at: {project_config_path}."

# --- DATASET ---
class AudioTextDataset(Dataset):
    def __init__(self, dataframe, wavs_dir, tokenizer, add_lang_id=True, lang_id=3, normalizer=None):
        self.df = dataframe
        self.wavs_dir = wavs_dir
        self.tokenizer = tokenizer
        self.add_lang_id = add_lang_id
        self.lang_id = lang_id
        self.normalizer = normalizer

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        filename = str(row['filename'])
        text = str(row['text'])
        speaker = str(row['speaker'])
        
        if self.normalizer:
            text = self.normalizer.normalize(text)
               
        wav_path = os.path.join(self.wavs_dir, filename)
        if not os.path.exists(wav_path):
            if not wav_path.lower().endswith(".wav"): wav_path += ".wav"
        
        if not os.path.exists(wav_path): return None

        try:
            wav, sr = torchaudio.load(wav_path)
            duration = wav.shape[-1] / sr
            if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.squeeze()
        except: return None

        text_tokens = self.tokenizer.tokenize(text)
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        
        if self.add_lang_id:
            if len(text_ids) > 0 and text_ids[0] != self.lang_id:
                text_ids = [self.lang_id] + text_ids
        
        return {
            "wav": wav, 
            "text_ids": np.array(text_ids, dtype=np.int32),
            "file_id": os.path.splitext(filename)[0],
            "speaker": speaker,    
            "duration": duration,
            "text": text,
            "wav_path": wav_path, 
            "valid": True
        }

def collate_batch(batch):
    
    batch = [x for x in batch if x is not None and x["valid"]]
    if len(batch) == 0: return None
    return {
        "wav_list": [x["wav"] for x in batch],
        "text_ids_list": [x["text_ids"] for x in batch],
        "file_ids": [x["file_id"] for x in batch],
        "speakers": [x["speaker"] for x in batch],   
        "durations": [x["duration"] for x in batch],
        "texts": [x["text"] for x in batch],
        "wav_paths": [x["wav_path"] for x in batch]
    }

class IndexTTSExtractor:
    
    def __init__(self, config_path, device="cuda", do_compile=False, base_model_type="official"):
        self.device = device
        self.cfg = OmegaConf.load(config_path)
        
        # --- 1. Feature Extractor ---
        try:
            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0", local_files_only=True)
        except:
            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        print("✅ Feature Extractor loaded successfully.")
        
        # --- 2. Semantic Model Stats ---
        stats_path = os.path.join(core.path_base, "indextts", "checkpoints", "wav2vec2bert_stats.pt")
        if not os.path.exists(stats_path):
            error_msg = f"CRITICAL ERROR: Semantic stats file not found at expected path: {stats_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(path_=stats_path)
        except Exception as e:
            error_msg = f"CRITICAL ERROR: Failed to load semantic model stats. File may be corrupted. Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
            
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        print("✅ Semantic Model loaded successfully.")
        
        # --- 3. Semantic Codec ---
        self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        try:
            from huggingface_hub import hf_hub_download
            codec_ckpt = hf_hub_download(repo_id="amphion/MaskGCT", filename="semantic_codec/model.safetensors")
            safetensors.torch.load_model(self.semantic_codec, codec_ckpt)
        except Exception as e:
            error_msg = f"CRITICAL ERROR: Failed to load semantic codec from Hugging Face cache. Error: {e}"
            print("CRITICAL ERROR: Failed to load semantic codec from Hugging Face cache.")
            raise RuntimeError(error_msg) from e           
        self.semantic_codec = self.semantic_codec.to(device).eval()
        print("✅ MaskGCT Semantic Codec loaded successfully.")

        # --- 4. GPT Model ---
        models_dir = core.models_directory()
        if base_model_type == "blank":
            gpt_path = os.path.join(models_dir, "blank_model.pth")
        else:
            gpt_path = os.path.join(models_dir, "official_model.pth")
            
        # Fallback to the original checkpoint if the resized one is missing
        if base_model_type == "official" and not os.path.exists(gpt_path):
            gpt_path = os.path.join(core.path_base, "indextts", "checkpoints", "gpt.pth")
            print(f"⚠️ Notice: Resized official model not found. Falling back to base checkpoint: {gpt_path}")
            
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        
        if os.path.exists(gpt_path):
            try:
                ckpt = torch.load(gpt_path, map_location="cpu")
                state = ckpt.get("model", ckpt)
                self.gpt.load_state_dict(state, strict=False)
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Failed to load GPT checkpoint weights. File may be corrupted. Error: {e}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            error_msg = f"CRITICAL ERROR: GPT checkpoint not found at expected path: {gpt_path}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.gpt = self.gpt.to(device).eval()
        print("✅ GPT Model loaded successfully.")

        # --- 5. Compilation Phase ---
        if do_compile:
            try:
                self.semantic_model = torch.compile(self.semantic_model)
                self.gpt = torch.compile(self.gpt)
                print("✅ Models compiled successfully via torch.compile.")
            except Exception as e: 
                print(f"⚠️ Warning: Model compilation failed. Falling back to uncompiled execution. Error: {e}")

    def process_batch(self, wav_list):
        
        try:
            return self._process_batch_unsafe(wav_list)
        except torch.cuda.OutOfMemoryError:
            n = len(wav_list)
            if n <= 1: raise 
            torch.cuda.empty_cache()
            mid = n // 2
            r1 = self.process_batch(wav_list[:mid])
            r2 = self.process_batch(wav_list[mid:])
            return (r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2])

    @torch.no_grad()
    def _process_batch_unsafe(self, wav_list):
        
        wavs_np = [w.numpy() for w in wav_list]
        inputs = self.feature_extractor(wavs_np, sampling_rate=16000, return_tensors="pt", padding=True)
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.semantic_model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True)
            feat = outputs.hidden_states[17]
            feat = (feat - self.semantic_mean) / self.semantic_std
            semantic_code, _ = self.semantic_codec.quantize(feat)
            if semantic_code.dim() == 1: semantic_code = semantic_code.unsqueeze(0)
            cond_lengths = attention_mask.sum(dim=1).long()
            feat_t = feat.transpose(1, 2)
            conditioning = self.gpt.get_conditioning(feat_t, cond_lengths)
            emo_vec = self.gpt.get_emovec(feat, cond_lengths)
            
        return (
            list(semantic_code.cpu().numpy().astype(np.int32)),
            list(conditioning.cpu().numpy().astype(np.float32)),
            list(emo_vec.cpu().numpy().astype(np.float32))
        )

def run_preprocessing_ui(
    folder_lang, dataset_name, val_split, num_workers, batch_size, 
    compile_model, use_relative, lang_id, case_format, normalize_text, 
    extract_graphemes, tok_type, vocab_type, show_live_tok, base_model_type
):
    
    logs = []
    def log(msg): 
        logs.append(msg)
        return "\n".join(logs)
        
    token_log = ""
    PROCESS_CONTROL["stop"] = False
    torch.cuda.empty_cache()

    if not dataset_name: 
        yield log("❌ Error: No dataset selected."), ""
        return

    # Use the lang_id passed from the UI
    if lang_id == "None":
        add_lang_id, selected_lang_id = False, -1
    else:
        add_lang_id = True
        lang_map = {"TR (ID-3)": 3, "EN (ID-4)": 4}
        selected_lang_id = lang_map.get(lang_id, 3)
        
    config_path, cfg_msg = ensure_config_exists()
    yield log(cfg_msg), ""
    if not config_path: return
    
    # --- AUTO-UPDATE CONFIG.YAML ---
    try:
        from omegaconf import open_dict
        cfg = OmegaConf.load(config_path)
        with open_dict(cfg):
            if "tokenizer" not in cfg or cfg.tokenizer is None:
                cfg.tokenizer = {}
            
            cfg.tokenizer.language = folder_lang
            cfg.tokenizer.tokenizer_type = tok_type
            cfg.tokenizer.vocab_type = vocab_type
            
            # Enforce uppercase for the official indextts tokenizer
            if tok_type == "itts-tr":
                cfg.tokenizer.case_format = case_format
            else:
                cfg.tokenizer.case_format = "uppercase"
                
        OmegaConf.save(cfg, config_path)
        yield log("💾 Auto-updated project config.yaml with current Tokenizer preferences."), ""
    except Exception as e:
        yield log(f"⚠️ Warning: Could not auto-update config.yaml: {str(e)}"), ""

    # Use folder_lang from the UI
    dataset_root = os.path.join(core.path_base, "datasets", folder_lang, dataset_name)
    wavs_dir = os.path.join(dataset_root, "wavs")
    metadata_path = os.path.join(dataset_root, "metadata.csv")    
    extract_root = core.extractions_directory()
    dataset_output_dir = os.path.join(extract_root, dataset_name)
    
    subdirs = {
        "codes": os.path.join(dataset_output_dir, "codes"),
        "condition": os.path.join(dataset_output_dir, "condition"),
        "emo": os.path.join(dataset_output_dir, "emo_vec"), 
        "text": os.path.join(dataset_output_dir, "text_ids"),
    }
    for p in subdirs.values(): os.makedirs(p, exist_ok=True)
    yield log(f"📂 Output Directories Ready in: {dataset_output_dir}"), ""
    
    try:
        p_name = core.project_name if core.project_name else "myproject"
        
        if vocab_type == "merged":
            sp_path = os.path.join(core.tokenizer_directory(), f"{p_name}_m_bpe.model")
        else:
            sp_path = os.path.join(core.tokenizer_directory(), f"{p_name}_bpe.model")

        if not os.path.exists(sp_path): 
            yield log(f"⚠️ Tokenizer not found at: {sp_path}"), ""
            return
   
        tokenizer_name = os.path.basename(sp_path)
   
    except Exception as e: 
        yield log(f"❌ Tokenizer Error: {e}"), ""
        return
        
    config_file_path = os.path.join(dataset_output_dir, "extraction.config")
    
    # Use use_relative from the UI
    final_tokenizer_path = os.path.relpath(sp_path, start=core.path_base).replace("\\", "/") if use_relative else sp_path
    
    current_run_config = {
        "dataset_name": dataset_name, 
        "val_split_percent": val_split, 
        "num_workers": num_workers, 
        "batch_size": batch_size, 
        "do_compile": compile_model, 
        "tokenizer_path": final_tokenizer_path, 
        "add_lang_id": add_lang_id,
        "is_merged_model": (vocab_type == "merged"),
        "lang_id": selected_lang_id, 
        "base_model": base_model_type,
        "started_at": str(time.time())
    }

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
            if os.path.basename(saved_config.get("tokenizer_path", "")) != tokenizer_name:
                yield log(f"⛔ SAFETY STOP: Tokenizer Mismatch! Started with: {os.path.basename(saved_config.get('tokenizer_path', ''))}"), ""
                return 
            yield log(f"🔒 Verified: Resuming task using extraction.config\n   ℹ️  Using Tokenizer: {tokenizer_name}"), ""
        except: yield log(f"⚠️ Warning: Could not read extraction.config. Proceeding..."), ""
    else:
        try:
            with open(config_file_path, "w", encoding="utf-8") as f: json.dump(current_run_config, f, indent=4)
            yield log(f"💾 Configuration saved to: extraction.config"), ""
        except: yield log(f"⚠️ Warning: Could not save config."), ""
            
    yield log(f"🗣️ Language ID Injection: {'ENABLED' if add_lang_id else 'DISABLED'}"), ""

    try:
        if tok_type == "indextts":
            yield log(f"⚙️ Initializing IndexTTS Tokenizer with dedicated Multilingual Normalizer (Lang: {folder_lang}, Wordify: True)"), ""
            # IndexTTS forces its own normalizer inside its class
            indextts_norm = MultilingualNormalizer(lang=folder_lang, wordify=True)
            tokenizer = TextTokenizer(vocab_file=sp_path, normalizer=indextts_norm)
            norm = None
        else:            
            yield log(f"⚙️ Initializing ITTS-TR Tokenizer (GenericSpiceTokenizer) [Lang: {folder_lang}, Case: {case_format}, Normalized: {normalize_text}, Wordify: True]"), ""
            tokenizer = GenericSpiceTokenizer(vocab_file=sp_path, normalizer=None, cjk=False)
            if normalize_text:
                norm = MultilingualNormalizer(lang=folder_lang, upper=(case_format == "uppercase"), extract=extract_graphemes, wordify=True)
            else:
                norm = None

        # Use compile_model from the UI
        extractor = IndexTTSExtractor(config_path, device="cuda", do_compile=compile_model, base_model_type=base_model_type)
        yield log("✅ Models Loaded."), ""
        df = pd.read_csv(metadata_path, sep="|", names=["lang", "dataset", "filename", "speaker", "text"], dtype=str).dropna(subset=['filename', 'text'])
        
    except Exception as e: 
        yield log(f"❌ Initialization Error: {e}"), ""
        return

    # Use extract_graphemes from the UI
    ds = AudioTextDataset(df, wavs_dir, tokenizer, add_lang_id=add_lang_id, lang_id=selected_lang_id, normalizer=norm)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), collate_fn=collate_batch, pin_memory=True)

    total_files = len(df)
    yield log(f"📂 Processing {total_files} files (Batch: {batch_size})..."), ""

    train_path, val_path = os.path.join(extract_root, f"{dataset_name}_train.jsonl"), os.path.join(extract_root, f"{dataset_name}_val.jsonl")
    if not os.path.exists(train_path): open(train_path, 'w').close()
    if not os.path.exists(val_path): open(val_path, 'w').close()
    
    existing_ids = set()
    for p in [train_path, val_path]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try: existing_ids.add(json.loads(line)['id'])
                    except: pass
                
    yield log(f"🔄 Resuming... Found {len(existing_ids)} existing entries."), ""
    
    # Use val_split from the UI
    split_idx = int(total_files * (1 - (val_split / 100)))
    processed_count, train_buffer, val_buffer = 0, [], []

    for i, batch_data in enumerate(loader):
        if PROCESS_CONTROL["stop"]:
            yield log("\n🛑 Processing Stopped by User. Saving buffer..."), token_log
            break
        if batch_data is None: continue
        
        if show_live_tok:
            token_display = []
            for b_idx in range(len(batch_data["file_ids"])):
                raw_text = batch_data['texts'][b_idx]
                tokens = tokenizer.tokenize(raw_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                paired_output = [f"({i}) '{t}'" for i, t in zip(ids, tokens)]
                token_display.append(f"[{batch_data['file_ids'][b_idx]}] ➔ [{', '.join(paired_output)}]")
            token_log = "--- CURRENT BATCH TOKENIZATION ---\n" + "\n".join(token_display)
        else:
            token_log = "⚡ Live Tokenizer Breakdown is DISABLED (Saving CPU power)"
        
        needs_processing = any(fid not in existing_ids for fid in batch_data["file_ids"])
        if not needs_processing:
            processed_count += len(batch_data["file_ids"])
            if i % 100 == 0: yield log(f"⏩ Skipped batch {i} (already done)..."), token_log
            continue

        try:
            codes_batch, cond_batch, emo_batch = extractor.process_batch(batch_data["wav_list"])
            for b_idx, file_id in enumerate(batch_data["file_ids"]):
                if file_id in existing_ids: continue 
                
                text_ids = batch_data["text_ids_list"][b_idx]
                p_text, p_code, p_cond, p_emo = os.path.join(subdirs["text"], f"{file_id}.npy"), os.path.join(subdirs["codes"], f"{file_id}.npy"), os.path.join(subdirs["condition"], f"{file_id}.npy"), os.path.join(subdirs["emo"], f"{file_id}.npy")
                np.save(p_text, text_ids); np.save(p_code, codes_batch[b_idx]); np.save(p_cond, cond_batch[b_idx]); np.save(p_emo, emo_batch[b_idx])

                entry = {
                    "id": file_id, "audio_path": os.path.relpath(batch_data["wav_paths"][b_idx], start=core.path_base).replace("\\", "/"),
                    "text": batch_data["texts"][b_idx], "speaker": batch_data["speakers"][b_idx], "language": "tr", "duration": float(batch_data["durations"][b_idx]),
                    "text_ids_path": os.path.relpath(p_text, start=core.project_path).replace("\\", "/"), "codes_path": os.path.relpath(p_code, start=core.project_path).replace("\\", "/"),
                    "condition_path": os.path.relpath(p_cond, start=core.project_path).replace("\\", "/"), "emo_vec_path": os.path.relpath(p_emo, start=core.project_path).replace("\\", "/"),
                    "text_len": int(text_ids.size), "code_len": int(codes_batch[b_idx].size), "condition_len": int(cond_batch[b_idx].shape[0])
                }
                if processed_count < split_idx: train_buffer.append(entry)
                else: val_buffer.append(entry)
                existing_ids.add(file_id); processed_count += 1
            
            if i % 5 == 0:
                for buf, path in [(train_buffer, train_path), (val_buffer, val_path)]:
                    if buf:
                        with open(path, 'a', encoding='utf-8') as f:
                            for e in buf: f.write(json.dumps(e, ensure_ascii=False) + '\n')
                train_buffer, val_buffer = [], []
                torch.cuda.empty_cache()
                yield log(f"🚀 Batch {i} saved. (Total: {processed_count})"), token_log
        except: continue

    for buf, path in [(train_buffer, train_path), (val_buffer, val_path)]:
        if buf:
            with open(path, 'a', encoding='utf-8') as f:
                for e in buf: f.write(json.dumps(e, ensure_ascii=False) + '\n')

    yield log(f"✅ DONE! \nManifests:\n{train_path}\n{val_path}"), token_log

# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    lang_options = core.language_list()
    
    with gr.Blocks() as demo:
        gr.Markdown(_("PREPROCESSOR_HEADER"))
        gr.Markdown(_("PREPROCESSOR_DESC"))
               
        # --- 1. DATASET & LANGUAGE CARD ---
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                gr.Markdown(_("PREPROCESSOR_HEADER_DATA_SOURCE"))
                with gr.Row():
                    dataset_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_TARGET_DATASET"), 
                        choices=list_datasets("tr"), 
                        value=None, 
                        scale=4
                    )
                    folder_lang_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_FOLDER_LANG"),
                        choices=lang_options,
                        value="tr",
                        scale=1
                    )
                    base_model_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_BASE_MODEL"),
                        choices=["official", "blank"],
                        value="official",
                        interactive=True,
                        visible=False
                    )
                    vocab_type_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_VOCAB_TYPE"), 
                        choices=["trained", "merged"],
                        value="trained",
                        interactive=True
                    )
                with gr.Row():
                    refresh_btn = gr.Button(_("COMMON_BTN_REFRESH"))
                with gr.Row():    
                    split_slider = gr.Slider(
                        1, 20, 5, step=1, 
                        label=_("PREPROCESSOR_SLIDER_VAL_SPLIT"),
                        info=_("PREPROCESSOR_INFO_VAL_SPLIT")
                    )
    
            with gr.Column(scale=1):
                with gr.Row():            
                    gr.Markdown(_("PREPROCESSOR_HEADER_INJECT_MARKER"))
                with gr.Row():     
                    lang_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_LANG_ID"), 
                        choices=["None", "TR (ID-3)", "EN (ID-4)"], 
                        value="None",
                        show_label=False,
                        scale=1,
                        interactive=True
                    )
                with gr.Row():
                    tok_type_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_TTYPE"),
                        choices=["itts-tr", "indextts"],
                        value="itts-tr",
                        interactive=True
                    )
                with gr.Row(visible=True) as case_row:
                    case_dd = gr.Dropdown(
                        label=_("PREPROCESSOR_LABEL_CASE_FORMAT"),
                        choices=["lowercase", "uppercase"],
                        value="lowercase",
                        interactive=True
                    )
                    
                tok_type_dd.change(
                    fn=lambda t: gr.update(visible=(t == "itts-tr")),
                    inputs=[tok_type_dd],
                    outputs=[case_row]
                )
                    
        # --- 2. PERFORMANCE SETTINGS ---
        with gr.Row(variant="panel"):
            with gr.Column():
                batch_slider = gr.Slider(
                    1, 64, 8, step=1, 
                    label=_("PREPROCESSOR_SLIDER_BATCH"),
                    info=_("PREPROCESSOR_INFO_BATCH")
                )
            with gr.Column():
                worker_slider = gr.Slider(
                    0, 32, 8, step=1, 
                    label=_("PREPROCESSOR_SLIDER_WORKERS"),
                    info=_("PREPROCESSOR_INFO_WORKERS")
                )
    
        # --- 3. ADVANCED OPTIONS (Hidden to reduce clutter) ---
        with gr.Accordion(_("PREPROCESSOR_ACC_ADVANCED"), open=False, elem_classes="wui-accordion"):
            with gr.Row():
                relative_chk = gr.Checkbox(
                    label=_("PREPROCESSOR_CHK_RELATIVE"), 
                    value=True, 
                    info=_("PREPROCESSOR_INFO_RELATIVE")
                )
                compile_chk = gr.Checkbox(
                    label=_("PREPROCESSOR_CHK_COMPILE"), 
                    value=False,
                    info=_("PREPROCESSOR_INFO_COMPILE")
                )
                show_tok_chk = gr.Checkbox(
                    label=_("PREPROCESSOR_CHK_SHOW_TOK"),
                    value=True,
                    info=_("PREPROCESSOR_INFO_SHOW_TOK")
                )
                normalize_chk = gr.Checkbox(
                    label=_("PREPROCESSOR_CHK_NORMALIZE"),
                    value=True,
                    info=_("PREPROCESSOR_INFO_NORMALIZE")
                )
                extract_chk = gr.Checkbox(
                    label=_("PREPROCESSOR_CHK_EXTRACT"), 
                    value=False,
                    info=_("PREPROCESSOR_INFO_EXTRACT")
                )

        
        # --- 4. ACTION BAR ---
        with gr.Row():
            run_btn = gr.Button(_("COMMON_BTN_START"), variant="primary", scale=1, size="lg")
            stop_btn = gr.Button(_("COMMON_BTN_STOP"), variant="stop", scale=1, size="lg")
            
        log_box = gr.Textbox(
            label=_("COMMON_LABEL_LOGS"), 
            lines=15, 
            elem_id="log_window", 
            autoscroll=True
        )
        
        tokenizer_log_box = gr.Textbox(
            label=_("COMMON_LABEL_LOGS"),
            lines=10,
            placeholder=_("PREPROCESSOR_PLACEHOLDER_TOKENIZER_LOG"),
            elem_id="tokenizer_log_window",
            autoscroll=True
        )
        
        # --- EVENT LISTENERS ---
        folder_lang_dd.change(
            fn=lambda l: gr.Dropdown(choices=list_datasets(l), value=None),
            inputs=[folder_lang_dd],
            outputs=[dataset_dd]
        )
        
        run_btn.click(
            run_preprocessing_ui, 
            inputs=[
                folder_lang_dd, dataset_dd, split_slider, worker_slider, 
                batch_slider, compile_chk, relative_chk, lang_dd, 
                case_dd, normalize_chk, extract_chk, tok_type_dd, 
                vocab_type_dd, show_tok_chk, base_model_dd
            ], 
            outputs=[log_box, tokenizer_log_box]
        )
        
        stop_btn.click(fn=stop_process, inputs=None, outputs=[log_box])
        
        refresh_btn.click(
            fn=lambda l: gr.Dropdown(choices=list_datasets(l)), 
            inputs=[folder_lang_dd], 
            outputs=[dataset_dd]
        )
        
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("preprocessor"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")
        
    return demo