import gradio as gr
import os
import json
import yaml
import torch
import sentencepiece as spm
from indextts.gpt.model_v2 import UnifiedVoice
from core import core
from core.core import _

def save_config_ui(
    # Core
    number_text_tokens, max_text_tokens, max_mel_tokens, model_dim, layers, heads, sample_rate,
    # Advanced - Tokenizer
    language, tokenizer_type, vocab_type, case_format, wordify, abbreviations, extract,
    # Advanced - Dataset & Mel
    bpe_model, squeeze, n_fft, hop_length, win_length, n_mels, mel_fmin, normalize_mel,
    # Advanced - GPT
    use_mel_codes_as_input, mel_length_compression, number_mel_codes, start_mel_token, stop_mel_token, start_text_token, stop_text_token, train_solo_embeddings, condition_type,
    # Advanced - Checkpoints
    gpt_checkpoint, w2v_stat, s2mel_checkpoint, emo_matrix, spk_matrix, qwen_emo_path, vocoder_type, vocoder_name, version
):
    # This reconstructs the EXACT nested structure of your config.yaml
    config_dict = {
        "tokenizer": {
            "language": language,
            "tokenizer_type": tokenizer_type,
            "vocab_type": vocab_type,
            "case_format": case_format,
            "wordify": bool(wordify),
            "abbreviations": bool(abbreviations),
            "extract": bool(extract)
        },
        "dataset": {
            "bpe_model": bpe_model,
            "sample_rate": int(sample_rate),
            "squeeze": bool(squeeze),
            "mel": {
                "sample_rate": int(sample_rate),
                "n_fft": int(n_fft),
                "hop_length": int(hop_length),
                "win_length": int(win_length),
                "n_mels": int(n_mels),
                "mel_fmin": int(mel_fmin),
                "normalize": bool(normalize_mel)
            }
        },
        "gpt": {
            "model_dim": int(model_dim),
            "max_mel_tokens": int(max_mel_tokens),
            "max_text_tokens": int(max_text_tokens),
            "heads": int(heads),
            "use_mel_codes_as_input": bool(use_mel_codes_as_input),
            "mel_length_compression": int(mel_length_compression),
            "layers": int(layers),
            "number_text_tokens": int(number_text_tokens),
            "number_mel_codes": int(number_mel_codes),
            "start_mel_token": int(start_mel_token),
            "stop_mel_token": int(stop_mel_token),
            "start_text_token": int(start_text_token),
            "stop_text_token": int(stop_text_token),
            "train_solo_embeddings": bool(train_solo_embeddings),
            "condition_type": condition_type,
            # Hardcoded internal structures
            "condition_module": {
                "output_size": 512, "linear_units": 2048, "attention_heads": 8,
                "num_blocks": 6, "input_layer": "conv2d2", "perceiver_mult": 2
            },
            "emo_condition_module": {
                "output_size": 512, "linear_units": 1024, "attention_heads": 4,
                "num_blocks": 4, "input_layer": "conv2d2", "perceiver_mult": 2
            }
        },
        "semantic_codec": {
            "codebook_size": 8192, "hidden_size": 1024, "codebook_dim": 8,
            "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12
        },
        "s2mel": {
            "preprocess_params": {
                "sr": 22050,
                "spect_params": {
                    "n_fft": 1024, "win_length": 1024, "hop_length": 256,
                    "n_mels": 80, "fmin": 0, "fmax": "None"
                }
            },
            "dit_type": "DiT",
            "reg_loss_type": "l1",
            "style_encoder": {"dim": 192},
            "length_regulator": {
                "channels": 512, "is_discrete": False, "in_channels": 1024,
                "content_codebook_size": 2048, "sampling_ratios": [1, 1, 1, 1],
                "vector_quantize": False, "n_codebooks": 1, "quantizer_dropout": 0.0,
                "f0_condition": False, "n_f0_bins": 512
            },
            "DiT": {
                "hidden_dim": 512, "num_heads": 8, "depth": 13, "class_dropout_prob": 0.1,
                "block_size": 8192, "in_channels": 80, "style_condition": True,
                "final_layer_type": 'wavenet', "target": 'mel', "content_dim": 512,
                "content_codebook_size": 1024, "content_type": 'discrete',
                "f0_condition": False, "n_f0_bins": 512, "content_codebooks": 1,
                "is_causal": False, "long_skip_connection": True, "zero_prompt_speech_token": False,
                "time_as_token": False, "style_as_token": False, "uvit_skip_connection": True,
                "add_resblock_in_transformer": False
            },
            "wavenet": {
                "hidden_dim": 512, "num_layers": 8, "kernel_size": 5,
                "dilation_rate": 1, "p_dropout": 0.2, "style_condition": True
            }
        },
        "gpt_checkpoint": gpt_checkpoint,
        "w2v_stat": w2v_stat,
        "s2mel_checkpoint": s2mel_checkpoint,
        "emo_matrix": emo_matrix,
        "spk_matrix": spk_matrix,
        "emo_num": [3, 17, 2, 8, 4, 5, 10, 24],
        "qwen_emo_path": qwen_emo_path,
        "vocoder": {
            "type": vocoder_type,
            "name": vocoder_name
        },
        "version": float(version)
    }

    try:
        # 1. Save config.yaml
        target_dir = core.configs_directory()
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, "config.yaml")

        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
        target_dir = core.models_directory()
        os.makedirs(target_dir, exist_ok=True)
           
        # 2. Create and save the COMPLETELY BLANK model
        blank_model_path = os.path.join(target_dir, "blank_model.pth")
        model = UnifiedVoice(**config_dict["gpt"])
        
        torch.save({
            "model": model.state_dict(),
            "step": 0,
            "epoch": 0
        }, blank_model_path)

        # 3. Load, resize, and save the OFFICIAL PRETRAINED model
        base_ckpt = os.path.join(core.path_base, "indextts", "checkpoints", "gpt.pth")
        official_model_path = os.path.join(target_dir, "official_model.pth")
        transfer_log = ""
        
        if os.path.exists(base_ckpt):
            try:
                ckpt = torch.load(base_ckpt, map_location="cpu")
                state_dict = ckpt.get("model", ckpt)
                new_state_dict = model.state_dict()
                filtered = {}
                
                target_vocab_size = config_dict["gpt"]["number_text_tokens"]
                if target_vocab_size > 1800 and target_vocab_size < 12001:
                    base_size = target_vocab_size - 1800
                    try:
                        sp = spm.SentencePieceProcessor()
                        official_bpe = os.path.join(core.path_base, "indextts", "checkpoints", "bpe.model")
                        sp.load(official_bpe)
                        
                        eng_vocab = []
                        
                        # The English block sits exactly from 10201 to 12000
                        
                        for i in range(10201, 12000):
                            eng_vocab.append({
                                "id": base_size + (i - 10200),
                                "piece": sp.id_to_piece(i),
                                "score": sp.get_score(i),
                                "is_control": sp.is_control(i),
                                "is_unknown": sp.is_unknown(i),
                                "is_unused": sp.is_unused(i),
                                "is_byte": sp.is_byte(i),
                            })
                            
                        eng_json_path = os.path.join(target_dir, "english.json")
                        with open(eng_json_path, "w", encoding="utf-8") as f:
                            json.dump(eng_vocab, f, ensure_ascii=False, indent=2)
                            
                        transfer_log += f"\n📦 English vocabulary block saved to: {eng_json_path}"
                    except Exception as e:
                        transfer_log += f"\n⚠️ Could not extract English vocab JSON: {e}"
                
                for k, v in state_dict.items():
                    new_k = k.replace(".base_layer.", ".")
                    if new_k not in new_state_dict: continue

                    target_shape = new_state_dict[new_k].shape
                    
                    if v.shape == target_shape:
                        filtered[new_k] = v
                        continue
                    
                    try:
                        if len(v.shape) == len(target_shape):
                            expanded_param = new_state_dict[new_k].clone()
                            
                            # --- SMART EMBEDDING SURGERY ---
                            # Target layers that map to the vocabulary size
                            is_vocab_layer = ("text_embedding.weight" in new_k) or ("text_head" in new_k)
                            
                            # Ensure we are shrinking, and have enough room for the 1800 English tokens
                            if is_vocab_layer and target_shape[0] < v.shape[0] and target_shape[0] > 1800:
                                eng_size = 1800
                                eng_start_old = 10201
                                base_size = target_shape[0] - eng_size
                                
                                if len(v.shape) == 1: 
                                    # 1D Tensor (e.g., text_head.bias)
                                    expanded_param[:base_size] = v[:base_size]
                                    expanded_param[base_size:] = v[eng_start_old : eng_start_old + eng_size]
                                else: 
                                    # 2D Tensor (e.g., text_embedding.weight)
                                    expanded_param[:base_size, :] = v[:base_size, :]
                                    expanded_param[base_size:, :] = v[eng_start_old : eng_start_old + eng_size, :]
                                    
                                transfer_log += f"\n✨ Surgery on {new_k}: {base_size} Base + {eng_size} English tokens appended (English block starts at ID {base_size})."
                            else:
                                # Standard naive slicing for non-vocab layers or incompatible dimensions
                                slices = tuple(slice(0, min(ds, ts)) for ds, ts in zip(v.shape, target_shape))
                                expanded_param[slices] = v[slices]
                                
                            filtered[new_k] = expanded_param
                    except Exception as e:
                        transfer_log += f"\nSkipped layer {new_k}: Dimension mismatch ({v.shape} vs {target_shape})."
                        continue # Safely skip incompatible dimensions
                        
                # Load the filtered/resized weights into the model graph
                model.load_state_dict(filtered, strict=False)
                
                # Save the successfully resized model
                torch.save({
                    "model": model.state_dict(),
                    "step": 0,
                    "epoch": 0
                }, official_model_path)
                
                transfer_log += f"\nSuccessfully generated resized base model: {official_model_path}"
            except Exception as e:
                transfer_log = f"\nWarning: Base checkpoint transfer failed: {str(e)}"
        else:
            transfer_log = "\nNotice: No base checkpoint found. Only untrained model was saved."

        return _("CONFIG_MSG_SUCCESS") + f"\n{out_path}\nSaved blank model: {blank_model_path}{transfer_log}"
        
    except Exception as e:
        return _("CONFIG_MSG_ERROR") + f" {str(e)}"
        
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    lang_options = core.language_list()
    
    with gr.Blocks() as demo:
        gr.Markdown(_("CONFIG_HEADER"))
        gr.Markdown(_("CONFIG_DESC"))

        # ==========================================
        # TOP SECTION: CORE SETTINGS (ALWAYS VISIBLE)
        # ==========================================
        with gr.Group():
            gr.Markdown(_("CONFIG_HEADER_CORE"))
            gr.Markdown(_("CONFIG_DESC_CORE"))
            
            with gr.Row():
                number_text_tokens = gr.Number(label=_("CONFIG_LABEL_VOCAB"), value=12000, precision=0, info=_("CONFIG_INFO_VOCAB"))
                sample_rate = gr.Number(label=_("CONFIG_LABEL_SR"), value=24000, precision=0)               
                
            with gr.Row():
                max_text_tokens = gr.Number(label=_("CONFIG_LABEL_MAX_TEXT"), value=600, precision=0)
                max_mel_tokens = gr.Number(label=_("CONFIG_LABEL_MAX_MEL"), value=1815, precision=0)
                
            with gr.Row():
                model_dim = gr.Number(label=_("CONFIG_LABEL_MOD_DIM"), value=1280, precision=0)
                layers = gr.Number(label=_("CONFIG_LABEL_LAYERS"), value=24, precision=0)
                heads = gr.Number(label=_("CONFIG_LABEL_HEADS"), value=20, precision=0)

        # ==========================================
        # ACCORDION: ADVANCED CONFIGURATIONS
        # ==========================================
        with gr.Accordion(_("CONFIG_ACC_ADVANCED"), open=False):
            
            with gr.Tab(_("CONFIG_TAB_TOKENIZER")):
                with gr.Row():
                    language = gr.Dropdown(label=_("COMMON_LABEL_LANG"), choices=lang_options, value="tr")
                    tokenizer_type = gr.Dropdown(label=_("CONFIG_LABEL_TOK_TYPE"), choices=["indextts", "itts-tr"], value="indextts")
                    vocab_type = gr.Dropdown(label=_("CONFIG_LABEL_VOCAB_TYPE"), choices=["trained", "merged"], value="trained")
                    case_format = gr.Dropdown(label=_("CONFIG_LABEL_CASE_FORMAT"), choices=["uppercase", "lowercase"], value="uppercase")
                with gr.Row():
                    wordify = gr.Checkbox(label=_("CONFIG_CHK_WORDIFY"), value=True)
                    abbreviations = gr.Checkbox(label=_("CONFIG_CHK_ABBREV"), value=False)
                    extract = gr.Checkbox(label=_("CONFIG_CHK_EXTRACT"), value=False)
            
            with gr.Tab(_("CONFIG_TAB_MEL")):
                with gr.Row():
                    bpe_model = gr.Textbox(label=_("CONFIG_LABEL_BPE"), value="bpe.model")
                    squeeze = gr.Checkbox(label=_("CONFIG_CHK_SQUEEZE"), value=False)
                    normalize_mel = gr.Checkbox(label=_("CONFIG_CHK_NORM_MEL"), value=False)
                with gr.Row():
                    n_fft = gr.Number(label="N FFT", value=1024, precision=0)
                    hop_length = gr.Number(label="Hop Length", value=256, precision=0)
                    win_length = gr.Number(label="Win Length", value=1024, precision=0)
                with gr.Row():
                    n_mels = gr.Number(label="N Mels", value=100, precision=0)
                    mel_fmin = gr.Number(label="Mel Fmin", value=0, precision=0)

            with gr.Tab(_("CONFIG_TAB_GPT")):
                with gr.Row():
                    use_mel_codes_as_input = gr.Checkbox(label=_("CONFIG_CHK_MEL_INPUT"), value=True)
                    train_solo_embeddings = gr.Checkbox(label=_("CONFIG_CHK_SOLO_EMB"), value=False)
                    condition_type = gr.Textbox(label=_("CONFIG_LABEL_COND_TYPE"), value="conformer_perceiver")
                with gr.Row():
                    mel_length_compression = gr.Number(label=_("CONFIG_LABEL_MEL_COMP"), value=1024, precision=0)
                    number_mel_codes = gr.Number(label=_("CONFIG_LABEL_NUM_MEL"), value=8194, precision=0)
                with gr.Row():
                    start_mel_token = gr.Number(label=_("CONFIG_LABEL_START_MEL"), value=8192, precision=0)
                    stop_mel_token = gr.Number(label=_("CONFIG_LABEL_STOP_MEL"), value=8193, precision=0)
                with gr.Row():
                    start_text_token = gr.Number(label=_("CONFIG_LABEL_START_TEXT"), value=0, precision=0)
                    stop_text_token = gr.Number(label=_("CONFIG_LABEL_STOP_TEXT"), value=1, precision=0)

            with gr.Tab(_("CONFIG_TAB_CKPT")):
                with gr.Row():
                    gpt_checkpoint = gr.Textbox(label=_("CONFIG_LABEL_GPT_CKPT"), value="gpt.pth")
                    s2mel_checkpoint = gr.Textbox(label=_("CONFIG_LABEL_S2MEL_CKPT"), value="s2mel.pth")
                    w2v_stat = gr.Textbox(label=_("CONFIG_LABEL_W2V_STAT"), value="wav2vec2bert_stats.pt")
                with gr.Row():
                    emo_matrix = gr.Textbox(label=_("CONFIG_LABEL_EMO_MAT"), value="feat2.pt")
                    spk_matrix = gr.Textbox(label=_("CONFIG_LABEL_SPK_MAT"), value="feat1.pt")
                    qwen_emo_path = gr.Textbox(label=_("CONFIG_LABEL_QWEN"), value="qwen0.6bemo4-merge/")
                with gr.Row():
                    vocoder_type = gr.Textbox(label=_("CONFIG_LABEL_VOC_TYPE"), value="bigvgan")
                    vocoder_name = gr.Textbox(label=_("CONFIG_LABEL_VOC_NAME"), value="nvidia/bigvgan_v2_22khz_80band_256x")
                    version = gr.Number(label=_("CONFIG_LABEL_VER"), value=2.0)

        # ==========================================
        # ACTION BUTTON & LOGS
        # ==========================================
        with gr.Row():
            save_btn = gr.Button(_("CONFIG_BTN_SAVE"), variant="primary")
            
        output_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=3, interactive=False)

        inputs_list = [
            number_text_tokens, max_text_tokens, max_mel_tokens, model_dim, layers, heads, sample_rate,
            language, tokenizer_type, vocab_type, case_format, wordify, abbreviations, extract,
            bpe_model, squeeze, n_fft, hop_length, win_length, n_mels, mel_fmin, normalize_mel,
            use_mel_codes_as_input, mel_length_compression, number_mel_codes, start_mel_token, stop_mel_token, start_text_token, stop_text_token, train_solo_embeddings, condition_type,
            gpt_checkpoint, w2v_stat, s2mel_checkpoint, emo_matrix, spk_matrix, qwen_emo_path, vocoder_type, vocoder_name, version
        ]

        save_btn.click(
            fn=save_config_ui,
            inputs=inputs_list,
            outputs=output_log
        )
        
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("config"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")

    return demo