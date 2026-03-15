import gradio as gr
import os
import time
import pandas as pd
import json
import re
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import warnings
from subprocess import CalledProcessError
import random
import torch.nn.functional as F

# Configuration imports
from omegaconf import OmegaConf

# ITTS Core & Backend Engine
from core import core
from core.core import _
from core.itts import IndexTTS2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION ---
MODEL_DIR = "./indextts/checkpoints"
tts = None

# ==========================================
# Helpers
# ==========================================
    
def get_train_folders():
    """Lists subdirectories in the 'trains' directory."""
    try:
        trains_path = os.path.join(core.path_base, "trains")
        if not os.path.exists(trains_path):
            return []
        folders = [
            d for d in os.listdir(trains_path) 
            if os.path.isdir(os.path.join(trains_path, d))
        ]
        return sorted(folders)
    except Exception as e:
        print(f"Error listing train folders: {e}")
        return []
    
def unwrap_checkpoint(ckpt_path, run_dir):
    clean_path = os.path.join(run_dir, "inference_weights.pth")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        torch.save(new_state, clean_path)
        return clean_path
    except: return ckpt_path

def sync_config_to_checkpoint(cfg, ckpt_path):
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if "model" in sd: sd = sd["model"]
        text_key = "text_embedding.weight"
        if text_key not in sd and "module."+text_key in sd: text_key = "module."+text_key
        if text_key in sd:
            actual = sd[text_key].shape[0] - 1
            if cfg.gpt.number_text_tokens != actual:
                cfg.gpt.number_text_tokens = actual
        mel_key = "mel_pos_embedding.emb.weight"
        if mel_key not in sd and "module."+mel_key in sd: mel_key = "module."+mel_key
        if mel_key in sd:
            actual = sd[mel_key].shape[0] - 3
            if cfg.gpt.max_mel_tokens != actual:
                cfg.gpt.max_mel_tokens = actual
    except: pass
    return cfg
    
def auto_discover_inference_settings(folder_name, is_original):
    """Automatically reads case_format and tok_type from the project's config.yaml"""
    if is_original or not folder_name:
        return gr.update(value="lowercase"), gr.update(value="itts-tr")
        
    # Check both potential config locations
    cfg_path_train = os.path.join(core.path_base, "trains", folder_name, "config.yaml")
    cfg_path_proj = os.path.join(core.path_base, "projects", folder_name, "configs", "config.yaml")
    
    cfg_path = cfg_path_train if os.path.exists(cfg_path_train) else cfg_path_proj
    
    if os.path.exists(cfg_path):
        try:
            conf = OmegaConf.load(cfg_path)
            if "tokenizer" in conf:
                tok_cfg = conf.tokenizer
                # Fallback to defaults if the keys are somehow missing
                c_format = getattr(tok_cfg, "case_format", "lowercase")
                t_type = getattr(tok_cfg, "tokenizer_type", "itts-tr")
                return gr.update(value=c_format), gr.update(value=t_type)
        except Exception as e:
            print(f"Error reading config for UI update: {e}")
            
    return gr.update(value="lowercase"), gr.update(value="itts-tr")
    
# ==========================================
# Application Logic
# ==========================================
def load_custom_model_logic(folder_name, use_cuda_ui, use_compile_ui, use_original_model, case_format, tok_type):
    global tts
    log_history = ""
    
    # -----------------------------------------------------
    # UPDATED LOGIC: BYPASS FOLDER IF "USE ORIGINAL" IS CHECKED
    # -----------------------------------------------------
    if use_original_model:
        log_history += f"[{time.strftime('%H:%M:%S')}] 🟡 Mode: Base Model (Bypassing custom folder)...\n"
        yield log_history
        
        # Force the project name to something distinct so we know we are in "Original" mode
        folder_name = "itts_base_model" 
        core.project_name = folder_name
        
        # Force paths to the base checkpoints directory
        target_train_dir = MODEL_DIR  # ./checkpoints
        target_project_dir = None
        
        log_history += f"📂 Loading directly from: {target_train_dir}\n"
        yield log_history
        
    else:
        # Standard Logic
        if not folder_name:
            yield ">> Error: No folder selected."
            return

        log_history += f"[{time.strftime('%H:%M:%S')}] Selecting project: {folder_name}...\n"
        yield log_history
        core.project_name = folder_name
        
        target_train_dir = os.path.join(core.path_base, "trains", folder_name)
        target_project_dir = os.path.join(core.path_base, "projects", folder_name)
        if not os.path.exists(target_project_dir):
            target_project_dir = None
            
        if not os.path.exists(target_train_dir):
                log_history += f"❌ Error: Directory not found: {target_train_dir}\n"
                yield log_history
                return

        log_history += f"📂 Train Dir: {target_train_dir}\n"
        yield log_history
    # -----------------------------------------------------
    
    try:
        if tts is not None:
            del tts
            torch.cuda.empty_cache()
            log_history += "🗑️ Old model unloaded. Memory cleared.\n"
            yield log_history

        # Attempt to load
        try:
            # 1. Initialize Wrapper (Fast)
            log_history += f"[{time.strftime('%H:%M:%S')}] Initializing wrapper...\n"
            yield log_history
            
            tts = IndexTTS2(
                model_dir=MODEL_DIR,
                project_dir=target_project_dir,
                train_dir=target_train_dir,
                loaded_project_name=folder_name,
                use_cuda_kernel=use_cuda_ui,
                use_torch_compile=use_compile_ui,
                do_load=False,
                case_format=case_format, 
                tok_type=tok_type                
            )
            
            # 2. Iterate Loading Steps (Streaming Logs)
            for msg in tts.load_resources():
                log_history += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
                yield log_history

            mode_str = "ON" if use_cuda_ui else "OFF"
            log_history += f"[{time.strftime('%H:%M:%S')}] 📛 BigVGAN Kernel: {mode_str}\n"
            log_history += f"[{time.strftime('%H:%M:%S')}] ✅ Success!\n"
            yield log_history
            
        except RuntimeError as re:
            if "Ninja is required" in str(re) or "ninja" in str(re).lower():
                log_history += "\n⚠️ WARNING: 'Ninja' missing. Falling back to standard mode.\n"
                yield log_history
                
                tts = IndexTTS2(
                    model_dir=MODEL_DIR,
                    project_dir=target_project_dir,
                    train_dir=target_train_dir,
                    loaded_project_name=folder_name,
                    use_cuda_kernel=False,
                    do_load=False,
                    case_format=case_format,
                    tok_type=tok_type
                )
                
                # Retry loading (Streaming Logs)
                for msg in tts.load_resources():
                    log_history += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
                    yield log_history

                log_history += f"[{time.strftime('%H:%M:%S')}] ✅ Success! (Fallback Mode)\n"
                yield log_history
            else:
                raise re

    except Exception as e:
        log_history += f"[{time.strftime('%H:%M:%S')}] ❌ Critical Error: {str(e)}\n"
        yield log_history
        import traceback
        traceback.print_exc()
# ----------------------------------

def generate_speech_logic(
        selected_folder, text, seed_val, prompt_audio, emo_method_idx, emo_upload, emo_weight,
        v1, v2, v3, v4, v5, v6, v7, v8, emo_text,
        do_sample, temp, top_p, max_mel, max_text_seg,
        use_cuda_ui, use_compile_ui, use_original_model,language_choice, case_format, tok_type
):
    global tts
    
    # 1. Initialize Log
    log_history = f"[{time.strftime('%H:%M:%S')}] Request Received.\n"
    yield None, log_history, ""

    # -----------------------------------------------------
    # DETERMINE TARGET CONFIG BASED ON CHECKBOX
    # -----------------------------------------------------
    if use_original_model:
        current_project = "itts_base_model"
        target_train_dir = MODEL_DIR
        target_project_dir = None
    else:
        # Use the choice from folder_dropdown directly
        current_project = selected_folder
        target_train_dir = os.path.join(core.path_base, "trains", current_project)
        target_project_dir = os.path.join(core.path_base, "projects", current_project)
        if not os.path.exists(target_project_dir):
            target_project_dir = None
    # -----------------------------------------------------
    
    current_kernel_setting = getattr(tts, 'use_cuda_kernel', False) if tts else False
    current_compile_setting = getattr(tts, 'use_torch_compile', False) if tts else False
    
    need_reload = (tts is None) or \
                  (tts.loaded_project != current_project) or \
                  (current_kernel_setting != use_cuda_ui) or \
                  (current_compile_setting != use_compile_ui) or \
                  (getattr(tts, 'case_format', 'lowercase') != case_format) or \
                  (getattr(tts, 'tok_type', 'itts-tr') != tok_type)

    if need_reload:
        log_history += f"[{time.strftime('%H:%M:%S')}] Auto-Initializing Model (Target: {current_project})...\n"
        yield None, log_history, ""
        print(f">> Initializing IndexTTS2 Model for project: {current_project}...")
        
        if tts is not None:
            del tts
            torch.cuda.empty_cache()
            
        try:
            try:
                # Use do_load=True (default) for blocking auto-load during generation
                tts = IndexTTS2(
                    model_dir=MODEL_DIR,
                    project_dir=target_project_dir,
                    train_dir=target_train_dir,
                    loaded_project_name=current_project,
                    use_cuda_kernel=use_cuda_ui,
                    use_torch_compile=use_compile_ui,
                    do_load=True,
                    case_format=case_format,
                    tok_type=tok_type
                )
            except RuntimeError as re:
                if "Ninja" in str(re) or "ninja" in str(re):
                    msg = "⚠️ Ninja missing. Falling back to standard mode."
                    print(f">> {msg}")
                    log_history += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
                    yield None, log_history, ""
                    
                    tts = IndexTTS2(
                        model_dir=MODEL_DIR,
                        project_dir=target_project_dir,
                        train_dir=target_train_dir,
                        loaded_project_name=current_project,
                        use_cuda_kernel=False,
                        do_load=True,
                        case_format=case_format,
                        tok_type=tok_type
                    )
                else:
                    raise re

            log_history += f"[{time.strftime('%H:%M:%S')}] Model Ready.\n"
            yield None, log_history, ""
        except Exception as e:
            yield None, log_history + f"Error initializing model: {str(e)}", ""
            return

    tokenizer_output_str = ""
    if tts is not None and getattr(tts, 'tokenizer', None) is not None:
        try:
            tokens = tts.tokenizer.tokenize(text)
            ids = tts.tokenizer.convert_tokens_to_ids(tokens)
            tokenizer_output_str = f"Tokens:\n{tokens}\n\nIDs:\n{ids}"
        except Exception as e:
            tokenizer_output_str = f"Tokenization error: {str(e)}"
   
    output_dir = os.path.join("outputs", "inference")
    output_path = os.path.join(output_dir, f"gen_{int(time.time())}.wav")
    os.makedirs(output_dir, exist_ok=True)

    vec = None
    if emo_method_idx == 2:
        raw_vec = [v1, v2, v3, v4, v5, v6, v7, v8]
        vec = tts.normalize_emo_vec(raw_vec, apply_bias=True)
        
    if seed_val != -1:
        torch.manual_seed(int(seed_val))
        random.seed(int(seed_val))
        log_history += f"[{time.strftime('%H:%M:%S')}] 🌱 Random Seed set to: {int(seed_val)}\n"
        yield None, log_history, ""

    try:
        gen = tts.infer_generator(
            spk_audio_prompt=prompt_audio,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_upload if emo_method_idx == 1 else None,
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_method_idx == 3),
            emo_text=emo_text if emo_text != "" else None,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
            max_mel_tokens=int(max_mel),
            max_text_tokens_per_segment=int(max_text_seg),
            stream_return=False,
            language=language_choice            
        )

        final_result = None
        for item in gen:
            if isinstance(item, str) and item.startswith("LOG:"):
                msg = item.replace("LOG: ", "")
                log_history += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
                yield None, log_history, tokenizer_output_str
            else:
                final_result = item
        
        log_history += f"[{time.strftime('%H:%M:%S')}] Success! Saved to {final_result}\n"
        yield final_result, log_history, tokenizer_output_str

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, log_history + f"\nError during generation: {str(e)}", tokenizer_output_str
        
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    EMO_CHOICES = [
        _("INFERENCE_CHOICE_SAME"), 
        _("INFERENCE_CHOICE_REF"), 
        _("INFERENCE_CHOICE_VEC"), 
        _("INFERENCE_CHOICE_DESC")
    ]

    with gr.Blocks() as demo:
        gr.Markdown(_("INFERENCE_HEADER"))
        gr.Markdown(_("INFERENCE_DESC"))

        with gr.Row():
            # ==========================================
            # LEFT COLUMN: Controls & Input
            # ==========================================
            with gr.Column(scale=1):
                current_folders = get_train_folders()                
                with gr.Row():                  
                    with gr.Column(scale=1):                       
                        with gr.Row():
                            folder_dropdown = gr.Dropdown(
                                label=_("INFERENCE_LABEL_FOLDER"), 
                                choices=current_folders, 
                                value=current_folders[0] if current_folders else None,
                                interactive=True,
                                scale=3
                            )                            
                        with gr.Row():
                            refresh_folders_btn = gr.Button(_("COMMON_BTN_REFRESH"), size="sm")                      
                    with gr.Column(scale=1):                        
                        with gr.Row():                        
                            use_original_model_cb = gr.Checkbox(
                                label=_("INFERENCE_CHK_ORIGINAL_MODEL"), 
                                value=False,
                                scale=1
                            )                   
                        with gr.Row():                   
                            language_dropdown = gr.Dropdown(
                            choices=["Auto", "TR", "EN"],
                            value="Auto",
                            show_label=False,
                            interactive=True
                            )
                            
                with gr.Row():
                    seed_input = gr.Number(
                    label=_("INFERENCE_LABEL_SEED"), 
                    value=-1, 
                    precision=0
                    )

                    
                refresh_folders_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_train_folders()),
                    outputs=[folder_dropdown]
                )
                
                load_btn = gr.Button(_("INFERENCE_BTN_LOAD_MODEL"), variant="secondary")
                
                # 2. Input Text (Varsayılan olarak Türkçe)
                text_input = gr.TextArea(
                    label=_("INFERENCE_LABEL_INPUT_TEXT"), 
                    value="Mum ışığının aydınlattığı köhne kulübede, yorgun yolcu yılmış bakışlarını yavaşça ocağın üzerindeki gümüş çaydanlığa kaydırdı.", 
                    lines=9
                )
                
                # 3. Generate Button
                gen_btn = gr.Button(_("INFERENCE_BTN_GENERATE"), variant="primary")

            # ==========================================
            # RIGHT COLUMN: Audio I/O
            # ==========================================
            with gr.Column(scale=1):
                gr.Markdown(_("INFERENCE_MARKDOWN_TTYPE"))
                with gr.Row():
                    tok_type_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_TTYPE"),
                        choices=["itts-tr", "indextts"],
                        value="itts-tr",
                        interactive=True
                    )
                with gr.Row(visible=True) as case_row:
                    case_format = gr.Dropdown(
                        label=_("INFERENCE_LABEL_CASE_FORMAT"),
                        choices=["lowercase", "uppercase"],
                        value="lowercase",
                        interactive=True
                    )
                with gr.Row():
                    ref_audio = gr.Audio(
                        label=_("INFERENCE_LABEL_REF_AUDIO"), 
                        type="filepath", 
                        value="examples/tr_female.wav"
                    )                     
                with gr.Row():
                    audio_out = gr.Audio(label=_("INFERENCE_LABEL_GEN_RESULT"))
                
            tok_type_dd.change(
                fn=lambda t: gr.update(visible=(t == "itts-tr")),
                inputs=[tok_type_dd],
                outputs=[case_row]
            )

        # ==========================================
        # SETTINGS AREAS
        # ==========================================
        with gr.Accordion("Tokenizer Output", open=False):
            tokenizer_out_tb = gr.Textbox(show_label=False, interactive=False, lines=4)
        with gr.Accordion(_("INFERENCE_ACC_EMOTION"), open=True):
            emo_method = gr.Radio(choices=EMO_CHOICES, type="index", value=EMO_CHOICES[0], label=_("INFERENCE_LABEL_CONTROL_MODE"))
            with gr.Row(visible=False) as weight_row:
                emo_weight = gr.Slider(label=_("INFERENCE_LABEL_EMO_INTENSITY"), minimum=0.0, maximum=1.0, value=1.0)
            with gr.Group(visible=False) as ref_group:
                emo_upload = gr.Audio(label=_("INFERENCE_LABEL_EMO_REF_AUDIO"), type="filepath")
            with gr.Group(visible=False) as vec_group:
                with gr.Row():
                    v1 = gr.Slider(label=_("INFERENCE_LABEL_EMO_HAPPY"), minimum=0, maximum=1, value=0, step=0.05)
                    v2 = gr.Slider(label=_("INFERENCE_LABEL_EMO_ANGRY"), minimum=0, maximum=1, value=0, step=0.05)
                    v3 = gr.Slider(label=_("INFERENCE_LABEL_EMO_SAD"), minimum=0, maximum=1, value=0, step=0.05)
                    v4 = gr.Slider(label=_("INFERENCE_LABEL_EMO_AFRAID"), minimum=0, maximum=1, value=0, step=0.05)
                with gr.Row():
                    v5 = gr.Slider(label=_("INFERENCE_LABEL_EMO_DISGUSTED"), minimum=0, maximum=1, value=0, step=0.05)
                    v6 = gr.Slider(label=_("INFERENCE_LABEL_EMO_MELANCHOLY"), minimum=0, maximum=1, value=0, step=0.05)
                    v7 = gr.Slider(label=_("INFERENCE_LABEL_EMO_SURPRISED"), minimum=0, maximum=1, value=0, step=0.05)
                    v8 = gr.Slider(label=_("INFERENCE_LABEL_EMO_CALM"), minimum=0, maximum=1, value=0, step=0.05)
            with gr.Group(visible=False) as text_group:
                emo_text = gr.Textbox(label=_("INFERENCE_LABEL_EMO_DESC"), placeholder=_("INFERENCE_PLACEHOLDER_EMO_DESC"))

        with gr.Accordion(_("INFERENCE_ACC_ADVANCED"), open=False):
            use_cuda_cb = gr.Checkbox(label=_("INFERENCE_CHK_CUDA"), value=False)
            use_compile_cb = gr.Checkbox(label=_("INFERENCE_CHK_USE_COMPILE"), value=False)
            with gr.Row():
                do_sample = gr.Checkbox(label=_("INFERENCE_CHK_SAMPLE"), value=True)
                temp = gr.Slider(label=_("INFERENCE_SLIDER_TEMP"), minimum=0.1, maximum=1.5, value=0.8)
                top_p = gr.Slider(label=_("INFERENCE_SLIDER_TOP_P"), minimum=0.0, maximum=1.0, value=0.8)
            with gr.Row():
                max_mel = gr.Slider(label=_("INFERENCE_SLIDER_MAX_MEL"), value=1500, minimum=50, maximum=2000)
                max_text_seg = gr.Slider(label=_("INFERENCE_SLIDER_MAX_TEXT"), value=120, minimum=20, maximum=200)

        # ==========================================
        # LOGS (Very Bottom)
        # ==========================================
        system_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=10, interactive=False)

        # ==========================================
        # EVENT WIRING
        # ==========================================
        
        def change_default_text(is_original):
            if is_original:
                return "After a long and quiet night, the curious traveler slowly opened the old wooden door and stepped into the warm morning light."
            else:
                return "Mum ışığının aydınlattığı köhne kulübede, yorgun yolcu yılmış bakışlarını yavaşça ocağın üzerindeki gümüş çaydanlığa kaydırdı."

        use_original_model_cb.change(
            fn=change_default_text,
            inputs=[use_original_model_cb],
            outputs=[text_input]
        )
        folder_dropdown.change(
            fn=auto_discover_inference_settings,
            inputs=[folder_dropdown, use_original_model_cb],
            outputs=[case_format, tok_type_dd]
        )        
        use_original_model_cb.change(
            fn=auto_discover_inference_settings,
            inputs=[folder_dropdown, use_original_model_cb],
            outputs=[case_format, tok_type_dd]
        )
        demo.load(
            fn=auto_discover_inference_settings,
            inputs=[folder_dropdown, use_original_model_cb],
            outputs=[case_format, tok_type_dd]
        )

        # UI Visibility
        def update_ui(method_idx):
            return {
                ref_group: gr.update(visible=(method_idx == 1)),
                vec_group: gr.update(visible=(method_idx == 2)),
                text_group: gr.update(visible=(method_idx == 3)),
                weight_row: gr.update(visible=(method_idx > 0))
            }
        emo_method.change(update_ui, inputs=[emo_method], outputs=[ref_group, vec_group, text_group, weight_row])

        # Load Logic -> Unified Log
        load_btn.click(
            load_custom_model_logic, 
            inputs=[folder_dropdown, use_cuda_cb, use_compile_cb, use_original_model_cb, case_format, tok_type_dd], 
            outputs=[system_log]
        )

        # Gen Logic -> Unified Log
        gen_btn.click(
            fn=generate_speech_logic,
            inputs=[
                folder_dropdown, text_input, seed_input, ref_audio, emo_method, emo_upload, emo_weight,
                v1, v2, v3, v4, v5, v6, v7, v8, emo_text,
                do_sample, temp, top_p, max_mel, max_text_seg,
                use_cuda_cb, use_compile_cb, use_original_model_cb, language_dropdown, case_format, tok_type_dd
            ],
            outputs=[audio_out, system_log, tokenizer_out_tb]
        )
        
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("inference"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")

    return demo