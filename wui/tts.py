import gradio as gr
import os
import time
import torch
import torchaudio
import safetensors
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextTokenizer, TextNormalizer
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from transformers import SeamlessM4TFeatureExtractor

from core import core
from core.core import _
from core.itts import IndexTTS2, QwenEmotion
from core.spice import GenericSpiceTokenizer
from core.normalizer import MultilingualNormalizer
import random

# ==========================================
# Standalone Backend Class
# ==========================================

class StandaloneTTS(IndexTTS2):
    """
    Standalone inference module for ITTS. 
    Bypasses the UI pipeline, targeting core.wui_ckpt directly with prefix support.
    """
    def __init__(self, gpt_file_name="gpt.pth", device=None, use_fp16=False, use_cuda_kernel=False, use_torch_compile=False):
        
        self.ckpt_dir = core.wui_ckpt
        # Default base model path for dependencies like bigvgan, s2mel, etc.
        self.model_dir = os.path.join(core.path_base, "indextts", "checkpoints") 
        
        # 1. Parse prefix and tokenizer type based on gpt_file_name
        self.prefix = gpt_file_name.replace("gpt.pth", "")
        
        if gpt_file_name == "gpt.pth":
            # Native Base Model (Loads from core checkpoints)
            self.tok_type = "indextts"
            self.cfg_path = os.path.join(self.model_dir, "config.yaml")
            self.gpt_path = os.path.join(self.model_dir, gpt_file_name) 
            self.bpe_path = os.path.join(self.model_dir, "bpe.model")
        elif gpt_file_name.startswith("en_"):
            # Downloaded English Models (Loads from wui_ckpt with official tokenizer)
            self.tok_type = "indextts"
            self.cfg_path = os.path.join(self.ckpt_dir, f"{self.prefix}config.yaml")
            self.gpt_path = os.path.join(self.ckpt_dir, gpt_file_name)
            self.bpe_path = os.path.join(self.ckpt_dir, f"{self.prefix}bpe.model")
        else:
            # Custom Turkish Models (Loads from wui_ckpt with ITTS-TR tokenizer)
            self.tok_type = "itts-tr"
            self.cfg_path = os.path.join(self.ckpt_dir, f"{self.prefix}config.yaml")
            self.gpt_path = os.path.join(self.ckpt_dir, gpt_file_name)
            self.bpe_path = os.path.join(self.ckpt_dir, f"{self.prefix}bpe.model")
            
        # 2. Map target files in wui_ckpt
        correct_cfg_path = os.path.join(self.ckpt_dir, f"{self.prefix}config.yaml")
        self.cfg_path = correct_cfg_path
        self.gpt_path = os.path.join(self.ckpt_dir, gpt_file_name)
        self.bpe_path = os.path.join(self.ckpt_dir, f"{self.prefix}bpe.model")
        
        # 3. Validate file existence before loading weights
        for path in [self.cfg_path, self.gpt_path, self.bpe_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required architecture file: {os.path.basename(path)}")
                
        # 4. Initialize base parameters (do_load=False prevents parent's default loading)
        super().__init__(model_dir=self.model_dir, do_load=False, device=device, use_fp16=use_fp16, 
                         use_cuda_kernel=use_cuda_kernel, use_torch_compile=use_torch_compile,
                         tok_type=self.tok_type, loaded_project_name="itts_base_model" if self.tok_type == "indextts" else None)
        
        # 5. FORCE RESTORE the correct custom config path! (Overriding the parent's fallback)
        self.cfg_path = correct_cfg_path
        self.cfg = OmegaConf.load(self.cfg_path)
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

    def load_resources(self):
        """Generator that loads resources and yields progress strings for the UI."""
        yield f"⚙️ Config Loaded: {os.path.basename(self.cfg_path)}"

        # 1. Qwen Emotion
        yield "🧠 Loading Emotion Model (Qwen)..."
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))
        yield "   ✅ Emotion Model Ready"

        # 2. GPT
        yield "🧠 Loading Semantic Model (GPT)..."
        
        try:
            sd = torch.load(self.gpt_path, map_location="cpu", weights_only=False)
            if "model" in sd: sd = sd["model"]
            
            t_key = "text_embedding.weight"
            if t_key not in sd and "module." + t_key in sd: t_key = "module." + t_key
            if t_key in sd: self.cfg.gpt.number_text_tokens = sd[t_key].shape[0] - 1
            
            m_key = "mel_pos_embedding.emb.weight"
            if m_key not in sd and "module." + m_key in sd: m_key = "module." + m_key
            if m_key in sd: self.cfg.gpt.max_mel_tokens = sd[m_key].shape[0] - 3
            del sd
        except Exception as e:
            yield f"   ⚠️ Warning: Could not auto-sync tensor shapes: {str(e)}"
        
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=getattr(self, 'use_accel', False))
        yield f"   📂 GPT Path: {self.gpt_path}"
        
        load_checkpoint(self.gpt, self.gpt_path)
        
        # --- RESTORED DEVICE MAPPING ---
        self.gpt = self.gpt.to(self.device)
        
        if getattr(self, 'use_fp16', False): 
            self.gpt.eval().half()
        else: 
            self.gpt.eval()
            
        use_ds = getattr(self, 'use_deepspeed', False)
        try: 
            import deepspeed
        except: 
            use_ds = False
            
        self.gpt.post_init_gpt2_config(use_deepspeed=use_ds, kv_cache=True, half=getattr(self, 'use_fp16', False))
        
        yield "   ✅ GPT Ready"

        # 3. Features
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device).eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device).eval()

        # 4. Acoustic Subsystem (S2Mel)
        yield "🌊 Loading Acoustic Model (S2Mel)..."
        s2mel_path = self._resolve_path(self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        self.s2mel = s2mel.to(self.device).eval()
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        if self.use_torch_compile: self.s2mel.enable_torch_compile()

        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device).eval()
        yield "   ✅ S2Mel Ready"

        # 5. Vocoder
        yield "🔊 Loading Vocoder (BigVGAN)..."
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(self.cfg.vocoder.name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device).eval()
        self.bigvgan.remove_weight_norm()
        yield "   ✅ Vocoder Ready"

        # 6. Tokenizer
        yield "📝 Loading Tokenizer..."
        yield f"   📂 BPE Path: {self.bpe_path}"
        if self.tok_type == "indextts":
            yield "   ℹ️ Auto-Selected: Standard Normalizer (Official)"
            self.normalizer = TextNormalizer()
            self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)  # Aligned with inference.py (Positional)
        else:
            case_fmt = "lowercase"
            if "tokenizer" in self.cfg:
                case_fmt = getattr(self.cfg.tokenizer, "case_format", "lowercase")               
            
            # --- DEBUG PRINTS ---
            print(f"\n>>> DEBUG: Raw case_format read from config is: '{case_fmt}'\n")
            yield f"   🐞 DEBUG: Raw case_format from config: '{case_fmt}'"
            # --------------------
            
            is_upper = (case_fmt == "uppercase")            
            yield f"   ℹ️ Auto-Selected: ITTS-TR BloomTokenizer (MultilingualNormalizer | wordify=True | upper={is_upper})"            
            
            self.normalizer = MultilingualNormalizer(lang="tr", wordify=True, upper=is_upper)
            self.tokenizer = GenericSpiceTokenizer(vocab_file=self.bpe_path, normalizer=self.normalizer)
            
        yield "   ✅ Tokenizer Ready"

        # 7. Matrices
        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)
        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)
        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)
        yield "✨ All Resources Loaded Successfully!"


# ==========================================
# Application UI & Logic
# ==========================================

tts = None

def get_wui_ckpt_models():
    """Scans core.wui_ckpt for all available gpt.pth files."""
    if not os.path.exists(core.wui_ckpt):
        return []
    files = os.listdir(core.wui_ckpt)
    models = [f for f in files if f.endswith("gpt.pth")]
    return sorted(models)

def load_standalone_model(gpt_file_name, use_cuda_ui, use_compile_ui):
    global tts
    log_history = f"[{time.strftime('%H:%M:%S')}] Attempting to load: {gpt_file_name}...\n"
    yield log_history
    
    if not gpt_file_name:
        yield log_history + "❌ Error: No model selected."
        return

    try:
        if tts is not None:
            del tts
            torch.cuda.empty_cache()
            log_history += "🗑️ Old model unloaded. Memory cleared.\n"
            yield log_history

        log_history += f"[{time.strftime('%H:%M:%S')}] Initializing Standalone Wrapper...\n"
        yield log_history
        
        tts = StandaloneTTS(
            gpt_file_name=gpt_file_name,
            use_cuda_kernel=use_cuda_ui,
            use_torch_compile=use_compile_ui
        )
        
        for msg in tts.load_resources():
            log_history += f"[{time.strftime('%H:%M:%S')}] {msg}\n"
            yield log_history

        log_history += f"[{time.strftime('%H:%M:%S')}] ✅ Success! Model ready for inference.\n"
        yield log_history
        
    except Exception as e:
        log_history += f"[{time.strftime('%H:%M:%S')}] ❌ Critical Error: {str(e)}\n"
        yield log_history
        import traceback
        traceback.print_exc()

def generate_speech_standalone(
        selected_model, text, seed_val, prompt_audio, emo_method_idx, emo_upload, emo_weight,
        v1, v2, v3, v4, v5, v6, v7, v8, emo_text,
        do_sample, temp, top_p, max_mel, max_text_seg,
        use_cuda_ui, use_compile_ui, language_choice
):
    global tts
    
    log_history = f"[{time.strftime('%H:%M:%S')}] Request Received.\n"
    yield None, log_history

    # Check if model needs reloading
    current_model_loaded = getattr(tts, 'gpt_path', "") if tts else ""
    target_gpt_path = os.path.join(core.wui_ckpt, selected_model) if selected_model else ""
    
    current_kernel_setting = getattr(tts, 'use_cuda_kernel', False) if tts else False
    current_compile_setting = getattr(tts, 'use_torch_compile', False) if tts else False
    
    need_reload = (tts is None) or \
                  (current_model_loaded != target_gpt_path) or \
                  (current_kernel_setting != use_cuda_ui) or \
                  (current_compile_setting != use_compile_ui)

    if need_reload:
        log_history += f"[{time.strftime('%H:%M:%S')}] Auto-Initializing Model...\n"
        yield None, log_history
        
        try:
            if tts is not None:
                del tts
                torch.cuda.empty_cache()
            
            tts = StandaloneTTS(
                gpt_file_name=selected_model,
                use_cuda_kernel=use_cuda_ui,
                use_torch_compile=use_compile_ui
            )
            # Run blocking load
            for msg in tts.load_resources(): pass 
            
            log_history += f"[{time.strftime('%H:%M:%S')}] Model Ready.\n"
            yield None, log_history
        except Exception as e:
            yield None, log_history + f"Error initializing model: {str(e)}"
            return

    output_dir = os.path.join("outputs", "standalone")
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
        yield None, log_history

    # --- TOKENIZER DIAGNOSTIC HOOK ---
    try:
        tokens = tts.tokenizer.tokenize(text)
        ids = tts.tokenizer.convert_tokens_to_ids(tokens)
        log_history += f"[{time.strftime('%H:%M:%S')}] Tokens: {tokens}\n"
        log_history += f"[{time.strftime('%H:%M:%S')}] IDs: {ids}\n"
        yield None, log_history
    except Exception as e:
        log_history += f"[{time.strftime('%H:%M:%S')}] ERROR - Tokenization failed: {str(e)}\n"
        yield None, log_history
    # ---------------------------------

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
                yield None, log_history
            else:
                final_result = item
        
        log_history += f"[{time.strftime('%H:%M:%S')}] Success! Saved to {final_result}\n"
        yield final_result, log_history

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, log_history + f"\nError during generation: {str(e)}"

# ==========================================
# Gradio UI Construction
# ==========================================

def create_demo():
    EMO_CHOICES = [
        _("INFERENCE_CHOICE_SAME"), 
        _("INFERENCE_CHOICE_REF"), 
        _("INFERENCE_CHOICE_VEC"), 
        _("INFERENCE_CHOICE_DESC")
    ]

    with gr.Blocks() as demo:
        gr.Markdown(_("TTS_HEADER"))
        gr.Markdown(_("TTS_DESC"))

        with gr.Row():
            # --- LEFT COLUMN ---
            with gr.Column(scale=1):
                models_list = get_wui_ckpt_models()                
                with gr.Row():                  
                    with gr.Column(scale=1):                       
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label=_("TTS_LABEL_SELECT_MODEL"), 
                                choices=models_list, 
                                value=models_list[0] if models_list else None,
                                interactive=True,
                                scale=3
                            )                            
                        with gr.Row():
                            refresh_btn = gr.Button(_("COMMON_BTN_REFRESH"), size="sm")                      
                    with gr.Column(scale=1):                        
                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                label=_("TTS_LABEL_LANG_INJECT"),
                                choices=["Auto", "TR", "EN"],
                                value="Auto",
                                interactive=True
                            )
                with gr.Row():
                    seed_input = gr.Number(
                        label=_("INFERENCE_LABEL_SEED"), 
                        value=-1, 
                        precision=0
                    )
                    
                refresh_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_wui_ckpt_models()),
                    outputs=[model_dropdown]
                )
                
                load_btn = gr.Button(_("TTS_BTN_LOAD"), variant="secondary")
                
                text_input = gr.TextArea(
                    label=_("INFERENCE_LABEL_INPUT_TEXT"), 
                    value="Sistem başarıyla yüklendi ve otonom üretim modunda çalışmaya hazır.", 
                    lines=9
                )
                
                gen_btn = gr.Button(_("INFERENCE_BTN_GENERATE"), variant="primary")

            # --- RIGHT COLUMN ---
            with gr.Column(scale=1):
                gr.Markdown(_("TTS_HEADER_AUDIO_OUT"))
                with gr.Row():
                    ref_audio = gr.Audio(
                        label=_("TTS_LABEL_SPEAKER_REF"), 
                        type="filepath", 
                        value="examples/tr_female.wav"
                    )                     
                with gr.Row():
                    audio_out = gr.Audio(label=_("INFERENCE_LABEL_GEN_RESULT"))
                
        # --- SETTINGS AREAS ---
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

        with gr.Accordion(_("TTS_ACC_ADVANCED"), open=False):
            use_cuda_cb = gr.Checkbox(label=_("TTS_CHK_CUDA"), value=False)
            use_compile_cb = gr.Checkbox(label=_("TTS_CHK_COMPILE"), value=False)
            with gr.Row():
                do_sample = gr.Checkbox(label=_("INFERENCE_CHK_SAMPLE"), value=True)
                temp = gr.Slider(label=_("INFERENCE_SLIDER_TEMP"), minimum=0.1, maximum=1.5, value=0.8)
                top_p = gr.Slider(label=_("INFERENCE_SLIDER_TOP_P"), minimum=0.0, maximum=1.0, value=0.8)
            with gr.Row():
                max_mel = gr.Slider(label=_("INFERENCE_SLIDER_MAX_MEL"), value=1500, minimum=50, maximum=2000)
                max_text_seg = gr.Slider(label=_("INFERENCE_SLIDER_MAX_TEXT"), value=120, minimum=20, maximum=200)

        # --- LOGS ---
        system_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=10, interactive=False)

        # --- EVENT WIRING ---
        def update_ui(method_idx):
            return {
                ref_group: gr.update(visible=(method_idx == 1)),
                vec_group: gr.update(visible=(method_idx == 2)),
                text_group: gr.update(visible=(method_idx == 3)),
                weight_row: gr.update(visible=(method_idx > 0))
            }
        emo_method.change(update_ui, inputs=[emo_method], outputs=[ref_group, vec_group, text_group, weight_row])

        load_btn.click(
            load_standalone_model, 
            inputs=[model_dropdown, use_cuda_cb, use_compile_cb], 
            outputs=[system_log]
        )

        gen_btn.click(
            fn=generate_speech_standalone,
            inputs=[
                model_dropdown, text_input, seed_input, ref_audio, emo_method, emo_upload, emo_weight,
                v1, v2, v3, v4, v5, v6, v7, v8, emo_text,
                do_sample, temp, top_p, max_mel, max_text_seg,
                use_cuda_cb, use_compile_cb, language_dropdown
            ],
            outputs=[audio_out, system_log]
        )

        gr.HTML("<div style='height:10px'></div>")

    return demo