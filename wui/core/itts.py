import os
import time
import json
import re
import random
import torch
import torchaudio
import torch.nn.functional as F
import safetensors
from subprocess import CalledProcessError

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from core.spice import GenericSpiceTokenizer
from core.normalizer import TurkishWalnutNormalizer, TurkishGenericWordifier

# ==========================================
# Helpers
# ==========================================
def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()
    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class IndexTTS2:
    
    def __init__(
        self, 
        model_dir="checkpoints", 
        project_dir=None, 
        train_dir=None, 
        loaded_project_name=None, 
        use_fp16=False, 
        device=None,
        use_cuda_kernel=None, 
        use_deepspeed=False, 
        use_accel=False, 
        use_torch_compile=False, 
        do_load=True, 
        case_format="lowercase", 
        tok_type=None
        ):
        
        # 1. Setup Directories
        self.model_dir = model_dir
        self.project_dir = project_dir
        self.train_dir = train_dir
        self.loaded_project = loaded_project_name
        self.use_cuda_kernel = use_cuda_kernel
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile
        self.use_deepspeed = use_deepspeed
        self.case_format = case_format
        self.tok_type = tok_type
        
        # 2. Setup Device
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        # 3. Load Config (Prioritize Train Dir, then Project Dir, then Fallback)
        potential_configs = []
        if self.train_dir:
            potential_configs.append(os.path.join(self.train_dir, "config.yaml"))
            potential_configs.append(os.path.join(self.train_dir, "config_original.yaml"))
        if self.project_dir:
            potential_configs.append(os.path.join(self.project_dir, "configs", "config.yaml"))
            potential_configs.append(os.path.join(self.project_dir, "config.yaml"))
        potential_configs.append(os.path.join(self.model_dir, "config.yaml"))

        self.cfg_path = potential_configs[-1] # Default fallback
        for p in potential_configs:
            if os.path.exists(p):
                self.cfg_path = p
                break

        print(f">> Loading Config from: {self.cfg_path}")
        self.cfg = OmegaConf.load(self.cfg_path)
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Initialize attributes
        self.qwen_emo = None
        self.gpt = None
        self.s2mel = None
        self.bigvgan = None
        self.tokenizer = None
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

        # --- Initialize Cache Attributes ---
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None
        # -----------------------------------

        # If blocking load is requested (default), do it now
        if do_load:
            for _ in self.load_resources():
                pass

    def _resolve_path(self, filename, subfolder=None):
        if not filename: return None
        basename = os.path.basename(filename)
        candidates = []
        if self.train_dir:
                candidates.append(os.path.join(self.train_dir, filename))
                candidates.append(os.path.join(self.train_dir, basename))
        if self.project_dir:
            candidates.append(os.path.join(self.project_dir, filename))
            candidates.append(os.path.join(self.project_dir, basename))
            if subfolder:
                candidates.append(os.path.join(self.project_dir, subfolder, basename))
        candidates.append(os.path.join(self.model_dir, filename))
        candidates.append(os.path.join(self.model_dir, basename))

        for p in candidates:
            if os.path.exists(p):
                return p
        return os.path.join(self.model_dir, filename)

    def load_resources(self):
        yield f"⚙️ Config Loaded: {os.path.basename(self.cfg_path)}"

        # 1. Qwen Emotion
        yield "🧠 Loading Emotion Model (Qwen)..."
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))
        yield "   ✅ Emotion Model Ready"

        # 2. GPT
        yield "🧠 Loading Semantic Model (GPT)..."
        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        
        # --- GPT CHECKPOINT LOGIC (Strict) ---
        if self.loaded_project == "itts_base_model":
             self.gpt_path = self._resolve_path(self.cfg.gpt_checkpoint)
        else:
            # Custom behavior: Strict lookup in train_dir
            # Priority: gpt.pth -> latest.pth -> (fail/fallback to strictly train_dir)
            p_gpt = os.path.join(self.train_dir, "gpt.pth")
            p_latest = os.path.join(self.train_dir, "latest.pth")
            
            if os.path.exists(p_gpt):
                self.gpt_path = p_gpt
            elif os.path.exists(p_latest):
                self.gpt_path = p_latest
            else:
                # If neither exists, point to gpt.pth in train_dir so it logs/fails 
                # correctly instead of silently loading the base model.
                self.gpt_path = p_gpt 
        # -------------------------------------

        yield f"   📂 GPT Path: {self.gpt_path}"
        
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        
        use_ds = self.use_deepspeed
        if use_ds:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError):
                use_ds = False
        self.gpt.post_init_gpt2_config(use_deepspeed=use_ds, kv_cache=True, half=self.use_fp16)
        yield "   ✅ GPT Ready"

        # 3. Features
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()

        # 4. S2Mel
        yield "🌊 Loading Acoustic Model (S2Mel)..."
        s2mel_path = self._resolve_path(self.cfg.s2mel_checkpoint)
        yield f"   📂 S2Mel Path: {s2mel_path}"
        
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        if self.use_torch_compile:
            self.s2mel.enable_torch_compile()
        self.s2mel.eval()

        campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        yield "   ✅ S2Mel Ready"

        # 5. Vocoder
        yield "🔊 Loading Vocoder (BigVGAN)..."
        bigvgan_name = self.cfg.vocoder.name
        yield f"   ℹ️ Model: {bigvgan_name} (Kernel: {self.use_cuda_kernel})"
        
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        yield "   ✅ Vocoder Ready"

        # 6. Tokenizer
        yield "📝 Loading Tokenizer..."
        
        if self.train_dir:
            self.bpe_path = os.path.join(self.train_dir, "bpe.model")
            # Strict Existence Check
            if not os.path.exists(self.bpe_path):
                error_msg = f"   ❌ CRITICAL ERROR: 'bpe.model' not found in training folder: {self.train_dir}"
                yield error_msg
                raise FileNotFoundError(error_msg) # This stops the loading process and triggers the UI catch-all
        else:
            self.bpe_path = None
            
        yield f"   📂 BPE Path: {self.bpe_path if self.bpe_path else 'Default (Base Model Context)'}"
        
        if self.loaded_project == "itts_base_model":
            yield "   ℹ️ Using Standard Normalizer (Official)"
            self.normalizer = TextNormalizer()
            self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        else:
            if self.tok_type == "indextts":
                yield "   ℹ️ Using IndexTTS Tokenizer with dedicated Turkish Walnut Normalizer (Wordify: True)"
                self.normalizer = TurkishWalnutNormalizer(wordify=True)
                self.tokenizer = TextTokenizer(vocab_file=self.bpe_path, normalizer=self.normalizer)
            else:
                yield f"   ℹ️ Using ITTS-TR Tokenizer (GenericSpiceTokenizer) [Case: {self.case_format}]"
                self.normalizer = TurkishWalnutNormalizer(upper=(self.case_format == "uppercase"), wordify=True)
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

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True)
        feat = vq_emb.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        if not wavs or interval_silence <= 0: return wavs
        channel_size = wavs[0].size(0)
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        if not wavs or interval_silence <= 0: return wavs
        channel_size = wavs[0].size(0)
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)
        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1: wavs_list.append(sil_tensor)
        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None: self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        try:
            audio, native_sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr is not None and sr != native_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=sr)
                audio = resampler(audio)
                current_sr = sr
            else:
                current_sr = native_sr
            max_audio_samples = int(max_audio_length_seconds * current_sr)
            if audio.shape[1] > max_audio_samples:
                if verbose:
                    print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples}")
                audio = audio[:, :max_audio_samples]
            return audio, current_sr
        except Exception as e:
            print(f"ERROR in _load_and_cut_audio: {str(e)}")
            raise e

    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        if apply_bias:
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]
        emo_sum = sum(emo_vector)
        # ALLOW FULL STRENGTH (Raised Cap to 1.0)
        if emo_sum > 1.0:
            scale_factor = 1.0 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]
        return emo_vector

    def infer(self, spk_audio_prompt, text, output_path, **kwargs):
        if kwargs.get('stream_return'):
            return self.infer_generator(spk_audio_prompt, text, output_path, **kwargs)
        else:
            try:
                gen = self.infer_generator(spk_audio_prompt, text, output_path, **kwargs)
                results = []
                for item in gen:
                    if isinstance(item, str) and item.startswith("LOG:"):
                        continue
                    results.append(item)
                return results[0] if results else None
            except IndexError:
                return None
                
    def trim_trailing_silence(self, wav, sr, top_db=30):
        """
        Trims silence from the end using native PyTorch to avoid librosa errors.
        """
        try:
            original_duration = wav.size(1) / sr
            
            # 1. Calculate absolute amplitude
            # wav shape is (1, T)
            y_abs = wav.abs()
            
            # 2. Determine threshold relative to max amplitude
            max_val = y_abs.max()
            
            # Safety check for completely silent audio
            if max_val <= 1e-5:
                return wav, "LOG: Audio is silent; no trimming performed."
            
            # Convert dB to linear scalar: 10^(-top_db/20)
            threshold = max_val * (10 ** (-top_db / 20))
            
            # 3. Find the last sample that exceeds the threshold
            # torch.nonzero returns indices of non-zero elements
            non_silent_indices = torch.nonzero(y_abs > threshold, as_tuple=True)
            
            # If no audio exceeds threshold
            if non_silent_indices[1].numel() == 0:
                return wav, "LOG: Audio below threshold; no trimming performed."

            # Get the last index on the time axis (dim 1)
            last_idx = non_silent_indices[1][-1].item()
            
            # 4. Add a buffer (0.1s) for natural release
            buffer_samples = int(0.1 * sr)
            end_sample = min(wav.size(1), last_idx + buffer_samples)
            
            # 5. Slice
            trimmed_wav = wav[:, :end_sample]
            
            # Calculate stats
            new_duration = trimmed_wav.size(1) / sr
            removed_sec = original_duration - new_duration
            
            log_msg = f"LOG: Trailing silence removed ({removed_sec:.2f}s removed)."
            return trimmed_wav, log_msg
            
        except Exception as e:
            # Fallback to returning original audio if anything unexpected happens
            return wav, f"LOG: Trimming skipped due to error: {str(e)}"

    def infer_generator(
        self, 
        spk_audio_prompt, 
        text, 
        output_path,
        emo_audio_prompt=None, 
        emo_alpha=1.0, 
        emo_vector=None,
        use_emo_text=False, 
        emo_text=None, 
        use_random=False, 
        interval_silence=200,
        verbose=False, 
        max_text_tokens_per_segment=120, 
        stream_return=False, 
        quick_streaming_tokens=0,
        language="Auto", 
        **generation_kwargs
        ):
        
        yield "LOG: Starting inference..."
        self._set_gr_progress(0, "starting inference...")
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            emo_audio_prompt = None

        if use_emo_text:
            if emo_text is None: emo_text = text
            yield "LOG: Analyzing emotion from text..."
            emo_dict = self.qwen_emo.inference(emo_text)
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0

        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            yield "LOG: Processing reference audio..."
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.campplus_model(feat.unsqueeze(0))

            prompt_condition = self.s2mel.models['length_regulator'](S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            
            manual_strength = torch.sum(weight_vector).item()
            orig_strength = max(0.0, 1.0 - manual_strength)
            yield f"LOG: Emotion Mixing -> [Manual: {manual_strength*100:.1f}% | Original: {orig_strength*100:.1f}%]"

            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            
            emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(self.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
            emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        yield "LOG: Tokenizing text..."
        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens=quick_streaming_tokens)
        segments_count = len(segments)
        
        # Generation Loop
        wavs = []
        silence = None
        sampling_rate = 22050
        
        # Unpack kwargs
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)

        for seg_idx, sent in enumerate(segments):
            yield f"LOG: Synthesizing segment {seg_idx + 1}/{segments_count}..."
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count, f"speech synthesis {seg_idx + 1}/{segments_count}...")

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            
            if language == "TR":
                text_tokens = [3] + text_tokens
                yield "LOG: Injected TR language ID (3)"
            elif language == "EN":
                text_tokens = [4] + text_tokens
                yield "LOG: Injected EN language ID (4)"
                
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb, emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )
                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb, text_tokens, emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True, top_p=top_p, top_k=top_k, temperature=temperature,
                        num_return_sequences=autoregressive_batch_size, length_penalty=length_penalty,
                        num_beams=num_beams, repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens, **generation_kwargs
                    )

                # Process codes
                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens).to(self.device)

                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent, text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes, torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec, use_speed=use_speed,
                    )

                # S2Mel and BigVGAN
                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2) + latent
                    target_lengths = (code_lens * 1.72).long()
                    cond = self.s2mel.models['length_regulator'](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    vc_target = self.s2mel.models['cfm'].inference(
                        cat_condition, torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        ref_mel, style, None, 25, inference_cfg_rate=0.7
                    )
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0).squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                wavs.append(wav.cpu())
                if stream_return:
                    yield wav.cpu()
                    if silence is None:
                        silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                    yield silence

        yield "LOG: Concatenating and trimming audio..."
        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1).cpu()

        # ==========================================
        # Trim silence
        # ==========================================
        wav, trim_log = self.trim_trailing_silence(wav, sampling_rate, top_db=25)
        yield trim_log
        
        yield "LOG: Saving to disk..."
        if output_path:
            if os.path.isfile(output_path): os.remove(output_path)
            if os.path.dirname(output_path) != "": os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            if stream_return: return None
            yield output_path
        else:
            if stream_return: return None
            wav_data = wav.type(torch.int16).numpy().T
            yield (sampling_rate, wav_data)
            
class QwenEmotion:
    
    def __init__(self, model_dir):
        
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",
            device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
            "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            "低落", "melancholy", "melancholic", "depression", "depressed", "gloomy",
            "üzgün", "kederli", "hüzünlü", "depresif", "melankolik", "karanlık", "çaresiz", "mutsuz"
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0
        return emotion_dict

    def inference(self, text_input):
        messages = [{"role": "system", "content": f"{self.prompt}"}, {"role": "user", "content": f"{text_input}"}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768, pad_token_id=self.tokenizer.eos_token_id)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            content = {m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)}
        
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
        return self.convert(content)