import os
import sys
import time
import json
import glob
import concurrent.futures
from random import shuffle

from core import core

import numpy as np
from scipy import signal
import librosa
import soundfile as sf

import torch
from torch.utils.data import DataLoader

# --- RVC INTERNAL IMPORT FIX ---
# Append the rvc/train directory to sys.path so RVC scripts can resolve their own local imports
rvc_train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc", "train"))
if rvc_train_dir not in sys.path:
    sys.path.append(rvc_train_dir)

# RVC: Data & Utils
from rvc.lib.utils import load_audio_16k, load_embedding
from rvc.train.utils import HParams
from rvc.lib.algorithm import commons
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.train.data_utils import TextAudioLoaderMultiNSFsid, TextAudioCollateMultiNSFsid

# RVC: Extractors
from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

# RVC: Architectures & Training
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.train.mel_processing import mel_spectrogram_torch
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.process.extract_model import extract_model

# ==========================================
# RVC PREPROCESSOR
# ==========================================

def _process_file_worker(args):
    """Top-level worker function to bypass multiprocessing pickling restrictions."""
    instance, filename, dataset_path, output_dir, output_dir_16k = args
    try:
        input_path = os.path.join(dataset_path, filename)
        audio, audio_16k = instance._normalize_and_filter(input_path)
        
        if audio is not None and audio_16k is not None:
            # RVC Filelist Generator için Hoparlör ID'sini (0) dosya ismine zorla ekle
            out_name = filename if filename.startswith("0_") else f"0_0_{filename}"
            
            # Save the target sample rate version (e.g., 40k or 48k from UI)
            out_path = os.path.join(output_dir, out_name)
            sf.write(out_path, audio, instance.target_sr)
            
            # Save the 16k version for HuBERT feature extraction
            out_16k_path = os.path.join(output_dir_16k, out_name)
            sf.write(out_16k_path, audio_16k, instance.sr_16k)
            return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    return False

class rvc_preprocessor:
    """Standalone ITTS-TR Preprocessor replacing external Applio dependencies."""
    def __init__(self, target_sr=40000):
        self.target_sr = target_sr
        self.sr_16k = 16000
        # 5th-order Butterworth high-pass filter at 48Hz
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.target_sr)

    def _normalize_and_filter(self, audio_path):
        """Loads, high-pass filters, and normalizes the audio amplitude."""
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # --- SHORT AUDIO FAILSAFE ---
        # RVC requires minimum lengths for tensor segment slicing during training. 
        # Reject chunks shorter than 1.5 seconds to prevent CUDA Gather crashes.
        if len(audio) < int(self.target_sr * 1.5):
            return None, None
            
        # Apply high-pass filter to remove low-frequency rumble
        audio = signal.lfilter(self.b_high, self.a_high, audio)
        
        # Max amplitude normalization
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5: return None, None # Skip corrupted/extreme clipping
        audio = (audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * audio
        
        # Generate 16kHz copy for HuBERT
        audio_16k = librosa.resample(audio, orig_sr=self.target_sr, target_sr=self.sr_16k, res_type="soxr_vhq")
        
        return audio, audio_16k

    def process_dataset(self, dataset_path, output_dir, output_dir_16k, cpu_cores):
        """Iterates through the workspace dataset and processes chunks using multiprocessing."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_16k, exist_ok=True)
        
        audio_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        # Bundle the arguments together into a tuple list for the map function
        args_list = [(self, f, dataset_path, output_dir, output_dir_16k) for f in audio_files]

        processed_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            # Send the top-level worker and the bundled arguments to the executor
            results = list(executor.map(_process_file_worker, args_list))
            processed_count = sum(1 for r in results if r)
            
        return f"Native ITTS-TR DSP Processing completed. Saved {processed_count}/{len(audio_files)} files to {self.target_sr}Hz and 16kHz formats."

# ==========================================
# RVC FEATURE EXTRACTOR
# ==========================================

def _f0_worker(args):
    """Top-level worker to extract Pitch (F0). Initializes model internally to bypass pickling."""
    file_chunk, f0_method, device = args

    if f0_method in ("crepe", "crepe-tiny"):
        model = CREPE(device=device, sample_rate=16000, hop_size=160)
    elif f0_method == "rmvpe":
        model = RMVPE(device=device, sample_rate=16000, hop_size=160)
    elif f0_method == "fcpe":
        model = FCPE(device=device, sample_rate=16000, hop_size=160)

    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    for file_info in file_chunk:
        inp_path, opt_path_full, opt_path_coarse, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full): continue
        
        try:
            np_arr = load_audio_16k(inp_path)
            if f0_method == "crepe": feature_pit = model.get_f0(np_arr, f0_min, f0_max, None, "full")
            elif f0_method == "crepe-tiny": feature_pit = model.get_f0(np_arr, f0_min, f0_max, None, "tiny")
            elif f0_method == "rmvpe": feature_pit = model.get_f0(np_arr, filter_radius=0.03)
            elif f0_method == "fcpe": feature_pit = model.get_f0(np_arr, None, filter_radius=0.006)

            np.save(opt_path_full, feature_pit, allow_pickle=False)

            # Coarse F0 calculation
            f0_mel = 1127.0 * np.log(1.0 + feature_pit / 700.0)
            f0_mel = np.clip((f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1, 1, 255)
            np.save(opt_path_coarse, np.rint(f0_mel).astype(int), allow_pickle=False)
        except Exception as e:
            print(f"F0 Error on {inp_path}: {e}")
    return True

def _embed_worker(args):
    """Top-level worker to extract Semantic Embeddings (HuBERT)."""
    file_chunk, embedder_model, device = args

    model = load_embedding(embedder_model, None).to(device).float()
    model.eval()

    for file_info in file_chunk:
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path): continue
        
        try:
            feats = torch.from_numpy(load_audio_16k(wav_file_path)).to(device).float()
            feats = feats.view(1, -1)
            with torch.no_grad():
                result = model(feats)["last_hidden_state"]
            feats_out = result.squeeze(0).float().cpu().numpy()
            if not np.isnan(feats_out).any():
                np.save(out_file_path, feats_out, allow_pickle=False)
        except Exception as e:
            print(f"Embedding Error on {wav_file_path}: {e}")
    return True

class rvc_extractor:
    """Standalone ITTS-TR Extractor replacing external Applio dependencies."""
    def __init__(self):
        pass

    def extract_features(self, exp_dir, f0_method, embedder_model="contentvec", sample_rate=40000):

        wav_path = os.path.join(exp_dir, "sliced_audios_16k")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"16kHz audio directory not found at: {wav_path}")

        # Setup standard RVC output directories
        os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

        files = []
        for file in glob.glob(os.path.join(wav_path, "*.wav")):
            file_name = os.path.basename(file)
            files.append([
                file,
                os.path.join(exp_dir, "f0", file_name + ".npy"),
                os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
                os.path.join(exp_dir, "extracted", file_name.replace(".wav", ".npy"))
            ])

        if not files:
            return "❌ Error: No sliced 16k files found. Please run Preprocessing first."

        device = "cuda:0" # Force primary GPU 

        # Execute GPU workers in isolated processes to free VRAM after completion
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            list(executor.map(_f0_worker, [(files, f0_method, device)]))
            list(executor.map(_embed_worker, [(files, embedder_model, device)]))

        # Compile final configuration manifests
        generate_config(sample_rate, exp_dir)
        generate_filelist(exp_dir, sample_rate, 2)

        return f"Successfully extracted F0 ({f0_method}) and Embeddings ({embedder_model}) for {len(files)} chunks."
        
# ==========================================
# RVC NEURAL NETWORK TRAINER
# ==========================================

def _train_worker(args):
    """Isolated PyTorch training loop to prevent VRAM memory leaks in Gradio."""
    (exp_dir, sr, total_epochs, batch_size, save_every, cache_gpu, f0_method) = args
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Configurations
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "r") as f:
        hps = HParams(**json.load(f))
        
    hps.data.training_files = os.path.join(exp_dir, "filelist.txt")
    
    # Dynamically determine the speaker dimension from Phase 2
    spk_dim = 109
    model_info_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            spk_dim = json.load(f).get("speakers_id", 109)
    hps.model.spk_embed_dim = spk_dim

    # 2. Initialize Datasets (Standardized for single-node execution)
    train_dataset = TextAudioLoaderMultiNSFsid(hps.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_loader = DataLoader(
        train_dataset, 
        num_workers=2, 
        shuffle=True, 
        pin_memory=True, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )

    # 3. Initialize Networks (Generator and Discriminator)
    net_g = Synthesizer(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
        use_f0=True,
        sr=hps.data.sample_rate,
        vocoder="HiFi-GAN",
        checkpointing=False,
        randomized=True,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, checkpointing=False).to(device)

    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    
    fn_mel_loss = torch.nn.L1Loss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    global_step = 0

    net_g.train()
    net_d.train()

    # 4. The Epoch Loop
    for epoch in range(1, total_epochs + 1):
        for batch_idx, info in enumerate(train_loader):
            info = [tensor.to(device, non_blocking=True) for tensor in info]
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Forward Pass Generator
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model_output
                
                # --- IMMUNITY FIX: DYNAMIC TENSOR PADDING ---
                # Prevents CUDA out-of-bounds crashes when PyDub slices don't perfectly align with hop_length frames
                max_idx = int(ids_slice.max().item())
                max_required_len = (max_idx * hps.data.hop_length) + hps.train.segment_size
                if wave.size(-1) < max_required_len:
                    pad_size = max_required_len - wave.size(-1)
                    wave = torch.nn.functional.pad(wave, (0, pad_size))
                # --------------------------------------------
                
                wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3)
                
                # Discriminator Forward & Loss
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            
            # Discriminator Backward Pass
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.step(optim_d)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Generator Loss Calculation
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                
                wave_mel = mel_spectrogram_torch(wave.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sample_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sample_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
                
                loss_mel = fn_mel_loss(wave_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            # Generator Backward Pass
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

        print(f"Epoch: {epoch}/{total_epochs} | Gen Loss: {loss_gen_all.item():.4f} | Disc Loss: {loss_disc.item():.4f}")

        # 5. Model Checkpointing
        if epoch % save_every == 0 or epoch == total_epochs:
            model_name = os.path.basename(exp_dir)
            infer_path = os.path.join(exp_dir, f"{model_name}_{epoch}e.pth")
            extract_model(
                ckpt=net_g.state_dict(),
                sr=hps.data.sample_rate,
                name=model_name,
                model_path=infer_path,
                epoch=epoch,
                step=global_step,
                hps=hps,
                overtrain_info="",
                vocoder="HiFi-GAN"
            )

    return True
    
def _ensure_mute_files(sample_rate):
    """Generates 3-second silent audio files matching RVC's exact hardcoded filelist paths."""
    import os
    import numpy as np
    import soundfile as sf
    
    mute_dir = os.path.join(core.path_base, "logs", "mute")
    
    os.makedirs(os.path.join(mute_dir, "sliced_audios"), exist_ok=True)
    os.makedirs(os.path.join(mute_dir, "sliced_audios_16k"), exist_ok=True)
    os.makedirs(os.path.join(mute_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(mute_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(mute_dir, "extracted"), exist_ok=True)
        
    # Strictly use RVC's hardcoded names (No 0_0_ prefix here)
    sr_path = os.path.join(mute_dir, "sliced_audios", f"mute{sample_rate}.wav")
    sr_16k_path = os.path.join(mute_dir, "sliced_audios_16k", "mute16000.wav")
    
    f0_path = os.path.join(mute_dir, "f0", "mute.wav.npy")
    f0_v_path = os.path.join(mute_dir, "f0_voiced", "mute.wav.npy")
    embed_path = os.path.join(mute_dir, "extracted", "mute.npy")
    
    # 3 seconds of silence to prevent CUDA gather crashes
    sr_len = sample_rate * 3
    sr_16k_len = 16000 * 3
    frames = 300 
    
    if not os.path.exists(sr_path): sf.write(sr_path, np.zeros(sr_len, dtype=np.float32), sample_rate)
    if not os.path.exists(sr_16k_path): sf.write(sr_16k_path, np.zeros(sr_16k_len, dtype=np.float32), 16000)
    
    if not os.path.exists(f0_path): np.save(f0_path, np.zeros(frames, dtype=np.float32))
    if not os.path.exists(f0_v_path): np.save(f0_v_path, np.zeros(frames, dtype=np.int32))
    if not os.path.exists(embed_path): np.save(embed_path, np.zeros((frames, 768), dtype=np.float32))

# ==========================================
# RVC NEURAL NETWORK TRAINER (SUBPROCESS BRIDGE)
# ==========================================

class rvc_trainer:
    """Triggers the external Applio train.py script via Subprocess to protect Gradio UI."""
    def __init__(self):
        pass

    def run_training(self, exp_dir, sample_rate, total_epochs, batch_size, save_every, cache_gpu, f0_method="rmvpe"):
        import subprocess
        
        # 1. Validate Dataset
        if not os.path.exists(os.path.join(exp_dir, "extracted")):
            return "❌ Error: Feature extraction not found. Please run Phase 2 first."
            
        _ensure_mute_files(sample_rate)
        
        # 2. Generate required filelists
        generate_config(sample_rate, exp_dir)
        generate_filelist(exp_dir, sample_rate, 2)
        
        print(f"🚀 Starting External PyTorch Training for {total_epochs} epochs...")

        # 3. Resolve Paths
        train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc", "train", "train.py"))
        
        pretrain_g = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc", "models", "pretraineds", "pretraineds_v2", f"f0G{sample_rate}.pth"))
        pretrain_d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc", "models", "pretraineds", "pretraineds_v2", f"f0D{sample_rate}.pth"))
        
        if not os.path.exists(pretrain_g): pretrain_g = ""
        if not os.path.exists(pretrain_d): pretrain_d = ""
        
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        os.makedirs(assets_dir, exist_ok=True)
        global_config_path = os.path.join(assets_dir, "config.json")
        if not os.path.exists(global_config_path):
            with open(global_config_path, "w") as f:
                json.dump({"version": "v2", "precision": "fp16"}, f, indent=4)

        # 4. Construct Command Line Arguments for train.py
        cmd = [
            sys.executable,
            train_script,
            str(exp_dir),              # sys.argv[1]: Absolute path to workspace
            str(save_every),           # sys.argv[2]: save_every_epoch
            str(total_epochs),         # sys.argv[3]: total_epoch
            pretrain_g,                # sys.argv[4]: pretrainG
            pretrain_d,                # sys.argv[5]: pretrainD
            "0",                       # sys.argv[6]: gpus (Force primary GPU)
            str(batch_size),           # sys.argv[7]: batch_size
            str(sample_rate),          # sys.argv[8]: sample_rate
            "False",                   # sys.argv[9]: save_only_latest
            "True",                    # sys.argv[10]: save_every_weights
            str(cache_gpu),            # sys.argv[11]: cache_data_in_gpu
            "False",                   # sys.argv[12]: overtraining_detector
            "50",                      # sys.argv[13]: overtraining_threshold
            "False",                   # sys.argv[14]: cleanup
            "HiFi-GAN",                # sys.argv[15]: vocoder
            "False"                    # sys.argv[16]: checkpointing
        ]

        # 5. Execute Subprocess
        try:
            # Popen streams the PyTorch training logs directly into your terminal
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            process.communicate() # Blocks Gradio UI until training finishes

            # 2333333 is Applio's custom success exit code
            if process.returncode in [0, 2333333]:
                return f"✅ Training complete! {total_epochs} epochs finished. Model saved in {exp_dir}."
            else:
                return f"❌ Training failed with exit code {process.returncode}. Check terminal for errors."
        except Exception as e:
            return f"❌ Subprocess execution failed: {str(e)}"
            
# ==========================================
# RVC FAISS INDEX GENERATOR
# ==========================================

def _index_worker(args):
    """Isolated worker to compile the FAISS index without blocking the Gradio UI."""
    (exp_dir,) = args
    import os
    import numpy as np
    import faiss

    feature_dir = os.path.join(exp_dir, "extracted")
    if not os.path.exists(feature_dir):
        return False, "❌ Error: Feature directory not found. Please run Phase 2."

    npys = []
    for name in os.listdir(feature_dir):
        if name.endswith(".npy"):
            npys.append(np.load(os.path.join(feature_dir, name)))

    if not npys:
        return False, "❌ Error: No extracted embeddings found."

    # Concatenate all features into a single array
    npys = np.concatenate(npys, axis=0)

    # RVC v2 strictly uses 768 dimensions for ContentVec/HuBERT
    embedding_dim = 768 
    n_samples = npys.shape[0]
    
    # Calculate optimal IVF clusters based on dataset size
    n_ivf = min(int(16 * np.sqrt(n_samples)), n_samples // 39)
    if n_ivf < 1: n_ivf = 1

    try:
        # Build and train the FAISS Index
        index = faiss.index_factory(embedding_dim, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        
        index.train(npys)
        index.add(npys)

        model_name = os.path.basename(exp_dir)
        index_filepath = os.path.join(exp_dir, f"added_{model_name}_v2.index")
        faiss.write_index(index, index_filepath)
        
        return True, f"✅ FAISS Index successfully trained on {n_samples} frames and saved to:\n{index_filepath}"
    except Exception as e:
        return False, f"❌ FAISS Indexing Error: {str(e)}"

class rvc_indexer:
    """Standalone ITTS-TR FAISS Index Generator."""
    def __init__(self):
        pass

    def generate_index(self, exp_dir):
        """Triggers the isolated FAISS indexing process."""
        print(f"📚 Starting FAISS Index generation for {exp_dir}...")
        
        # Execute in isolated process to prevent memory spikes
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_index_worker, (exp_dir,))
            success, msg = future.result()

        return msg