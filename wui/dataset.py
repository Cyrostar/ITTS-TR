import gradio as gr
import os
import re
import io
import json

import torch
import librosa
import gc
import whisper
from datasets import load_dataset, Audio
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline

from core import core
from core.core import _
from core.normalizer import MultilingualNormalizer

# Helper for Sample Rate Mapping
SR_MAP = {
    "16Khz": 16000,
    "22Khz": 22050,
    "24Khz": 24000,
    "44Khz": 44100,
    "48Khz": 48000
}

def handle_stop(current_logs):
    stop_msg = "🛑 Process stopped by user."
    return stop_msg, f"{current_logs}\n{stop_msg}"
    
def handle_resume(current_logs):
    resume_msg = "♻️ Resume triggered by user. Checking for existing files..."
    return resume_msg, f"{current_logs}\n{resume_msg}"   
    
def get_fleurs_subset(lang_code):

    fleurs_mapping = {
        "tr": "tr_tr",      # Turkish (Turkey)
        "en": "en_us",      # English (United States)
        "es": "es_419"      # Spanish (Latin America)
    }
    
    return fleurs_mapping.get(lang_code.lower(), lang_code.lower())
    
def get_hf_datasets(lang):
    json_path = os.path.join(core.path_base, "datasets.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(lang, ["google/fleurs"])
        except Exception as e:
            print(f"Error reading datasets.json: {e}")
            
    return ["google/fleurs"]

# ======================================================
# METHOD 1: HUGGING FACE DATASET PROCESSING
# ======================================================

def process_dataset_ui(dataset_id, output_folder_name, resample_sr, lang, save_every):
    logs = []
    total_ds = 0
    
    def log(msg, current_idx=None):
        """Returns (Formatted Status, Full Logs)."""
        logs.append(msg)
        full_logs = "\n".join(logs)
        if current_idx is not None and total_ds > 0:
            percent = int((current_idx / total_ds) * 100)
            # Format: Progress : x out of y processed : %z
            status_text = f"📊 Progress : {current_idx} out of {total_ds} processed : %{percent}"
            return status_text, full_logs
        return msg, full_logs

    dataset_display_name = output_folder_name or "unknown_dataset"
    output_dir = os.path.join(core.path_base, "datasets", lang, dataset_display_name)
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    target_sr = SR_MAP.get(resample_sr, None)
    normalizer = MultilingualNormalizer(lang=lang, wordify=True, abbreviations=True)

    metadata_lines = []
    existing_filenames = set()
    if os.path.exists(metadata_path):
        yield log("📂 Existing metadata found. Loading for resume...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_lines = [line.strip() for line in f.readlines() if line.strip()]
            for line in metadata_lines:
                parts = line.split("|")
                if len(parts) > 2:
                    existing_filenames.add(parts[2])

    try:

        if dataset_id == "google/fleurs":
            lang = get_fleurs_subset(lang)
            
        yield log(f"⬇️ Loading source: {dataset_id} (Targeting '{lang}' subset)...")
        
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            # Attempt 1: Standard modern dataset loading (uses subset 'configs')
            ds = load_dataset(dataset_id, lang, split="train", token=HF_TOKEN)
        except Exception as e:
            error_msg = str(e).lower()
            # Broadened to catch BOTH script errors AND local cache mismatch errors
            if "scripts are no longer supported" in error_msg or "couldn't find cache" in error_msg or "builderconfig" in error_msg:
                yield log(f"⚠️ Cache/Script mismatch detected. Rerouting '{lang}' to Parquet data directory...")
                # The Parquet bot flattens configs. We must use data_dir instead of config name.
                ds = load_dataset(dataset_id, data_dir=lang, split="train", token=HF_TOKEN, revision="refs/convert/parquet")
            else:
                yield log(f"⚠️ Subset '{lang}' not found. Falling back to default root loading...")
                try:
                    # Attempt 2: Fallback for standard/flat datasets without language subsets
                    ds = load_dataset(dataset_id, split="train", token=HF_TOKEN)
                except Exception as root_e:
                    root_error_msg = str(root_e).lower()
                    if "scripts are no longer supported" in root_error_msg or "couldn't find cache" in root_error_msg:
                        yield log("⚠️ Legacy/Cache issue at root. Rerouting to Parquet branch...")
                        ds = load_dataset(dataset_id, split="train", token=HF_TOKEN, revision="refs/convert/parquet")
                    else:
                        raise root_e
            
    except Exception as e:
        yield log(f"❌ Load Error: {str(e)}")
        return
        
    ds = ds.cast_column("audio", Audio(decode=False))

    os.makedirs(wavs_dir, exist_ok=True)
    processed_count = 0
    
    for i, item in enumerate(ds):
        filename = f"audio_{i:06d}.wav"
        curr_step = i + 1
        
        if filename in existing_filenames:
            processed_count += 1
            continue

        try:
            # Safely extract text across varying dataset schemas
            text = item.get('transcription') or item.get('text') or item.get('sentence') or ""
            
            # --- FFMPEG / PYDUB BYPASS ---
            audio_data = item['audio']
            
            # Pydub securely pipes the raw bytes to your local FFmpeg installation
            clip = AudioSegment.from_file(io.BytesIO(audio_data['bytes']))
            orig_sr = clip.frame_rate

            # Handle Audio Resampling natively via Pydub
            if target_sr and orig_sr != target_sr:
                yield log(f"🔄 Resampling {filename} to {target_sr}Hz", current_idx=curr_step)
                clip = clip.set_frame_rate(target_sr)

            # Force pure mono (Standard TTS format) and export
            clip = clip.set_channels(1)
            clip.export(os.path.join(wavs_dir, filename), format="wav")
            
            # Text Processing Pipeline (Wordifier integrated via Normalizer)
            try:
                clean_sentence = normalizer.normalize(text)
            except Exception as e:
                yield log(f"⚠️ Normalization Error in {filename}: {str(e)}", current_idx=curr_step)
                continue
            
            metadata_lines.append(f"{lang}|{dataset_display_name}|{filename}|{lang}_speaker|{clean_sentence}")
            processed_count += 1

            if processed_count > 0 and processed_count % save_every == 0:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(metadata_lines))
                yield log(f"💾 Progress saved at {processed_count} clips.", current_idx=curr_step)
            
        except Exception as e:
            yield log(f"⚠️ Error in {filename}: {str(e)}", current_idx=curr_step)

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    yield log(f"🎉 DONE! Total in metadata: {len(metadata_lines)} clips.", current_idx=total_ds)

# ======================================================
# METHOD 2: WHISPER LONG AUDIO SLICER
# ======================================================

def process_long_audio_ui(audio_file, dataset_name, batch_size, resample_sr, lang, save_every, max_clip_seconds):
    logs = []
    total_segments = 0

    def log(msg, current_idx=None):
        """Returns (Formatted Status, Full Logs)."""
        logs.append(msg)
        full_logs = "\n".join(logs)
        if current_idx is not None and total_segments > 0:
            percent = int((current_idx / total_segments) * 100)
            status_text = f"📊 Progress : {current_idx} out of {total_segments} processed : %{percent}"
            return status_text, full_logs
        return msg, full_logs

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        yield log("❌ HF_TOKEN not found in environment.")
        return

    if not audio_file:
        yield log("❌ No audio file uploaded.")
        return
        
    normalizer = MultilingualNormalizer(lang=lang, wordify=True, abbreviations=True)
    target_sr = SR_MAP.get(resample_sr, None)

    metadata_lines = []

    dataset_name = dataset_name or "unknown_dataset"
    output_dir = os.path.join(core.path_base, "datasets", lang, dataset_name)
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata_lines = []
    processed_files = set()
    global_clip_index = 0 
    
    if os.path.exists(metadata_path):
        yield log("📂 Existing metadata found. Resuming...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_lines = [line.strip() for line in f.readlines() if line.strip()]
            for line in metadata_lines:
                parts = line.split("|")
                if len(parts) > 2: processed_files.add(parts[2]) 
        
        max_idx = -1
        for fname in processed_files:
            match = re.search(r"audio_(\d+)\.wav", fname)
            if match:
                idx = int(match.group(1))
                if idx > max_idx: max_idx = idx
        if max_idx >= 0: global_clip_index = max_idx + 1 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        yield log("🧬 Initializing Pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        pipeline.to(device)
        
        yield log("🎙️ Running VAD...")
        diarization = pipeline(audio_file)
        
        yield log("🧠 Loading Whisper...")
        model = whisper.load_model("large-v3", device=device)
        
        full_audio = AudioSegment.from_file(audio_file)
        orig_sr = full_audio.frame_rate
        pad_ms = 50 
        saved_count = 0
        segments = list(diarization.itertracks(yield_label=True))
        total_segments = len(segments)
        max_dur_limit = float(max_clip_seconds)

        for i, (segment, _, speaker) in enumerate(segments):
            curr_step = i + 1
            start_ms = max(0, int(segment.start * 1000) - pad_ms)
            end_ms = min(len(full_audio), int(segment.end * 1000) + pad_ms)
            clip_dur = (end_ms - start_ms) / 1000.0
            
            if clip_dur < 1.0: 
                yield log(f"⏩ Segment {curr_step} too short, skipping.", current_idx=curr_step)
                continue

            clip = full_audio[start_ms:end_ms]
            temp_path = os.path.join(output_dir, f"temp_{i}.wav")
            clip.export(temp_path, format="wav")
            
            yield log(f"✍️ Transcribing segment {curr_step}/{total_segments}", current_idx=curr_step)
            task_result = model.transcribe(temp_path, language=lang)
            full_text = task_result["text"].strip()
            whisper_segments = task_result.get("segments", [])
            
            if os.path.exists(temp_path): os.remove(temp_path)
            if len(full_text) < 2: continue

            clips_to_save = []
            if clip_dur <= max_dur_limit:
                clips_to_save.append((clip, full_text))
            else:
                current_start = 0.0
                current_text_parts = []
                for w_seg in whisper_segments:
                    seg_text = w_seg['text'].strip()
                    if (w_seg['end'] - current_start) > max_dur_limit and current_text_parts:
                        sub_audio = clip[int(current_start*1000) : int(w_seg['start']*1000)]
                        clips_to_save.append((sub_audio, " ".join(current_text_parts)))
                        current_start = w_seg['start']
                        current_text_parts = [seg_text]
                    else:
                        current_text_parts.append(seg_text)
                if current_text_parts:
                    clips_to_save.append((clip[int(current_start*1000):], " ".join(current_text_parts)))

            for sub_audio, sub_text in clips_to_save:
                if len(sub_audio) < 500 or len(sub_text) < 2: continue
                
                final_filename = f"audio_{global_clip_index:06d}.wav"
                if final_filename in processed_files:
                    global_clip_index += 1 
                    continue 
                
                if target_sr and orig_sr != target_sr:
                    yield log(f"🔄 Resampling {final_filename}: {orig_sr}Hz -> {target_sr}Hz", current_idx=curr_step)
                    sub_audio = sub_audio.set_frame_rate(target_sr)
                
                final_path = os.path.join(wavs_dir, final_filename)
                sub_audio.set_channels(1).export(final_path, format="wav")
                
                # Text Processing Pipeline (Wordifier integrated via Normalizer)
                clean_text = normalizer.normalize(sub_text)
                clean_text = clean_text.replace('"', '')
                metadata_lines.append(f"{lang}|{dataset_name}|{final_filename}|{lang}_speaker|{clean_text}")
                saved_count += 1
                global_clip_index += 1

            if saved_count > 0 and saved_count % save_every == 0:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(metadata_lines))
                yield log(f"💾 Progress saved: {saved_count} clips.", current_idx=curr_step)

        del model, pipeline
        gc.collect()
        torch.cuda.empty_cache()

        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))
        yield log(f"🎉 DONE! Dataset size: {len(metadata_lines)} clips.", current_idx=total_segments)

    except Exception as e:
        yield log(f"❌ Error: {str(e)}")
        
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    sr_options = ["None", "16Khz", "22Khz", "24Khz", "44Khz", "48Khz"]
    lang_options = core.language_list()
    
    with gr.Blocks() as demo:
        gr.Markdown(_("DATASET_HEADER"))
        gr.Markdown(_("DATASET_DESC"))
        
        with gr.Tab(_("DATASET_TAB_HF")):
            with gr.Row():
                initial_datasets = get_hf_datasets("tr")
                hf_dataset_id = gr.Dropdown(
                    label=_("DATASET_LABEL_HF_ID"),
                    choices=initial_datasets,
                    value=initial_datasets[0] if initial_datasets else None,
                    allow_custom_value=True,
                    scale=2
                )
                hf_lang = gr.Dropdown(label=_("COMMON_LABEL_LANG"), choices=lang_options, value="tr")
                hf_out_name = gr.Textbox(label=_("DATASET_LABEL_TARGET_DIR"), placeholder="dataset_name", value="")                                          
                hf_sr = gr.Dropdown(label=_("DATASET_LABEL_RESAMPLE"), choices=sr_options, value="None")
                hf_save_every = gr.Number(label=_("DATASET_LABEL_SAVE_EVERY"), value=1000, precision=0)
                
            with gr.Row():
                hf_btn = gr.Button(_("COMMON_BTN_START"), variant="primary")
                hf_stop = gr.Button(_("COMMON_BTN_STOP"), variant="stop")
                hf_resume = gr.Button(_("DATASET_BTN_RESUME"))
            
            with gr.Column():
                hf_status = gr.Textbox(label=_("DATASET_LABEL_STATUS"), lines=1, interactive=False)
                hf_logs = gr.Textbox(label=_("DATASET_LABEL_LOGS"), lines=12, autoscroll=True)
            
            
            hf_lang.change(
                fn=lambda l: gr.Dropdown(
                    choices=get_hf_datasets(l), 
                    value=get_hf_datasets(l)[0] if get_hf_datasets(l) else None
                ),
                inputs=[hf_lang],
                outputs=[hf_dataset_id]
            )
            hf_event = hf_btn.click(
                process_dataset_ui, 
                inputs=[hf_dataset_id, hf_out_name, hf_sr, hf_lang, hf_save_every], 
                outputs=[hf_status, hf_logs],
                show_progress="hidden"
            )
            hf_resume.click(
                fn=handle_resume, 
                inputs=[hf_logs], 
                outputs=[hf_status, hf_logs]
            ).then(
                fn=process_dataset_ui, 
                inputs=[hf_dataset_id, hf_out_name, hf_sr, hf_lang, hf_save_every], 
                outputs=[hf_status, hf_logs],
                show_progress="hidden"
            )
            hf_stop.click(
                fn=handle_stop, 
                inputs=[hf_logs],
                outputs=[hf_status, hf_logs],
                cancels=[hf_event]
            )

        with gr.Tab(_("DATASET_TAB_CUSTOM")):
            with gr.Row():
                audio_input = gr.File(label=_("COMMON_LABEL_UPLOAD"), file_types=["audio"], type="filepath")
            
            with gr.Row():
                wx_out_name = gr.Textbox(label=_("DATASET_LABEL_TARGET_DIR"), value="my_custom_dataset")
                wx_lang = gr.Dropdown(label=_("COMMON_LABEL_LANG"), choices=lang_options, value="tr")
                wx_sr = gr.Dropdown(label=_("DATASET_LABEL_RESAMPLE"), choices=sr_options, value="None")
                wx_save_every = gr.Number(label=_("DATASET_LABEL_SAVE_EVERY"), value=100, precision=0)
                wx_max_dur = gr.Number(label=_("DATASET_LABEL_MAX_DUR"), value=15)
                
            with gr.Row():
                wx_btn = gr.Button(_("COMMON_BTN_START"), variant="primary")
                wx_stop = gr.Button(_("COMMON_BTN_STOP"), variant="stop")
                wx_resume = gr.Button(_("COMMON_BTN_RESUME"))
                
            with gr.Column():
                wx_status = gr.Textbox(label=_("DATASET_LABEL_STATUS"), lines=1, interactive=False)
                wx_logs = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=12, autoscroll=True)
            
            wx_event = wx_btn.click(
                process_long_audio_ui, 
                inputs=[audio_input, wx_out_name, gr.State(16), wx_sr, wx_lang, wx_save_every, wx_max_dur], 
                outputs=[wx_status, wx_logs],
                show_progress="hidden"
            )
            wx_resume.click(
                fn=handle_resume, 
                inputs=[wx_logs], 
                outputs=[wx_status, wx_logs]
            ).then(
                fn=process_long_audio_ui, 
                inputs=[audio_input, wx_out_name, gr.State(16), wx_sr, wx_lang, wx_save_every, wx_max_dur], 
                outputs=[wx_status, wx_logs],
                show_progress="hidden"
            )
            wx_stop.click(
                fn=handle_stop, 
                inputs=[wx_logs], 
                outputs=[wx_status, wx_logs], 
                cancels=[wx_event]
            )
            
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("dataset"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")

    return demo