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
        "tr": "tr_tr",      # Turkish
        "en": "en_us",      # English
        "es": "es_419"      # Spanish
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
            
        HF_TOKEN = os.environ.get("HF_TOKEN")
        cache_dir = os.path.join(core.path_root, "models", "hf", "datasets")
        os.makedirs(cache_dir, exist_ok=True)
        
        if dataset_id == "erenfazlioglu/turkishvoicedataset":
            yield log(f"⬇️ Loading source: {dataset_id} (No subset needed)...")
            ds = load_dataset(dataset_id, split="train", token=HF_TOKEN, cache_dir=cache_dir)
        else:
            yield log(f"⬇️ Loading source: {dataset_id} (Targeting '{lang}' subset)...")
            
            try:
                # Attempt 1: Standard modern dataset loading (uses subset 'configs')
                ds = load_dataset(dataset_id, lang, split="train", token=HF_TOKEN, cache_dir=cache_dir)
            except Exception as e:
                error_msg = str(e).lower()
                # Broadened to catch BOTH script errors AND local cache mismatch errors
                if "scripts are no longer supported" in error_msg or "couldn't find cache" in error_msg or "builderconfig" in error_msg:
                    yield log(f"⚠️ Cache/Script mismatch detected. Rerouting '{lang}' to Parquet data directory...")
                    # The Parquet bot flattens configs. We must use data_dir instead of config name.
                    ds = load_dataset(dataset_id, data_dir=lang, split="train", token=HF_TOKEN, revision="refs/convert/parquet", cache_dir=cache_dir)
                else:
                    yield log(f"⚠️ Subset '{lang}' not found. Falling back to default root loading...")
                    try:
                        # Attempt 2: Fallback for standard/flat datasets without language subsets
                        ds = load_dataset(dataset_id, split="train", token=HF_TOKEN, cache_dir=cache_dir)
                    except Exception as root_e:
                        root_error_msg = str(root_e).lower()
                        if "scripts are no longer supported" in root_error_msg or "couldn't find cache" in root_error_msg:
                            yield log("⚠️ Legacy/Cache issue at root. Rerouting to Parquet branch...")
                            ds = load_dataset(dataset_id, split="train", token=HF_TOKEN, revision="refs/convert/parquet", cache_dir=cache_dir)
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

def process_long_audio_ui(audio_file, dataset_name, batch_size, resample_sr, lang, save_every, max_clip_seconds, whisper_model_size, cpu_workers):
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
        # Explicitly set CPU threading to prevent thrashing and stabilize speed
        torch.set_num_threads(int(cpu_workers))
        yield log(f"⚙️ CPU Workers allocated: {int(cpu_workers)}")
        
        yield log("🧬 Initializing Pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        pipeline.to(device)
        
        yield log("🎙️ Running VAD...")
        diarization = pipeline(audio_file)
        
        yield log(f"🧠 Loading Whisper ({whisper_model_size})...")
        model = whisper.load_model(whisper_model_size, device=device)
        
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
                
                # Safely find the next available filename index without dropping the clip
                final_filename = f"audio_{global_clip_index:06d}.wav"
                while final_filename in processed_files:
                    global_clip_index += 1
                    final_filename = f"audio_{global_clip_index:06d}.wav"
                
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
# METHOD 3: MERGE DATASETS
# ======================================================

def get_all_datasets():
    datasets_dir = os.path.join(core.path_base, "datasets")
    dataset_list = []
    if os.path.exists(datasets_dir):
        for lang in os.listdir(datasets_dir):
            lang_dir = os.path.join(datasets_dir, lang)
            if os.path.isdir(lang_dir):
                for ds in os.listdir(lang_dir):
                    ds_dir = os.path.join(lang_dir, ds)
                    if os.path.isdir(ds_dir):
                        # Use forward slashes for cross-platform compatibility in UI
                        dataset_list.append(f"{lang}/{ds}")
    return dataset_list

def merge_datasets_ui(merged_name, target_lang, resample_sr, selected_datasets):
    if not merged_name:
        yield "❌ Please enter a name for the merged dataset."
        return
    if not selected_datasets:
        yield "❌ Please select at least one dataset to merge."
        return
        
    output_dir = os.path.join(core.path_base, "datasets", target_lang, merged_name)
    wavs_dir = os.path.join(output_dir, "wavs")
    metadata_path = os.path.join(output_dir, "metadata.csv")
    
    os.makedirs(wavs_dir, exist_ok=True)
    
    target_sr_val = SR_MAP.get(resample_sr, None)
    
    merged_metadata_lines = []
    global_index = 0
    
    import shutil
    from pydub import AudioSegment
    
    total_datasets = len(selected_datasets)
    
    for i, ds_path in enumerate(selected_datasets):
        yield f"🔄 Merging dataset {i+1}/{total_datasets}: {ds_path} ..."
        
        # ds_path is "lang/ds_name"
        ds_full_path = os.path.join(core.path_base, "datasets", ds_path.replace("/", os.sep))
        ds_metadata = os.path.join(ds_full_path, "metadata.csv")
        ds_wavs = os.path.join(ds_full_path, "wavs")
        
        if not os.path.exists(ds_metadata):
            continue
            
        with open(ds_metadata, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 5:
                orig_lang = parts[0]
                orig_ds_name = parts[1]
                orig_filename = parts[2]
                speaker = parts[3]
                text = parts[4]
                
                orig_wav_path = os.path.join(ds_wavs, orig_filename)
                if os.path.exists(orig_wav_path):
                    new_filename = f"audio_{global_index:06d}.wav"
                    new_wav_path = os.path.join(wavs_dir, new_filename)
                    
                    if target_sr_val is not None:
                        clip = AudioSegment.from_file(orig_wav_path)
                        if clip.frame_rate != target_sr_val:
                            clip = clip.set_frame_rate(target_sr_val)
                        clip = clip.set_channels(1)
                        clip.export(new_wav_path, format="wav")
                    else:
                        shutil.copy2(orig_wav_path, new_wav_path)
                    
                    new_line = f"{target_lang}|{merged_name}|{new_filename}|{speaker}|{text}"
                    merged_metadata_lines.append(new_line)
                    global_index += 1

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_metadata_lines))
        
    yield f"🎉 Successfully merged {total_datasets} datasets into {merged_name}! Total clips: {global_index}"

# ======================================================
# METHOD 4: METADATA VALIDATION
# ======================================================

def format_transcript_html(text):
    if not text:
        return ""
    html = text
    green_chars = ["ā", "ē", "ī", "ō", "ū", "Ā", "Ē", "Ī", "Ō", "Ū"]
    blue_chars = ["â", "é", "î", "ô", "û", "Â", "É", "Î", "Ô", "Û"]
    
    for c in green_chars:
        html = html.replace(c, f'<span style="color: #66ff66; font-weight: bold;">{c}</span>')
    for c in blue_chars:
        html = html.replace(c, f'<span style="color: #66ccff; font-weight: bold;">{c}</span>')
        
    return f'<div style="font-size: 1.5em; line-height: 1.6; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px; margin-bottom: 10px;">{html}</div>'

def load_validation_dataset(dataset_name):
    if not dataset_name:
        return [], 0, "❌ No dataset selected.", "", None, ""
        
    ds_full_path = os.path.join(core.path_base, "datasets", dataset_name.replace("/", os.sep))
    metadata_path = os.path.join(ds_full_path, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        return [], 0, f"❌ metadata.csv not found in {dataset_name}", "", None, ""
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if not lines:
        return [], 0, "❌ metadata.csv is empty.", "", None, ""
        
    total_clips = len(lines)
    status_msg = f"✅ Loaded {total_clips} clips from {dataset_name}"
    
    # Load the first clip (index 0)
    audio_path, transcript, info_msg = get_clip_by_index(lines, 0, ds_full_path)
    
    return lines, 0, status_msg, info_msg, audio_path, transcript

def get_clip_by_index(lines, index, ds_full_path):
    if not lines or index < 0 or index >= len(lines):
        return None, "", "❌ Invalid index."
        
    parts = lines[index].split("|")
    if len(parts) >= 5:
        filename = parts[2]
        transcript = parts[4]
        audio_path = os.path.join(ds_full_path, "wavs", filename)
        if not os.path.exists(audio_path):
            return None, transcript, f"⚠️ Audio missing: {filename}"
        return audio_path, transcript, f"Clip {index + 1} of {len(lines)}"
    return None, "", "❌ Corrupt metadata line."

def change_clip_index(lines, current_index, action, ds_name, target_goto=None):
    if not lines:
        return current_index, "❌ No dataset loaded.", None, ""
        
    ds_full_path = os.path.join(core.path_base, "datasets", ds_name.replace("/", os.sep))
    
    new_index = current_index
    if action == "prev":
        new_index = max(0, current_index - 1)
    elif action == "next":
        new_index = min(len(lines) - 1, current_index + 1)
    elif action == "goto" and target_goto is not None:
        try:
            new_index = int(target_goto) - 1
            new_index = max(0, min(len(lines) - 1, new_index))
        except:
            pass
            
    audio_path, transcript, info_msg = get_clip_by_index(lines, new_index, ds_full_path)
    return new_index, info_msg, audio_path, transcript

def save_transcript_edit(lines, current_index, new_transcript, ds_name):
    if not lines or current_index < 0 or current_index >= len(lines):
        return lines, "❌ Cannot save: invalid state."
        
    parts = lines[current_index].split("|")
    if len(parts) >= 5:
        parts[4] = new_transcript.strip()
        lines[current_index] = "|".join(parts)
        
        ds_full_path = os.path.join(core.path_base, "datasets", ds_name.replace("/", os.sep))
        metadata_path = os.path.join(ds_full_path, "metadata.csv")
        
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return lines, f"💾 Saved clip {current_index + 1} successfully."
        except Exception as e:
            return lines, f"❌ Save error: {e}"
            
    return lines, "❌ Corrupt line, cannot save."

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
                hf_btn = gr.Button(_("COMMON_BTN_START"), variant="primary", elem_classes="wui-button-green")
                hf_stop = gr.Button(_("COMMON_BTN_STOP"), variant="stop")
                hf_resume = gr.Button(_("DATASET_BTN_RESUME"), elem_classes="wui-button-blue")
            
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
                wx_whisper_model = gr.Dropdown(
                    label="Whisper Model", 
                    choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"], 
                    value="large-v3"
                )
                
            with gr.Row():
                wx_sr = gr.Dropdown(label=_("DATASET_LABEL_RESAMPLE"), choices=sr_options, value="None")
                wx_save_every = gr.Number(label=_("DATASET_LABEL_SAVE_EVERY"), value=100, precision=0)
                wx_max_dur = gr.Number(label=_("DATASET_LABEL_MAX_DUR"), value=15)
                
            with gr.Row():
                wx_cpu_workers = gr.Slider(
                    label=_("COMMON_CPU_WORKERS"),  
                    minimum=1, 
                    maximum=os.cpu_count(), 
                    value=max(1, os.cpu_count() // 2), 
                    step=1
                )
                
            with gr.Row():
                wx_btn = gr.Button(_("COMMON_BTN_START"), variant="primary", elem_classes="wui-button-green")
                wx_stop = gr.Button(_("COMMON_BTN_STOP"), variant="stop")
                wx_resume = gr.Button(_("COMMON_BTN_RESUME"), elem_classes="wui-button-blue")
                
            with gr.Column():
                wx_status = gr.Textbox(label=_("DATASET_LABEL_STATUS"), lines=1, interactive=False)
                wx_logs = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=12, autoscroll=True)
            
            wx_event = wx_btn.click(
                process_long_audio_ui, 
                inputs=[audio_input, wx_out_name, gr.State(16), wx_sr, wx_lang, wx_save_every, wx_max_dur, wx_whisper_model, wx_cpu_workers], 
                outputs=[wx_status, wx_logs],
                show_progress="hidden"
            )
            wx_resume.click(
                fn=handle_resume, 
                inputs=[wx_logs], 
                outputs=[wx_status, wx_logs]
            ).then(
                fn=process_long_audio_ui, 
                inputs=[audio_input, wx_out_name, gr.State(16), wx_sr, wx_lang, wx_save_every, wx_max_dur, wx_whisper_model, wx_cpu_workers], 
                outputs=[wx_status, wx_logs],
                show_progress="hidden"
            )
            wx_stop.click(
                fn=handle_stop, 
                inputs=[wx_logs], 
                outputs=[wx_status, wx_logs], 
                cancels=[wx_event]
            )
            
        with gr.Tab(_("DATASET_TAB_MERGE")):
            with gr.Row():
                merge_out_name = gr.Textbox(label=_("DATASET_LABEL_MERGE_NAME"), placeholder="my_merged_dataset", scale=2)
                merge_lang = gr.Dropdown(label=_("COMMON_LABEL_LANG"), choices=lang_options, value="tr", scale=1)
                merge_sr = gr.Dropdown(label=_("DATASET_LABEL_RESAMPLE"), choices=sr_options, value="None", scale=1)
                refresh_btn = gr.Button(_("DATASET_BTN_REFRESH_MERGE"), scale=1)
                
            with gr.Row():
                dataset_checkboxes = gr.CheckboxGroup(
                    label=_("DATASET_LABEL_MERGE_SEL"),
                    choices=get_all_datasets(),
                    interactive=True
                )
                
            with gr.Row():
                merge_btn = gr.Button(_("DATASET_BTN_MERGE"), variant="primary", elem_classes="wui-button-green")
                
            with gr.Row():
                merge_status = gr.Textbox(label=_("DATASET_LABEL_STATUS"), lines=1, interactive=False)
                
            refresh_btn.click(
                fn=lambda: gr.CheckboxGroup(choices=get_all_datasets()),
                inputs=[],
                outputs=[dataset_checkboxes]
            )
            
            merge_btn.click(
                fn=merge_datasets_ui,
                inputs=[merge_out_name, merge_lang, merge_sr, dataset_checkboxes],
                outputs=[merge_status]
            )
            
        with gr.Tab(_("DATASET_TAB_VAL")):
            gr.Markdown(_("DATASET_DESC_VAL"), elem_classes="wui-markdown")
            
            with gr.Row():
                val_dataset_dd = gr.Dropdown(label=_("DATASET_LABEL_SEL_DS"), choices=get_all_datasets(), interactive=True, scale=3)
                val_refresh_btn = gr.Button(_("DATASET_BTN_REFRESH_DS"), scale=1)
                val_load_btn = gr.Button(_("DATASET_BTN_LOAD_DS"), variant="primary", scale=1)
                
            val_status_msg = gr.Textbox(label=_("DATASET_LABEL_STATUS"), interactive=False, lines=1)
            
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    val_clip_info = gr.Textbox(label=_("DATASET_LABEL_CURR_CLIP"), interactive=False, value="Clip 0 of 0")
                    
                    with gr.Row():
                        val_prev_btn = gr.Button(_("DATASET_BTN_PREV"))
                        val_next_btn = gr.Button(_("DATASET_BTN_NEXT"))
                        
                    with gr.Row():
                        val_goto_num = gr.Number(label=_("DATASET_LABEL_GOTO"), value=1, precision=0)
                    with gr.Row():
                        val_goto_btn = gr.Button(_("DATASET_BTN_GO"), variant="secondary")
                        
                with gr.Column(scale=2):
                    val_audio = gr.Audio(label=_("DATASET_LABEL_AUDIO_PREV"), type="filepath", interactive=False)
                    val_transcript_html = gr.HTML()
                    val_transcript = gr.Textbox(label=_("DATASET_LABEL_TRANSCRIPT"), lines=3)
                    val_save_btn = gr.Button(_("DATASET_BTN_SAVE_TRANSCRIPT"), variant="primary", elem_classes="wui-button-green")
                    
            # States
            val_metadata_state = gr.State([])
            val_index_state = gr.State(0)
            
            # Events
            val_refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=get_all_datasets()),
                inputs=[],
                outputs=[val_dataset_dd]
            )
            
            val_load_btn.click(
                fn=load_validation_dataset,
                inputs=[val_dataset_dd],
                outputs=[val_metadata_state, val_index_state, val_status_msg, val_clip_info, val_audio, val_transcript]
            )
            
            val_prev_btn.click(
                fn=lambda lines, idx, ds: change_clip_index(lines, idx, "prev", ds),
                inputs=[val_metadata_state, val_index_state, val_dataset_dd],
                outputs=[val_index_state, val_clip_info, val_audio, val_transcript]
            )
            
            val_next_btn.click(
                fn=lambda lines, idx, ds: change_clip_index(lines, idx, "next", ds),
                inputs=[val_metadata_state, val_index_state, val_dataset_dd],
                outputs=[val_index_state, val_clip_info, val_audio, val_transcript]
            )
            
            val_goto_btn.click(
                fn=lambda lines, idx, ds, goto: change_clip_index(lines, idx, "goto", ds, goto),
                inputs=[val_metadata_state, val_index_state, val_dataset_dd, val_goto_num],
                outputs=[val_index_state, val_clip_info, val_audio, val_transcript]
            )
            
            val_save_btn.click(
                fn=save_transcript_edit,
                inputs=[val_metadata_state, val_index_state, val_transcript, val_dataset_dd],
                outputs=[val_metadata_state, val_status_msg]
            )
            
            val_transcript.change(
                fn=format_transcript_html,
                inputs=[val_transcript],
                outputs=[val_transcript_html]
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