import gradio as gr
import os
import sys
import shutil
import subprocess
import re
import string
import datetime
from pypdf import PdfReader

import gc
import torch
import torchaudio
import whisper
import soundfile as sf
from pyannote.audio import Pipeline
from demucs.pretrained import get_model
from demucs.apply import apply_model

from core import core
from core.core import _
from core.normalizer import MultilingualNormalizer, MultilingualWordifier

# --- HELPER FUNCTIONS ---
    
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages.append(txt)
    return "\n".join(pages)

def get_pdf_list():
    """Returns a list of full paths to PDF files for gr.Files component."""
    d = os.path.join(core.corpus_directory(), "pdf")
    if not os.path.exists(d):
        return []
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".pdf")]
    files.sort()
    return files

def get_txt_list():
    """Returns a list of full paths to TXT files for gr.Files component."""
    d = os.path.join(core.corpus_directory(), "txt")
    if not os.path.exists(d):
        return []
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".txt")]
    files.sort()
    return files

def refresh_lists():
    """Manual refresh handler returning formatted strings."""
    return list_files_formatted("pdf", ".pdf"), list_files_formatted("txt", ".txt")
    
def list_files_formatted(subfolder, extension):
    """Lists files in a specific corpus subfolder with icons, matching models.py format."""
    target_dir = os.path.join(core.corpus_directory(), subfolder)
    
    if not os.path.exists(target_dir):
        return "📂 Directory not created yet."
    
    try:
        items = [f for f in os.listdir(target_dir) if f.lower().endswith(extension)]
        if not items:
            return f"📂 No {extension.upper()} files found."
            
        formatted_list = [f"📄 {item}" for item in sorted(items)]
        return "\n".join(formatted_list)
    except Exception as e:
        return f"Error: {str(e)}"

# --- MERGE LOGIC ---

def merge_mix_files_ui(progress=gr.Progress()):
    """
    Reads all files from corpus/mix and combines them into corpus/corpus.txt
    """
    logs = []
    def log(msg):
        logs.append(msg)
        return "\n".join(logs)
    
    corpus_dir = core.corpus_directory()
    mix_dir = os.path.join(corpus_dir, "mix")
    output_path = os.path.join(corpus_dir, "corpus.txt")
    
    if not os.path.exists(mix_dir):
        return log("❌ 'Mix' folder does not exist. Process some files first.")
        
    # Get all .txt files in mix
    files = [f for f in os.listdir(mix_dir) if f.lower().endswith(".txt")]
    files.sort()
    
    if not files:
        return log("⚠️ No processed text files found in 'corpus/mix'.")
        
    log(f"🔄 Found {len(files)} files. Merging...")
    
    all_content = []
    
    # Read and accumulate
    for filename in files:
        p = os.path.join(mix_dir, filename)
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_content.append(content)
        except Exception as e:
            log(f"⚠️ Skipped {filename}: {str(e)}")
            
    if not all_content:
        return log("❌ No valid content extracted to merge.")
        
    # Join with newlines
    final_text = "\n".join(all_content)
    
    # Write to corpus root
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        log("✅ SUCCESS: All mix files combined.")
        log(f"📂 Created: {output_path}")
        log(f"📊 Total Characters: {len(final_text)}")
        log(f"📄 Total Lines: {final_text.count(chr(10))}")
        
    except Exception as e:
        return log(f"❌ Error writing corpus file: {str(e)}")
        
    return "\n".join(logs)

# --- CORE PROCESSING LOGIC ---

def add_file_to_corpus_ui(file_objs, corpus_name, is_unique, lang, progress=gr.Progress()):
    """
    Handles LIST of uploaded files.
    """
    logs = []
    
    # Helper to yield consistent output
    def update_step(msg):
        logs.append(msg)
        return "\n".join(logs)
    
    def fail_return(msg):
        return update_step(msg), gr.update(), gr.update()

    if not file_objs:
        return fail_return("❌ No files uploaded.")
    
    # Ensure it's a list (Gradio might pass single object if file_count was singular, but with multiple it sends list)
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    # Directories
    corpus_dir = core.corpus_directory()
    pdf_dir = os.path.join(corpus_dir, "pdf")
    txt_dir = os.path.join(corpus_dir, "txt")
    mix_dir = os.path.join(corpus_dir, "mix")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(mix_dir, exist_ok=True)

    total_files = len(file_objs)
    yield update_step(f"🚀 Starting batch process for {total_files} file(s)..."), gr.update(), gr.update()

    # Iterate over files
    for idx, file_obj in enumerate(file_objs):
        
        # Determine name for this specific file
        original_filename = os.path.basename(file_obj.name) # .name is safe in recent gradio versions for temp paths
        base_name = os.path.splitext(original_filename)[0]
        
        # LOGIC: Use 'corpus_name' ONLY if it's a single file upload. 
        # If multiple, ignore input and use original filename to prevent overwrite.
        if total_files == 1 and corpus_name:
            final_name = corpus_name
        else:
            final_name = base_name

        file_ext = os.path.splitext(original_filename)[1].lower()
        
        progress(idx / total_files, desc=f"Processing {original_filename}")
        logs.append(f"\n--- 📄 File {idx+1}/{total_files}: {original_filename} ---")
        
        try:
            raw_text = ""
            
            if file_ext == ".pdf":
                dest_path = os.path.join(pdf_dir, f"{final_name}.pdf")
                if os.path.exists(dest_path): os.remove(dest_path)
                shutil.move(file_obj.name, dest_path)
                
                logs.append(f"   💾 Saved PDF to corpus/pdf/")
                raw_text = extract_text_from_pdf(dest_path)

            elif file_ext == ".txt":
                dest_path = os.path.join(txt_dir, f"{final_name}.txt")
                if os.path.exists(dest_path): os.remove(dest_path)
                shutil.move(file_obj.name, dest_path)
                
                logs.append(f"   💾 Saved TXT to corpus/txt/")
                with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            else:
                logs.append(f"   ⚠️ Skipped: Unsupported format {file_ext}")
                continue

            # Process Words
            if not raw_text.strip():
                logs.append("   ⚠️ Empty text content.")
                continue
                
            normalizer = MultilingualNormalizer(lang=lang, wordify=True, abbreviations=True)
            normalized_text = normalizer.normalize(raw_text)
                    
            sentences = [s.strip() for s in normalized_text.split('.') if s.strip()]
            
            if is_unique:
                sentences = list(dict.fromkeys([s.strip() for s in sentences if s.strip()]))

            # Save Mix
            mix_output = os.path.join(mix_dir, f"{final_name}.txt")
            with open(mix_output, "w", encoding="utf-8") as f:
                for sentence in sentences:
                    f.write(f"{sentence}.\n")
            
            logs.append(f"   ✅ Processed: {len(sentences)} lines -> corpus/mix/")

        except Exception as e:
            logs.append(f"   ❌ Error: {str(e)}")
            
        # Yield update for log stream
        yield "\n".join(logs), gr.update(), gr.update()

    # Final Yield
    logs.append("\n✨ BATCH COMPLETE ✨")
    yield "\n".join(logs), list_files_formatted("pdf", ".pdf"), list_files_formatted("txt", ".txt")

def open_video_folder():
    folder_path = os.path.join(core.wui_outs, "video")

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."
    
def run_ytdlp(url):

    exe_path = os.getenv("ARTHA_YT_DIP_DIR")
    
    if not exe_path:
        return "❌ Error: 'ARTHA_YT-DIP_DIR' environment variable is not set."
        
    exe_file = os.path.join(exe_path, "yt-dlp_x86.exe")
    
    if not os.path.exists(exe_file):
        return f"❌ Error: Executable not found at {exe_file}"
    
    out_path = os.path.join(core.wui_outs, "video")
    os.makedirs(out_path, exist_ok=True)
            
    out_template = os.path.join(out_path, "%(title)s.%(ext)s")
    
    try:
        meta_cmd = [
            exe_file,
            "--print", "title",
            url
        ]
        meta_result = subprocess.run(meta_cmd, capture_output=True, text=True, check=True)
     
        title = meta_result.stdout.strip()
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        
        out_file = os.path.join(out_path, f"{safe_title}.mp3")
    
        command = [
            exe_file,
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            url,
            "-o", out_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        log = result.stdout.strip()
        return title, out_file

    except subprocess.CalledProcessError as e:
        return f"❌ Executable failed with error code {e.returncode}.\n\nError Output:\n{e.stderr}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"
        
def clean_audio_with_demucs_api(audio_input, progress=gr.Progress()):
    if not audio_input:
        return "❌ Error: No audio file provided."
        
    # Extract path from Gradio file object
    audio_path = audio_input.name if hasattr(audio_input, "name") else str(audio_input)
    
    # Setup output directory
    output_dir = os.path.join(core.wui_outs, "cleaner")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    final_vocal_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        progress(0.1, desc="Loading Demucs model to VRAM...")
        # 1. Load the model ('htdemucs' is the standard high-quality one)
        model = get_model('htdemucs')
        model.to(device)
        model.eval()
        
        progress(0.3, desc="Loading and formatting audio...")
        # 2. Load audio using torchaudio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to model's expected sample rate (44100 Hz)
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            
        # Demucs expects stereo (2 channels). 
        # If mono, duplicate it. If surround (5.1), mix it down.
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
            
        # Add a batch dimension: (channels, length) -> (1, channels, length)
        wav = wav.unsqueeze(0).to(device)
        
        progress(0.5, desc="Separating stems (this takes time)...")
        # 3. Apply the model
        # split=True chunks long files so you don't run out of VRAM!
        with torch.no_grad():
            sources = apply_model(model, wav, shifts=1, split=True, overlap=0.25)
            
        progress(0.8, desc="Extracting vocal track...")
        # sources shape is (batch, sources, channels, length)
        # Find exactly which index holds the 'vocals'
        vocal_idx = model.sources.index('vocals')
        
        # Extract just the vocals and pull it back to System RAM (CPU)
        vocals_tensor = sources[0, vocal_idx].cpu()
        
        # 4. Save the file directly
        torchaudio.save(final_vocal_path, vocals_tensor, model.samplerate)
        
        progress(0.9, desc="Flushing GPU memory...")
        # 5. CRITICAL: Clean up VRAM so Whisper has room to run next!
        del model
        del wav
        del sources
        del vocals_tensor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        progress(1.0, desc="Done!")
        return f"✅ Success!\nCleaned audio saved to:\n{final_vocal_path}"
        
    except Exception as e:
        return f"❌ Demucs Error: {str(e)}"

def transcribe_audio_ui(audio_input, model_size, use_normalizer, single_paragraph, lang, progress=gr.Progress()):
    if not audio_input:
        return "❌ Error: No audio file provided."
    
    # Extract the file path from the Gradio file object
    audio_path = audio_input.name if hasattr(audio_input, "name") else str(audio_input)
    
    try:
        # Explicitly check for CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        progress(0.1, desc=f"Loading Whisper model on {device.upper()}...")
        model = whisper.load_model(model_size, device=device)
        
        progress(0.3, desc="Transcribing (this may take a while)...")
        # Pass the dynamic language argument to Whisper
        result = model.transcribe(audio_path, language=lang)
        
        progress(0.8, desc="Formatting text...")
        processed_lines = []
        normalizer = MultilingualNormalizer(lang=lang, wordify=True) if use_normalizer else None
        
        # Whisper segments naturally correspond to sentences
        for segment in result["segments"]:
            text = segment["text"].strip()
            if not text:
                continue
                
            if use_normalizer and normalizer:
                processed_text = normalizer.normalize(text)
            else:
                wordifier = MultilingualWordifier(text, language_code=lang)
                processed_text = getattr(wordifier.processor, 'normalized_text', 
                                 getattr(wordifier.processor, 'get_text', lambda: text)())
            
            processed_lines.append(processed_text)

        progress(1.0, desc="Done!")
        
        # Toggle between Single Paragraph vs Line-by-Line
        if single_paragraph:
            return " ".join(processed_lines)
        else:
            return "\n".join(processed_lines)

    except Exception as e:
        return f"❌ Transcription Error: {str(e)}"
        
def diarization_audio_ui(input_file, trim_silence, gap_seconds, min_spks, max_spks):
    if input_file is None: return None
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipeline.to(device)
    
    # Load audio to GPU
    waveform, fs = torchaudio.load(input_file)
    waveform = waveform.to(device)
    audio_data = {"waveform": waveform, "sample_rate": fs}
    
    print(f"--- 🚀 Multi-Speaker Engine Started ---")
    
    # Run Diarization with user-defined speaker limits
    diarization = pipeline(audio_data, min_speakers=min_spks, max_speakers=max_spks)
    
    found_speakers = sorted(diarization.labels())
    print(f"--- 🎤 Found {len(found_speakers)} Speakers: {found_speakers} ---")
    
    generated_files = []
    target_fs = 44100
    resampler = torchaudio.transforms.Resample(fs, target_fs).to(device)

    # Pre-calculate gap buffer
    silence_buffer = None
    if trim_silence and gap_seconds > 0:
        num_gap_samples = int(gap_seconds * fs)
        silence_buffer = torch.zeros((waveform.shape[0], num_gap_samples), device=device)

    # LOOP THROUGH EVERY DETECTED SPEAKER
    for spk_id in found_speakers:
        segments_to_merge = []
        has_audio = False
        
        # --- PROCESSING LOGIC ---
        if not trim_silence:
            output_waveform = torch.zeros_like(waveform)
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == spk_id:
                    start_s, end_s = int(segment.start * fs), int(segment.end * fs)
                    output_waveform[:, start_s:end_s] = waveform[:, start_s:end_s]
                    has_audio = True
            final_tensor = output_waveform
        else:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == spk_id:
                    start_s, end_s = int(segment.start * fs), int(segment.end * fs)
                    segments_to_merge.append(waveform[:, start_s:end_s])
                    if silence_buffer is not None:
                        segments_to_merge.append(silence_buffer)
            if segments_to_merge:
                final_tensor = torch.cat(segments_to_merge, dim=-1)
                has_audio = True

        # --- SAVE EACH SPEAKER ---
        if has_audio:
            audio_resampled = resampler(final_tensor)
            audio_out = audio_resampled.t().cpu().numpy()
            
            # Precise naming for high-speed export
            timestamp = datetime.datetime.now().strftime("%H%M%S_%f")
            out_path = os.path.join(core.wui_outs, "diarization")
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, f"{spk_id}_{timestamp}.wav")
            
            sf.write(save_path, audio_out, target_fs, subtype='PCM_16')
            generated_files.append(save_path)
            print(f"✅ Exported {spk_id}")
    
    first_audio_preview = generated_files[0] if generated_files else None
    # Return the full list of files to the gr.File component
    return first_audio_preview, generated_files

def get_genre_list():
    """Returns a comprehensive list of document genres for the naming tool."""
    genres = [
        "Academic", "Anthropology", "Archaeology", "Architecture", "Art", 
        "Astrology", "Biography", "Biology", "Business", "Chemistry", 
        "Cinema", "Culinary", "Drama", "Economy", "Education", 
        "Engineering", "Essay", "Fantasy", "Fashion", "Finance", 
        "Health", "History", "Journalism", "Law", "Literature", 
        "Medicine", "Memoir", "Metaphysic", "Music", "Mythology", "Novel", 
        "Philosophy", "Physics", "Poetry", "Politics", "Psychology", 
        "Religion", "Science", "ScienceFiction", "Sociology", "Spiritual",
        "Sports", "Technology", "Theology", "Travel", "Other"
    ]
    genres.sort() # Ensure they are alphabetical
    return genres
    
# Turkish character mapping for standardized naming
TR_NAME_MAP = {
    ord('ç'): 'c', ord('Ç'): 'C',
    ord('ğ'): 'g', ord('Ğ'): 'G',
    ord('ı'): 'i', ord('I'): 'I',
    ord('İ'): 'I', 
    ord('ö'): 'o', ord('Ö'): 'O',
    ord('ş'): 's', ord('Ş'): 'S',
    ord('ü'): 'u', ord('Ü'): 'U'
}

def _clean_for_naming(text):
    """Internal helper to apply standard naming syntax."""
    if not text: return ""
    # 1. Convert Turkish characters
    t = text.translate(TR_NAME_MAP)
    # 2. Replace hyphens with spaces (prevents "Jean-Paul" becoming "JeanPaul")
    t = t.replace("-", " ")
    # 3. Remove all other punctuation (except underscores)
    chars_to_remove = string.punctuation.replace("_", "") 
    t = t.translate(str.maketrans('', '', chars_to_remove))
    return t
    
def generate_standardized_name(genre, author, title):
    if not genre or not author or not title:
        return "Please fill all fields (Genre, Author, and Title)."

    # Process Author: Clean -> Title Case -> Underscores
    c_author = _clean_for_naming(author)
    c_author = " ".join([word.capitalize() for word in c_author.split()]).replace(" ", "_")

    # Process Title: Clean -> Sentence Case -> Underscores
    c_title = _clean_for_naming(title)
    c_title = " ".join([word.capitalize() for word in c_title.split()]).replace(" ", "_")

    return f"{genre}-{c_author}-{c_title}"
    
def generate_audiobook_name(source, narrator, genre, author, title):

    if not all([source, narrator, genre, author, title]):
        return "Please fill all fields (Source, Narrator, Genre, Author, and Title)."

    # Process Source & Narrator: Clean -> Title Case -> Underscores
    c_source = _clean_for_naming(source)
    c_source = " ".join([word.capitalize() for word in c_source.split()]).replace(" ", "_")
    
    c_narrator = _clean_for_naming(narrator)
    c_narrator = " ".join([word.capitalize() for word in c_narrator.split()]).replace(" ", "_")

    # Process Author & Title (Same as Document Namer)
    c_author = _clean_for_naming(author)
    c_author = " ".join([word.capitalize() for word in c_author.split()]).replace(" ", "_")
    
    c_title = _clean_for_naming(title)
    c_title = " ".join([word.capitalize() for word in c_title.split()]).replace(" ", "_")

    return f"Audiobook-{c_source}-{c_narrator}-{genre}-{c_author}-{c_title}"
    
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    lang_options = core.language_list()
    
    with gr.Blocks() as demo:
        gr.Markdown(_("CORPUS_HEADER"))
        gr.Markdown(_("CORPUS_DESC"))        

        with gr.Group():
            gr.Markdown(_("CORPUS_HEADER_ADD"))
            gr.Markdown(_("CORPUS_DESC_ADD"))
        
            with gr.Row():
                file_input = gr.File(
                    label=_("COMMON_LABEL_UPLOAD"),
                    file_types=[".pdf", ".txt"],
                    file_count="multiple" 
                )
        
            with gr.Row():
                corpus_name = gr.Textbox(
                    label=_("CORPUS_LABEL_NAME"),
                    placeholder=_("CORPUS_PLACEHOLDER_NAME"),
                    scale=4
                )
                corpus_lang = gr.Dropdown(
                    label=_("COMMON_LABEL_LANG"),
                    choices=lang_options, 
                    value="tr",
                    scale=1
                )
                is_unique = gr.Checkbox(
                    label=_("CORPUS_CHK_UNIQUE"),
                    value=False,
                    scale=1
                )
        
            with gr.Row():
                add_btn = gr.Button(_("CORPUS_BTN_PROCESS"), variant="primary")
        
            corpus_log = gr.Textbox(
                label=_("COMMON_LABEL_LOGS"),
                lines=6,
                max_lines=12
            )
    
            # --- SECTION: FILE REPOSITORIES ---  
            with gr.Accordion(_("CORPUS_ACC_REPO"), open=False, elem_classes="wui-accordion"):            
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(_("CORPUS_HEADER_PDF"))
                        pdf_files = gr.Textbox(
                            label=_("CORPUS_LABEL_PDF"), 
                            value=lambda: list_files_formatted("pdf", ".pdf"), 
                            lines=10,
                            max_lines=10,
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown(_("CORPUS_HEADER_TXT"))
                        txt_files = gr.Textbox(
                            label=_("CORPUS_LABEL_TXT"), 
                            value=lambda: list_files_formatted("txt", ".txt"), 
                            lines=10,
                            max_lines=10,
                            interactive=False
                        )
                        
                gr.HTML("<br>")
                with gr.Row():
                    refresh_btn = gr.Button(_("COMMON_BTN_REFRESH"), variant="secondary")
                
            # --- SECTION: MERGE BUTTON ---
            gr.HTML("<br>")
            with gr.Row():
                merge_btn = gr.Button(_("CORPUS_BTN_MERGE"), variant="primary")
                
        gr.HTML("<div style='height:10px'></div>")
        
        # ==========
        # UTILITIES
        # ==========
        with gr.Group():
            gr.Markdown(_("CORPUS_HEADER_UTILS"), elem_classes="wui-markdown")
            
        # --- SECTION: YOUTUBE DOWNLOADER ---
        with gr.Accordion(_("CORPUS_ACC_YT"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_YT"))   
            with gr.Row():
                yt_url = gr.Textbox(label=_("CORPUS_LABEL_URL"))
            
            yt_run_btn = gr.Button(_("CORPUS_BTN_FETCH"), variant="primary")
            
            yt_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=1, max_lines=6, interactive=False)
            
            yt_aud = gr.Audio(label=_("CORPUS_LABEL_PREVIEW"), type="filepath")
            
            yt_folder_btn = gr.Button(_("CORPUS_BTN_OPEN_DIR"))
            
        # --- SECTION: AUDIO CLEANER ---
        with gr.Accordion(_("CORPUS_ACC_CLEANER"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_CLEANER"))   
            with gr.Row():
                clean_audio_input = gr.File(label=_("CORPUS_LABEL_UPLOAD_AUDIO"), file_types=[".wav", ".mp3"])
            
            clean_btn = gr.Button(_("CORPUS_BTN_ISOLATE"), variant="primary")
            
            clean_output = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=5)
           
        # --- SECTION: AUDIO TRANSCRIPTOR ---
        with gr.Accordion(_("CORPUS_ACC_WHISPER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_WHISPER"))
            with gr.Row():
                transcribe_audio_input = gr.File(
                    label=_("CORPUS_LABEL_WHISPER_AUDIO"), 
                    file_types=[".wav", ".mp3"]
                )
                with gr.Column():
                    transcribe_model_size = gr.Dropdown(
                        label=_("CORPUS_LABEL_WHISPER_MODEL"), 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        value="large-v3",
                        info=_("CORPUS_INFO_WHISPER_MODEL")
                    )
                    transcribe_lang = gr.Dropdown(
                        label=_("COMMON_LABEL_LANG"), 
                        choices=lang_options, 
                        value="tr",
                        info=_("CORPUS_INFO_WHISPER_LANG")
                    )
                with gr.Column():
                    with gr.Row():
                        transcribe_use_normalizer = gr.Checkbox(
                            label=_("CORPUS_CHK_NORMALIZER"), 
                            value=False,
                            info=_("CORPUS_INFO_NORMALIZER")
                        )
                    with gr.Row():
                        transcribe_single_paragraph = gr.Checkbox(
                            label=_("CORPUS_CHK_PARAGRAPH"), 
                            value=True,
                        )
            
            transcribe_btn = gr.Button(_("CORPUS_BTN_TRANSCRIBE"), variant="primary")
            transcribe_output = gr.Textbox(
                label=_("CORPUS_LABEL_TRANSCRIBE_OUT"), 
                lines=8, 
                placeholder=_("CORPUS_PLACEHOLDER_TRANSCRIBE"),
                interactive=True,
                buttons=["copy"]
            )

        # --- SECTION: DIARIZATION ---
        with gr.Accordion(_("CORPUS_ACC_DIARIZATION"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_DIARIZATION"))           
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(type="filepath", label=_("CORPUS_LABEL_DIA_AUDIO"))
                    
                    with gr.Group():
                        gr.Markdown(_("CORPUS_HEADER_DIA_SETTINGS"))
                        trim_toggle = gr.Checkbox(label=_("CORPUS_CHK_TRIM"), value=False)
                        gap_input = gr.Number(label=_("CORPUS_LABEL_GAP"), value=0.0, step=0.1)
                        
                    with gr.Row():
                        min_s = gr.Slider(1, 10, value=1, step=1, label=_("CORPUS_LABEL_MIN_SPK"))
                        max_s = gr.Slider(1, 20, value=10, step=1, label=_("CORPUS_LABEL_MAX_SPK"))
                        
                    diarize_btn = gr.Button(_("CORPUS_BTN_DIA_START"), variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown(_("CORPUS_HEADER_DIA_FILES"))
                    first_speaker_audio = gr.Audio(label=_("CORPUS_LABEL_DIA_PREVIEW"), type="filepath", interactive=False)
                    file_output = gr.File(label=_("CORPUS_LABEL_DIA_DOWNLOAD"), file_count="multiple")

        diarize_btn.click(
            fn=diarization_audio_ui, 
            inputs=[audio_input, trim_toggle, gap_input, min_s, max_s], 
            outputs=[first_speaker_audio, file_output]
        )
    
        # --- SECTION: DOCUMENT NAMER ---
        with gr.Accordion(_("CORPUS_ACC_NAMER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_NAMER"))
            
            with gr.Row():
                namer_genre = gr.Dropdown(
                    label=_("CORPUS_LABEL_GENRE"),
                    choices=get_genre_list(), 
                    value="Novel",
                    filterable=True
                )
                namer_author = gr.Textbox(
                    label=_("CORPUS_LABEL_AUTHOR"), 
                    placeholder=_("CORPUS_PLACEHOLDER_AUTHOR")
                )
                namer_title = gr.Textbox(
                    label=_("CORPUS_LABEL_TITLE"), 
                    placeholder=_("CORPUS_PLACEHOLDER_TITLE")
                )
            
            with gr.Row():
                namer_btn = gr.Button(_("CORPUS_BTN_GEN_NAME"), variant="secondary")
            
            namer_output = gr.Textbox(
                label=_("CORPUS_LABEL_RESULT"),
                interactive=True
            )
        
        # --- SECTION: AUDIOBOOK NAMER ---
        with gr.Accordion(_("CORPUS_ACC_AB_NAMER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_AB_NAMER"))
            
            with gr.Row():
                ab_source = gr.Textbox(label=_("CORPUS_LABEL_SOURCE"), placeholder=_("CORPUS_PLACEHOLDER_SOURCE"))
                ab_narrator = gr.Textbox(label=_("CORPUS_LABEL_NARRATOR"), placeholder=_("CORPUS_PLACEHOLDER_NARRATOR"))
            
            with gr.Row():
                ab_genre = gr.Dropdown(
                    label=_("CORPUS_LABEL_GENRE"),
                    choices=get_genre_list(),
                    value="Novel",
                    filterable=True
                )
                ab_author = gr.Textbox(label=_("CORPUS_LABEL_AUTHOR"), placeholder=_("CORPUS_PLACEHOLDER_AB_AUTHOR"))
                ab_title = gr.Textbox(label=_("CORPUS_LABEL_TITLE"), placeholder=_("CORPUS_PLACEHOLDER_AB_TITLE"))
            
            with gr.Row():
                ab_btn = gr.Button(_("CORPUS_BTN_GEN_AB"), variant="secondary")
            
            ab_output = gr.Textbox(label=_("CORPUS_LABEL_RESULT"), interactive=True)
             
        # ACTIONS       
        add_btn.click(
            fn=add_file_to_corpus_ui,
            inputs=[file_input, corpus_name, is_unique, corpus_lang],
            outputs=[corpus_log, pdf_files, txt_files] 
        )
        
        refresh_btn.click(
            fn=refresh_lists,
            inputs=[],
            outputs=[pdf_files, txt_files]
        )
        
        merge_btn.click(
            fn=merge_mix_files_ui,
            inputs=[],
            outputs=[corpus_log] 
        )
        
        yt_run_btn.click(
            fn=run_ytdlp,
            inputs=[yt_url],
            outputs=[yt_log, yt_aud]
        )
        
        yt_folder_btn.click(
            fn=open_video_folder,
            outputs=None
        )
        
        clean_btn.click(
            fn=clean_audio_with_demucs_api,
            inputs=[clean_audio_input],
            outputs=[clean_output]
        )
        
        transcribe_btn.click(
            fn=transcribe_audio_ui,
            inputs=[transcribe_audio_input, transcribe_model_size, transcribe_use_normalizer, transcribe_single_paragraph, transcribe_lang],
            outputs=[transcribe_output]
        )
        
        namer_btn.click(
            fn=generate_standardized_name,
            inputs=[namer_genre, namer_author, namer_title],
            outputs=[namer_output]
        )
        
        ab_btn.click(
            fn=generate_audiobook_name,
            inputs=[ab_source, ab_narrator, ab_genre, ab_author, ab_title],
            outputs=[ab_output]
        )
        
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("corpus"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")
        
    return demo