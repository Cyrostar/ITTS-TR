import os
import gradio as gr
from core import core
from core.core import _
from core.voice import rvc_preprocessor, rvc_extractor, rvc_trainer, rvc_indexer
from pydub import AudioSegment
from pydub.silence import split_on_silence

RVC_WORKSPACE = os.path.join(core.path_base, "logs")

def slice_audio_for_rvc(audio_file, model_name, min_silence_len, silence_thresh, max_length_sec):
    """
    Slices a long audio file into smaller chunks based on natural silences.
    Saves the chunks into an isolated RVC workspace directory.
    """
    logs = []
    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    if not audio_file:
        return log("❌ Error: No audio file uploaded.")
    if not model_name or model_name.strip() == "":
        return log("❌ Error: Please provide a Model Name.")

    try:
        # 1. Setup the isolated directory structure
        model_name = model_name.strip().replace(" ", "_")
        dataset_dir = os.path.join(RVC_WORKSPACE, model_name, "dataset").replace("\\", "/")
        os.makedirs(dataset_dir, exist_ok=True)
        
        yield log(f"📂 Workspace created at: {dataset_dir}")
        yield log("⏳ Loading audio file... (This may take a moment for large files)")

        # 2. Load the audio file
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1) # Force mono for RVC
        
        yield log(f"✂️ Slicing audio based on silences (Threshold: {silence_thresh}dB, Min Silence: {min_silence_len}ms)...")
        
        # 3. Split on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=int(min_silence_len),
            silence_thresh=int(silence_thresh),
            keep_silence=200 # Keep 200ms of silence padding at the edges so it sounds natural
        )
        
        if not chunks:
            # Fallback if no silences were detected
            yield log("⚠️ No silences detected! Splitting strictly by max length...")
            chunk_length_ms = int(max_length_sec * 1000)
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        yield log(f"🔍 Found {len(chunks)} potential chunks. Filtering and exporting...")

        # 4. Export the chunks
        saved_count = 0
        max_ms = int(max_length_sec * 1000)
        
        for i, chunk in enumerate(chunks):
            # If a chunk is way too long, split it forcefully to prevent GPU OOM crashes
            if len(chunk) > max_ms:
                sub_chunks = [chunk[j:j+max_ms] for j in range(0, len(chunk), max_ms)]
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    # Ignore tiny fragments less than 2 seconds
                    if len(sub_chunk) > 2000: 
                        export_path = os.path.join(dataset_dir, f"audio_{saved_count:05d}.wav").replace("\\", "/")
                        sub_chunk.export(export_path, format="wav")
                        saved_count += 1
            else:
                if len(chunk) > 2000: # Ignore tiny fragments less than 2 seconds
                    export_path = os.path.join(dataset_dir, f"audio_{saved_count:05d}.wav").replace("\\", "/")
                    chunk.export(export_path, format="wav")
                    saved_count += 1
                    
            if i % 10 == 0 and i > 0:
                yield log(f"💾 Processed {i}/{len(chunks)} chunks...")

        yield log(f"✅ DONE! Successfully saved {saved_count} audio clips into the workspace.\n\nNext Step: Go to the 'Feature Extraction' tab.")

    except Exception as e:
        yield log(f"❌ Critical Error: {str(e)}")
        
def get_workspace_models():
    """Scans the workspace folder to populate the Model Name dropdown."""
    os.makedirs(RVC_WORKSPACE, exist_ok=True)
    return [d for d in os.listdir(RVC_WORKSPACE) if os.path.isdir(os.path.join(RVC_WORKSPACE, d))]
    
def preprocess_dataset_ui(model_name, sample_rate, cpu_cores):
    """Wrapper for Phase 2a: Audio Resampling and Formatting"""
    try:
        if not model_name:
            return "❌ Error: Please select a model from the workspace."
            
        dataset_path = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name, "dataset"))
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            return f"❌ Error: No audio found in {dataset_path}. Please slice audio in Tab 1 first."

        # Define the output directories for both target SR and 16k audio inside the model's workspace
        output_dir = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name, "sliced_audios"))
        output_dir_16k = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name, "sliced_audios_16k"))

        # ITTS-TR Native Implementation
        preprocessor = rvc_preprocessor(target_sr=int(sample_rate))
        msg = preprocessor.process_dataset(dataset_path, output_dir, output_dir_16k, int(cpu_cores))
        
        return f"✅ Preprocessing Complete:\n{msg}\n\nNext Step: Click '2. Extract Pitch & Features'"
    except Exception as e:
        return f"❌ Preprocessing Error: {str(e)}"

def extract_features_ui(model_name, f0_method, cpu_cores, sample_rate):
    """Wrapper for Phase 2b: Pitch (F0) and Semantic (HuBERT) Extraction"""
    try:
        if not model_name:
            return "❌ Error: Please select a model from the workspace."
            
        dataset_path = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name))

        # ITTS-TR Native Implementation
        extractor = rvc_extractor()
        msg = extractor.extract_features(
            exp_dir=dataset_path, 
            f0_method=f0_method, 
            embedder_model="contentvec",
            sample_rate=int(sample_rate)
        )
        return f"✅ Extraction Complete:\n{msg}\n\nNext Step: Go to the 'Train Model' tab."
    except Exception as e:
        return f"❌ Extraction Error: {str(e)}"
        
def train_model_ui(model_name, sample_rate, total_epoch, batch_size, save_every, cache_gpu, progress=gr.Progress()):
    """Wrapper for Phase 3: RVC Deep Learning Loop"""
    try:
        if not model_name:
            return "❌ Error: Please select a model from the workspace."

        dataset_path = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name))
        
        progress(0, desc="Initializing PyTorch Training Loop...")
        
        # ITTS-TR Native Implementation
        trainer = rvc_trainer()
        msg = trainer.run_training(
            exp_dir=dataset_path,
            sample_rate=int(sample_rate),
            total_epochs=int(total_epoch),
            batch_size=int(batch_size),
            save_every=int(save_every),
            cache_gpu=cache_gpu,
            f0_method="rmvpe"
        )
        return f"✅ Training Loop Finished:\n{msg}"
    except Exception as e:
        return f"❌ Training Error: {str(e)}"

def train_index_ui(model_name, progress=gr.Progress()):
    """Wrapper for Phase 4: FAISS Index Generation"""
    try:
        if not model_name:
            return "❌ Error: Please select a model."
            
        dataset_path = os.path.abspath(os.path.join(RVC_WORKSPACE, model_name))
        
        progress(0, desc="Clustering Semantic Embeddings (FAISS)...")
        
        # ITTS-TR Native Implementation
        indexer = rvc_indexer()
        msg = indexer.generate_index(exp_dir=dataset_path)
        
        return f"✅ Index Generation Complete:\n{msg}"
    except Exception as e:
        return f"❌ Index Error: {str(e)}"

# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    """Generates the Gradio layout for the RVC Training Dashboard."""
    
    with gr.Blocks() as demo:
        gr.Markdown("## 🎙️ RVC Voice Training Dashboard")
        gr.Markdown("Completely isolated training environment for Retrieval-based Voice Conversion.")
        
        with gr.Tabs():
            # ---------------------------------------------------------
            # TAB 1: DATASET SLICER
            # ---------------------------------------------------------
            with gr.TabItem("1. Workspace & Slicer"):
                gr.Markdown("### Phase 1: Prepare your Audio Dataset\nUpload a long audio file of your target voice. This tool will automatically detect silences, slice the audio into safe chunks, and create a workspace for your model.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(label="Raw Audio File", type="filepath")
                    
                    with gr.Column(scale=2):
                        model_name = gr.Textbox(
                            label="Model Name", 
                            placeholder="e.g., Arnica_V2", 
                            info="No spaces or special characters. This names your workspace folder."
                        )
                        max_clip_len = gr.Slider(
                            minimum=5, maximum=20, value=15, step=1, 
                            label="Max Clip Length (Seconds)", 
                            info="Prevents 'Out of Memory' crashes by strictly limiting max audio length."
                        )
                        
                with gr.Accordion("Advanced Slicing Parameters", open=False):
                    with gr.Row():
                        silence_thresh = gr.Slider(
                            minimum=-60, maximum=-10, value=-40, step=1, 
                            label="Silence Threshold (dB)", 
                            info="How quiet must it be to count as silence? Lower = stricter."
                        )
                        min_silence_len = gr.Slider(
                            minimum=200, maximum=2000, value=500, step=100, 
                            label="Minimum Silence Length (ms)", 
                            info="How long must the silence last to trigger a slice?"
                        )

                with gr.Row():
                    slice_btn = gr.Button("✂️ Create Workspace & Slice Audio", variant="primary")
                    
                slice_log = gr.Textbox(label="Slicing Progress", lines=8, interactive=False, autoscroll=True)

                # Event Mapping
                slice_btn.click(
                    fn=slice_audio_for_rvc,
                    inputs=[audio_input, model_name, min_silence_len, silence_thresh, max_clip_len],
                    outputs=[slice_log]
                )
                
            # ---------------------------------------------------------
            # TAB 2: FEATURE EXTRACTION
            # ---------------------------------------------------------
            with gr.TabItem("2. Feature Extraction"):
                gr.Markdown("### Phase 2: Prepare Audio for GPU\nThis phase resamples your dataset, extracts the pitch (F0), and extracts the phonetic features using HuBERT.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        extract_model_name = gr.Dropdown(
                            label="Target Model Workspace", 
                            choices=get_workspace_models(), 
                            interactive=True,
                            info="Select the model you sliced in Tab 1."
                        )
                    with gr.Column(scale=1):
                        refresh_workspace_btn = gr.Button("🔄 Refresh List", variant="secondary")

                with gr.Row():
                    sample_rate = gr.Dropdown(
                        label="Target Sample Rate", 
                        choices=["32000", "40000", "48000"], 
                        value="40000",
                        info="40000 is the recommended standard for RVC v2."
                    )
                    f0_method = gr.Dropdown(
                        label="Pitch Extraction Algorithm", 
                        choices=["rmvpe", "crepe", "dio", "harvest"], 
                        value="rmvpe",
                        info="RMVPE is highly recommended for vocal accuracy."
                    )
                    cpu_cores = gr.Slider(
                        minimum=1, 
                        maximum=os.cpu_count() or 4, 
                        value=max(1, (os.cpu_count() or 4) // 2), 
                        step=1, 
                        label="CPU Threads to Use",
                        info="More threads = faster extraction, but higher CPU usage."
                    )

                with gr.Row():
                    preprocess_btn = gr.Button("1. Preprocess Dataset", variant="primary")
                    extract_btn = gr.Button("2. Extract Pitch & Features", variant="primary")
                    
                extract_log = gr.Textbox(label="Extraction Progress", lines=6, interactive=False)

                # Event Mapping
                refresh_workspace_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_workspace_models()),
                    inputs=None,
                    outputs=[extract_model_name]
                )
                
                preprocess_btn.click(
                    fn=preprocess_dataset_ui,
                    inputs=[extract_model_name, sample_rate, cpu_cores],
                    outputs=[extract_log]
                )

                extract_btn.click(
                    fn=extract_features_ui,
                    inputs=[extract_model_name, f0_method, cpu_cores, sample_rate], # Added sample_rate here
                    outputs=[extract_log]
                )
                
            # ---------------------------------------------------------
            # TAB 3: MODEL TRAINING
            # ---------------------------------------------------------
            with gr.TabItem("3. Train Model"):
                gr.Markdown("### Phase 3 & 4: Deep Learning Loop\nTrain the neural network to replicate your target voice, and compile the FAISS index to preserve pronunciation.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        train_model_name = gr.Dropdown(
                            label="Target Model Workspace", 
                            choices=get_workspace_models(), 
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        refresh_train_btn = gr.Button("🔄 Refresh List", variant="secondary")

                with gr.Row():
                    train_sample_rate = gr.Dropdown(
                        label="Sample Rate", 
                        choices=["32000", "40000", "48000"], 
                        value="40000",
                        interactive=False,
                        info="Must match the rate you preprocessed with."
                    )
                    
                with gr.Row(variant="panel"):
                    with gr.Column():
                        total_epochs = gr.Slider(
                            minimum=10, maximum=1000, value=200, step=10, 
                            label="Total Epochs", 
                            info="How many times the AI reviews the dataset. (200-300 recommended)"
                        )
                        save_every = gr.Slider(
                            minimum=5, maximum=100, value=25, step=5, 
                            label="Save Frequency (Epochs)", 
                            info="Creates a backup checkpoint every X epochs."
                        )
                    with gr.Column():
                        batch_size = gr.Slider(
                            minimum=1, maximum=32, value=4, step=1, 
                            label="Batch Size", 
                            info="Higher = faster training, but uses more GPU VRAM. (Lower this if you get Out of Memory errors)"
                        )
                        cache_gpu = gr.Checkbox(
                            label="Cache Dataset to GPU", 
                            value=False,
                            info="Speeds up training massively, but requires a GPU with 12GB+ VRAM."
                        )

                with gr.Row():
                    train_btn = gr.Button("🧠 1. Start Neural Network Training", variant="primary")
                    index_btn = gr.Button("📚 2. Generate FAISS Index", variant="primary")
                    
                train_log = gr.Textbox(label="Training Status", lines=6, interactive=False)

                # Event Mapping
                refresh_train_btn.click(
                    fn=lambda: gr.Dropdown(choices=get_workspace_models()),
                    inputs=None,
                    outputs=[train_model_name]
                )
                
                train_btn.click(
                    fn=train_model_ui,
                    inputs=[train_model_name, train_sample_rate, total_epochs, batch_size, save_every, cache_gpu],
                    outputs=[train_log]
                )
                
                index_btn.click(
                    fn=train_index_ui,
                    inputs=[train_model_name],
                    outputs=[train_log]
                )

    return demo