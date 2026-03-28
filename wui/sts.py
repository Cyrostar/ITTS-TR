import os
import gradio as gr
from core import core
from core.core import _
from rvc.core import run_infer_script

CKPT_DIR = os.path.join(core.path_base, "ckpt", "rvc")

def get_rvc_models():
    """Scans the checkpoints folder for all .pth files."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    return [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]

def run_rvc_inference_ui(
    input_path, pth_name, f0_method, pitch, 
    volume_envelope, protect, split_audio, f0_autotune, 
    f0_autotune_strength, export_format, embedder_model
):
    """Wrapper function to map Gradio inputs to the RVC backend script."""
    try:
        if not input_path or not pth_name:
            return _("RVC_ERR_MISSING_PATHS"), None

        # Resolve the full path for the selected model
        pth_path = os.path.join(CKPT_DIR, pth_name).replace("\\", "/")

        # Sanitize Windows paths for the uploaded temp file
        input_path = input_path.strip().strip('"').strip("'").replace("\\", "/")
        
        # --- FIXED OUTPUT PATH HANDLING ---
        # Route to outputs/rvc/ and prefix with 'converted_'
        output_dir = os.path.join(os.getcwd(), "outputs", "rvc").replace("\\", "/")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"converted_{base_name}.{export_format.lower()}").replace("\\", "/")
        # ----------------------------------------

        # Execute the RVC backend logic
        message, out_file = run_infer_script(
            pitch=int(pitch),
            index_rate=0.0,             # Hardcoded since index fields are removed
            volume_envelope=float(volume_envelope),
            protect=float(protect),
            f0_method=f0_method,
            input_path=input_path,
            output_path=output_path,
            pth_path=pth_path,
            index_path="",              # Hardcoded to empty string
            split_audio=split_audio,
            f0_autotune=f0_autotune,
            f0_autotune_strength=float(f0_autotune_strength),
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            clean_audio=False,          
            clean_strength=0.0,         
            export_format=export_format,
            embedder_model=embedder_model,
            embedder_model_custom=None,
            sid=0
        )
        
        if not os.path.exists(out_file):
            return f"{_('RVC_MSG_ERROR')} Internal backend failure. No file was saved.", None

        return f"{_('RVC_MSG_SUCCESS')} {message}", out_file
        
    except Exception as e:
        return f"{_('RVC_MSG_ERROR')} {str(e)}", None

# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    """Generates the Gradio layout for the RVC Inference tab."""
    with gr.Blocks() as demo:
        gr.Markdown(_("RVC_HEADER"))
        gr.Markdown(_("RVC_DESC"))

        # ==========================================
        # TOP SECTION: I/O PATHS
        # ==========================================
        with gr.Group():
            gr.Markdown(_("RVC_HEADER_IO"))
            
            with gr.Row():
                # Replaced Textbox with Audio File Upload
                input_audio = gr.Audio(
                    label=_("RVC_LABEL_INPUT"), 
                    type="filepath", 
                    scale=1
                )
            
            with gr.Row():
                pth_dropdown = gr.Dropdown(
                    label=_("RVC_LABEL_PTH"), 
                    choices=get_rvc_models(),
                    interactive=True,
                    scale=1
                )
                
            with gr.Row():
                refresh_models_btn = gr.Button("🔄", variant="secondary", scale=1)

        # ==========================================
        # CORE INFERENCE SETTINGS
        # ==========================================
        with gr.Group():
            gr.Markdown(_("RVC_HEADER_CORE"))
            with gr.Row():
                f0_method = gr.Dropdown(
                    label=_("RVC_LABEL_F0"), 
                    choices=["crepe", "rmvpe", "fcpe", "hybrid[crepe+rmvpe]"], 
                    value="rmvpe"
                )
                embedder_model = gr.Dropdown(
                    label=_("RVC_LABEL_EMBEDDER"),
                    choices=["contentvec", "chinese-hubert-base", "japanese-hubert-base"],
                    value="contentvec"
                )
                export_format = gr.Dropdown(
                    label=_("RVC_LABEL_EXPORT"),
                    choices=["WAV", "MP3", "FLAC", "OGG"],
                    value="WAV"
                )

            with gr.Row():
                pitch = gr.Slider(label=_("RVC_LABEL_PITCH"), minimum=-24, maximum=24, step=1, value=0)
                volume_envelope = gr.Slider(label=_("RVC_LABEL_VOL_ENV"), minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                protect = gr.Slider(label=_("RVC_LABEL_PROTECT"), minimum=0.0, maximum=0.5, step=0.01, value=0.33)

        # ==========================================
        # ACCORDION: ADVANCED AUDIO PROCESSING
        # ==========================================
        with gr.Accordion(_("RVC_ACC_ADVANCED"), open=False):
            with gr.Row():
                split_audio = gr.Checkbox(label=_("RVC_CHK_SPLIT"), value=False)
                f0_autotune = gr.Checkbox(label=_("RVC_CHK_AUTOTUNE"), value=False)
                f0_autotune_strength = gr.Slider(label=_("RVC_LABEL_AUTOTUNE_STR"), minimum=0.0, maximum=1.0, step=0.1, value=1.0)

        # ==========================================
        # ACTION BUTTONS & OUTPUT
        # ==========================================
        with gr.Row():
            infer_btn = gr.Button(_("RVC_BTN_INFER"), variant="primary", scale=3)
         
        with gr.Row():
            output_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=2, interactive=False, scale=2)
            output_audio = gr.Audio(label=_("RVC_LABEL_RESULT"), type="filepath", scale=2)

        # ==========================================
        # EVENT MAPPING
        # ==========================================
        
        # 1. Refresh the models dropdown list when the refresh button is clicked
        refresh_models_btn.click(
            fn=lambda: gr.Dropdown(choices=get_rvc_models()),
            inputs=None,
            outputs=pth_dropdown
        )

        # 2. Map UI elements to backend inference function
        inputs_list = [
            input_audio, pth_dropdown, f0_method, pitch, 
            volume_envelope, protect, split_audio, f0_autotune, 
            f0_autotune_strength, export_format, embedder_model
        ]

        infer_btn.click(
            fn=run_rvc_inference_ui,
            inputs=inputs_list,
            outputs=[output_log, output_audio]
        )
              
    return demo