import gradio as gr
import os
import shutil
import urllib.request
from huggingface_hub import snapshot_download
from core import core
from core.core import _

def list_files(local_dir_name):

    if local_dir_name in ["", "."]:
        target_dir = core.path_base
    else:
        target_dir = os.path.join(core.path_base, local_dir_name)
    
    if not os.path.exists(target_dir):
        return "⚠️ Directory does not exist yet."
    
    try:
        items = os.listdir(target_dir)
        if not items:
            return "📂 Directory is empty."
            
        formatted_list = []
        for item in sorted(items):
            full_path = os.path.join(target_dir, item)
            if os.path.isdir(full_path):
                formatted_list.append(f"📂 {item}")
            else:
                formatted_list.append(f"📄 {item}")
                
        return "\n".join(formatted_list)
        
    except Exception as e:
        return f"Error reading directory: {e}"

def download_repo(repo_id, local_dir_name):

    target_dir = os.path.join(core.path_base, local_dir_name)
    
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        target_files = ["bpe.model", "gpt.pth", "config.yaml"]
        copied_files = []
        
        for file_name in target_files:
            src_file = os.path.join(target_dir, file_name)
            if os.path.exists(src_file):
                new_file_name = f"en_{file_name}"
                dest_file = os.path.join(core.wui_ckpt, new_file_name)
                shutil.copy2(src_file, dest_file)
                copied_files.append(new_file_name)
        
        copy_msg = f"\n📦 Copied to wui_ckpt: {', '.join(copied_files)}" if copied_files else ""
        
        return f"✅ Successfully downloaded '{repo_id}' to:\n{target_dir}{copy_msg}", list_files(local_dir_name)
        
    except Exception as e:
        return f"❌ Download Failed:\n{str(e)}", list_files(local_dir_name)
        
def download_tr_weights():
    repo_id = "ruygar/itts_tr_lex"
    target_dir = core.wui_ckpt
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id, 
            allow_patterns=["tr_bpe.model", "tr_config.yaml", "tr_gpt.pth"], # Only pulls these 3
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        return f"✅ Successfully downloaded Turkish weights directly to:\n{target_dir}", list_files("wui/ckpt")
    except Exception as e:
        return f"❌ Download Failed:\n{str(e)}", list_files("wui/ckpt")

def download_to_global_cache(repo_id):

    try:
        path = snapshot_download(repo_id=repo_id, local_dir_use_symlinks=False)
        return f"✅ Successfully cached '{repo_id}' at:\n{path}"
    except Exception as e:
        return f"❌ Cache Download Failed: {str(e)}"

def download_url_to_root(url, filename):

    try:
        target_path = os.path.join(core.path_base, "core", filename)
        
        # Download with standard urllib
        urllib.request.urlretrieve(url, target_path)
        
        return f"✅ Successfully downloaded '{filename}' to:\n{target_path}", list_files(".")
    except Exception as e:
        return f"❌ Download Error: {str(e)}", list_files(".")
        
def download_whisper_model(model_id):
    
    import whisper
    
    whisper_dir = os.path.join(core.path_root, 'models', 'whisper')
    os.makedirs(whisper_dir, exist_ok=True)
      
    try:
        whisper.load_model(
            model_id, 
            device="cpu",
        )
        return f"✅ Whisper model '{model_id}' successfully downloaded to:\n{whisper_dir}"
    except Exception as e:
        return f"❌ Whisper Download Error: {str(e)}"

def create_demo():
    
    with gr.Blocks() as demo:
        
        gr.Markdown(_("MODELS_HEADER"))
        gr.Markdown(_("MODELS_DESC"))
        
        # --- SECTION 1: Project Checkpoints (Local Dir) ---
        with gr.Group():
            gr.Markdown(_("MODELS_SECTION_LOCAL"))
            with gr.Row():
                with gr.Column(scale=1):
                    repo_input = gr.Textbox(label=_("MODELS_LABEL_REPO_ID"), value="IndexTeam/IndexTTS-2")
                    dir_input = gr.Textbox(label=_("MODELS_LABEL_TARGET_DIR"), value="indextts/checkpoints", interactive=False)
                    download_btn = gr.Button(_("MODELS_BTN_DOWNLOAD_LOCAL"), variant="primary")
                    status_output = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=4, interactive=False)
                
                with gr.Column(scale=1):
                    files_display = gr.TextArea(
                        label=_("MODELS_CURRENT_FILES"), 
                        value=list_files("indextts/checkpoints"), 
                        lines=13, 
                        interactive=False
                    )
                    refresh_btn = gr.Button(_("COMMON_BTN_REFRESH"))

        # --- SECTION 2: Global Cache Models ---
        gr.HTML("<div style='height:20px'></div>")
        with gr.Group():
            gr.Markdown(_("MODELS_SECTION_GLOBAL"))
            cache_status_output = gr.Textbox(label=_("MODELS_CACHE_STATUS"), lines=2, interactive=False)

            with gr.Row():
                m1_btn = gr.Button(_("MODELS_BTN_W2V_BERT"), variant="secondary")
                m2_btn = gr.Button(_("MODELS_BTN_MASKGCT"), variant="secondary")
                m3_btn = gr.Button(_("MODELS_BTN_CAMPPLUS"), variant="secondary")
                m4_btn = gr.Button(_("MODELS_BTN_BIGVGAN"), variant="secondary")
        
        # --- SECTION 3: Whisper Models ---        
        gr.HTML("<div style='height:20px'></div>")
        with gr.Group():
            gr.Markdown(_("MODELS_SECTION_WHISPER"))
           
            whisper_status = gr.Textbox(label=_("MODELS_WHISPER_STATUS"), lines=2, interactive=False)
            
            with gr.Row():
                whisper_model_select = gr.Dropdown(
                    label=_("MODELS_LABEL_SELECT_MODEL"), 
                    choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"], 
                    value="large-v3",
                    allow_custom_value=True,
                    scale=3
                )
            with gr.Row():
                whisper_dl_btn = gr.Button(_("MODELS_BTN_DOWNLOAD_WHISPER"), variant="primary", scale=1)

        # --- SECTION 4: Dependency Fixes (New) ---
        gr.HTML("<div style='height:20px'></div>")
        with gr.Group():
            gr.Markdown(_("MODELS_SECTION_DEPENDENCY"))
           
            with gr.Row():
                pb2_url = gr.Textbox(
                    label=_("MODELS_LABEL_FILE_URL"), 
                    value="https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/sentencepiece_model_pb2.py"
                )
                pb2_name = gr.Textbox(label=_("MODELS_LABEL_TARGET_FILENAME"), value="sentencepiece_model_pb2.py", interactive=False)
            with gr.Row():    
                pb2_btn = gr.Button(_("MODELS_BTN_DOWNLOAD_PB2"), variant="stop")
            
            pb2_status = gr.Textbox(label=_("MODELS_DEPENDENCY_STATUS"), lines=1)
            
        # --- SECTION 5: Download Turkish Weights ---
        gr.HTML("<div style='height:20px'></div>")
        with gr.Group():
            gr.Markdown(_("MODELS_SECTION_TR_WEIGHTS"))
            gr.Markdown(_("MODELS_DESC_TR_WEIGHTS"))
           
            tr_status_output = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=2, interactive=False)
            
            with gr.Row():
                tr_dl_btn = gr.Button(_("MODELS_BTN_DOWNLOAD_TR"), variant="primary")

        # --- Event Handlers ---
        
        # Local Download
        download_btn.click(
            fn=download_repo, 
            inputs=[repo_input, dir_input], 
            outputs=[status_output, files_display]
        )
        refresh_btn.click(
            fn=list_files, 
            inputs=[dir_input], 
            outputs=[files_display]
        )

        # Global Cache Downloads
        m1_btn.click(fn=download_to_global_cache, inputs=[gr.State("facebook/w2v-bert-2.0")], outputs=[cache_status_output])
        m2_btn.click(fn=download_to_global_cache, inputs=[gr.State("amphion/MaskGCT")], outputs=[cache_status_output])
        m3_btn.click(fn=download_to_global_cache, inputs=[gr.State("funasr/campplus")], outputs=[cache_status_output])
        m4_btn.click(fn=download_to_global_cache, inputs=[gr.State("nvidia/bigvgan_v2_22khz_80band_256x")], outputs=[cache_status_output])
        
        # Whisper Download Handler
        whisper_dl_btn.click(
            fn=download_whisper_model, 
            inputs=[whisper_model_select], 
            outputs=[whisper_status]
        )
        
        # Dependency Fix
        pb2_btn.click(
            fn=download_url_to_root,
            inputs=[pb2_url, pb2_name],
            outputs=[pb2_status, files_display] # Updates file list to show if it appeared
        )
        
        # Turkish Weights Download Handler
        tr_dl_btn.click(
            fn=download_tr_weights,
            inputs=[],
            outputs=[tr_status_output, files_display] # Updates the log and UI list
        )

        # =============
        # DOCUMENTATION
        # =============
        gr.HTML("<div style='height:20px'></div>")
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("models"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")
    
    return demo