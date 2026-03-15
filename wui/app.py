import warnings
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"

# 1. Block pkg_resources warning (from ctranslate2/whisperx)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# 2. Block torchaudio backend warning (from speechbrain)
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

import sys
import gradio as gr

from core import core
from core.core import _, get_available_languages

import main
import models
import corpus
import config
import dataset
import tokenizer
import preprocessor
import trainer
import inference
import tts

def update_language(new_lang):
    """Updates the global language and saves it to config."""
    core.ui_language = new_lang
    # Save the current project and the new language to config.json
    core.save_wui(core.project_name, new_lang)
    # Notify the user that a restart is needed
    gr.Info("Language preference saved! Please restart the app to apply the new language.")
    
def get_header_text():
    """Helper to format the header string safely."""
    p_name = core.project_name if core.project_name else "None"
    return f"# ITTS-TR WEBUI / {_('PROJECT')} : {p_name}"
    
def refresh_project_state(current_selection):
    """Called when user manually selects a project from the dropdown."""
    projects = core.list_projects()

    if current_selection in projects:
        core.project_name = current_selection
        core.project_path = os.path.join(core.prj_path, current_selection)
        final_value = current_selection
        core.save_wui(final_value, core.wui_lang)
    else:
        # If selection is invalid, try to auto-select the first available
        if len(projects) > 0:
            final_value = projects[0]
            core.project_name = final_value
            core.save_wui(final_value, core.wui_lang)
        else:
            final_value = None
            core.project_name = None
            core.project_path = core.prj_path

    return (gr.Dropdown(choices=projects, value=final_value), get_header_text())
    
def refresh_project_folder(current_selection):
    """
    Called automatically by the Timer every 5 seconds.
    Checks disk for new/deleted folders, updates list, and AUTO-SELECTS if needed.
    """
    projects = core.list_projects()

    # Logic:
    # 1. If current selection is valid, keep it.
    # 2. If current selection is invalid/None but we have projects, auto-select the first one.
    # 3. Otherwise, set to None.
    
    if current_selection in projects:
        final_value = current_selection
        core.project_path = os.path.join(core.prj_path, current_selection)
    elif len(projects) > 0:
        final_value = projects[0]
        core.project_name = final_value
        core.project_path = os.path.join(core.prj_path, final_value)
        core.save_wui(final_value, core.wui_lang)
    else:
        final_value = None
        core.project_name = None
        core.project_path = core.prj_path

    # Sync Global State
    core.project_name = final_value

    # Return new Dropdown AND update the Header so it matches the auto-selection
    return (
        gr.Dropdown(choices=projects, value=final_value),
        get_header_text()
    )

with gr.Blocks() as root_demo:
    
    with gr.Row():
        with gr.Column(scale=3):
            header_md = gr.Markdown(value=get_header_text())
            
        with gr.Column(scale=2):
            existing_projects = core.list_projects()
            project_selector = gr.Dropdown(
                choices=existing_projects, 
                show_label=False,
                value=core.project_name, 
                interactive=True
            )
            
        with gr.Column(scale=1):
            language_selector = gr.Dropdown(
                choices=get_available_languages(), 
                value=core.ui_language,
                show_label=False,
                interactive=True
            )
    
    #################################################
    
    with gr.Tabs():
        
        with gr.Tab(_("HOME")):
            main.create_demo(project_selector, header_md)
            
        with gr.Tab(_("MODELS")):
            models.create_demo()
            
        with gr.Tab(_("CORPUS")):
            corpus.create_demo()
            
        with gr.Tab(_("CONFIG")):
            config.create_demo()
                    
        with gr.Tab(_("DATASET")):
            dataset.create_demo()
            
        with gr.Tab(_("TOKENIZER")):
            tokenizer.create_demo()
            
        with gr.Tab(_("PREPROCESSOR")):
            preprocessor.create_demo()
            
        with gr.Tab(_("TRAINER")):
            trainer.create_demo()
            
        with gr.Tab(_("INFERENCE")):
            inference.create_demo()
            
        with gr.Tab(_("TTS")):
            tts.create_demo()
            
    #################################################
     
    project_selector.change(
        fn=refresh_project_state, 
        inputs=project_selector, 
        outputs=[project_selector, header_md]
    )

    language_selector.change(
        fn=update_language,
        inputs=language_selector,
        outputs=None
    )    

    refresh_timer = gr.Timer(value=5.0)
    
    refresh_timer.tick(
        fn=refresh_project_folder, 
        inputs=project_selector, 
        outputs=[project_selector, header_md] 
    )  

if __name__ == "__main__":
    root_demo.launch(css=core.my_css)