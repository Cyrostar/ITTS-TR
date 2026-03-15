import gradio as gr
import os
import sys
import shutil
import platform
import time

from core import core
from core.core import _

# --- Imports ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False
# --------------------

def get_cpu_temp():
    """Safely tries to get CPU temp. Often requires Admin on Windows."""
    if not PSUTIL_AVAILABLE or not hasattr(psutil, "sensors_temperatures"):
        return "N/A"
    try:
        temps = psutil.sensors_temperatures()
        # Common sensor names for various platforms
        for name in ['coretemp', 'k10temp', 'cpu_thermal', 'package_id_0']:
            if name in temps:
                return f"{temps[name][0].current} °C"
        return "N/A"
    except:
        return "N/A"

def get_live_metrics():
    """Fast metrics that update every 2 seconds."""
    # 1. System Metrics (CPU, RAM, Swap)
    if not PSUTIL_AVAILABLE:
        cpu, cpu_temp, ram, swap_text = "N/A", "N/A", "N/A", "N/A"
    else:
        cpu = f"{psutil.cpu_percent()}%"
        cpu_temp = get_cpu_temp()
        mem = psutil.virtual_memory()
        ram = f"{round(mem.used / (1024**3), 1)} / {round(mem.total / (1024**3), 1)} GB ({mem.percent}%)"
        swap = psutil.swap_memory()
        swap_text = f"{round(swap.used / (1024**3), 1)} GB ({swap.percent}%)"

    # 2. GPU Metrics (VRAM, Load, and Temperature)
    gpu_vram = "N/A"
    gpu_load = "N/A"
    gpu_temp = "N/A"

    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # VRAM Usage
            free, total = torch.cuda.mem_get_info(0)
            used_gb = (total - free) / (1024**3)
            total_gb = total / (1024**3)
            vram_percent = (used_gb / total_gb) * 100
            gpu_vram = f"{round(used_gb, 1)} / {round(total_gb, 1)} GB ({round(vram_percent, 1)}%)"
            
            # GPU Load & Temp (requires pynvml)
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Load
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_load = f"{util.gpu}%"
                
                # Temperature
                t_raw = pynvml.nvmlDeviceGetTemperature(handle, 0) # 0 = NVML_TEMPERATURE_GPU
                gpu_temp = f"{t_raw} °C"
        except:
            gpu_vram, gpu_load, gpu_temp = "Error", "Error", "Error"

    # 3. Project Status
    if core.project_name:
        p_path = os.path.join(core.prj_path, core.project_name) if core.prj_path else "Unknown"
        p_info = f"✅ Active Project: {core.project_name}\n📂 Path: {p_path}\n(Click Refresh for file stats)"
    else:
        p_info = "⚠️ No active project selected."
    
    return cpu, cpu_temp, ram, swap_text, gpu_vram, gpu_load, gpu_temp, p_info

def get_static_and_project_info():
    """Slow checks (GPU name, Disk, Project Size) for Manual Refresh."""
    python_ver = sys.version.split()[0]
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        gpu_status = f"✅ {gpu_name} ({vram} GB)"
    else:
        gpu_status = "⚠️ CPU Mode (No GPU Detected)"

    if core.project_name:
        proj_path = os.path.join(core.prj_path, core.project_name)
    else:
        proj_path = "None"
    
    total, used, free = shutil.disk_usage(".")
    disk_free = f"{round(free / (1024**3), 2)} GB Free"

    if core.project_name and os.path.exists(proj_path):
        file_count = 0
        total_size = 0
        try:
            for dp, _, fn in os.walk(proj_path):
                for f in fn:
                    file_count += 1
                    total_size += os.path.getsize(os.path.join(dp, f))
            
            size_mb = round(total_size / (1024**2), 2)
            proj_status = (f"✅ Active Project: {core.project_name}\n"
                           f"📂 Location: {proj_path}\n"
                           f"📄 Files: {file_count}\n"
                           f"📦 Size: {size_mb} MB")
        except Exception as e:
            proj_status = f"✅ Active: {core.project_name}\n❌ Error scanning files: {e}"
    else:
        proj_status = "⚠️ No active project selected."

    return python_ver, gpu_status, disk_free, proj_status

def rename_project_fn(new_name):
    if not new_name or not new_name.strip():
        return gr.update(), gr.update(), "❌ Error: Name cannot be empty."
    
    if not core.project_name:
         return gr.update(), gr.update(), "❌ Error: No project selected to rename."

    clean_name = new_name.strip()
    old_path = core.project_path
    new_path = os.path.join(core.prj_path, clean_name)
    
    if os.path.exists(new_path):
        return gr.update(), gr.update(), f"❌ Error: Folder '{clean_name}' already exists."
    
    try:
        os.rename(old_path, new_path)
        core.project_name = clean_name
        core.project_path = new_path
        
        core.save_wui(clean_name, core.wui_lang)
        
        new_project_list = core.list_projects()
        dd_update = gr.Dropdown(choices=new_project_list, value=clean_name)
        
        hd_update = f"# ITTS-TR WEBUI / {_('project')} : {clean_name}"
        
        return dd_update, hd_update, f"✅ Renamed to: {clean_name}"
    except Exception as e:
        return gr.update(), gr.update(), f"❌ System Error: {str(e)}"

def create_project_fn(new_name):
    if not new_name or not new_name.strip():
        return gr.update(), gr.update(), "❌ Error: Name cannot be empty."
    
    clean_name = new_name.strip()
    new_path = os.path.join(core.prj_path, clean_name)
    
    if os.path.exists(new_path):
        return gr.update(), gr.update(), f"❌ Error: Project '{clean_name}' already exists."
        
    try:
        os.makedirs(new_path)
        core.project_name = clean_name
        core.project_path = new_path
        
        core.save_wui(clean_name, core.wui_lang)
        
        new_project_list = core.list_projects()
        dd_update = gr.Dropdown(choices=new_project_list, value=clean_name)
        
        hd_update = f"# ITTS-TR WEBUI / {_('project')} : {clean_name}"
        
        return dd_update, hd_update, f"✅ Created: {clean_name}"
    except Exception as e:
        return gr.update(), gr.update(), f"❌ System Error: {str(e)}"
        
def delete_project_fn(confirm_name):
    if not core.project_name:
        return gr.update(), gr.update(), "❌ Error: No active project."
    
    if confirm_name != core.project_name:
        return gr.update(), gr.update(), f"❌ Error: Name '{confirm_name}' does not match active project."

    target_name = core.project_name
    try:
        if core.delete_project(target_name):
            remaining = core.list_projects()
            if not remaining:
                os.makedirs(os.path.join(core.prj_path, "myproject"), exist_ok=True)
                remaining = ["myproject"]
            
            new_active = remaining[0]
            core.project_name = new_active
            core.project_path = os.path.join(core.prj_path, new_active)
            
            core.save_wui(new_active, core.wui_lang)
            
            dd_update = gr.Dropdown(choices=remaining, value=new_active)
            
            hd_update = f"# ITTS-TR WEBUI / {_('project')} : {new_active}"
            return dd_update, hd_update, f"🗑️ Deleted project: {target_name}"
        else:
            return gr.update(), gr.update(), "❌ Error: Project folder not found."
    except Exception as e:
        return gr.update(), gr.update(), f"❌ System Error: {str(e)}"
        
def run_torch_compile_test():
    
    import io
    from contextlib import redirect_stdout, redirect_stderr
    import time

    output = io.StringIO()
    
    # Capture both standard output and standard errors into the StringIO buffer
    with redirect_stdout(output), redirect_stderr(output):
        print("--- 🔬 CLEAN TORCH.COMPILE DIAGNOSTIC ---")
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("❌ Execution Failed: PyTorch with CUDA is not available on this system.")
            return output.getvalue()
            
        try:           
            @torch.compile(backend="inductor")
            def test_kernel(x): 
                return x * 2 + 1
            
            print("🚀 Launching Torch Inductor kernel...")
            input_data = torch.randn(10, device="cuda")
            start = time.time()
            result = test_kernel(input_data)
            torch.cuda.synchronize() 
            
            print(f"✅ SUCCESS! Kernel executed in {time.time() - start:.4f}s")
            print(f"Result Preview: {result[0].item():.4f}")

        except Exception as e:
            import traceback
            print(f"\n❌ Execution Failed:\n{traceback.format_exc()}")

    return output.getvalue()
    
# ======================================================
# UI CREATION
# ======================================================

def create_demo(project_selector=None, header_md=None):
    with gr.Blocks() as demo:
        # --- SECTION 0: PROJECT MANAGEMENT ---
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                gr.Markdown(_("HOME_CREATE_PROJECT"))
                with gr.Row():
                    create_name_input = gr.Textbox(placeholder=_("HOME_PLACEHOLDER_NEW_PROJECT"), show_label=False, container=False, scale=3)
                    create_btn = gr.Button(_("HOME_BTN_CREATE"), variant="primary", scale=1)
                create_log = gr.Textbox(label=_("HOME_STATUS"), lines=1, interactive=False)

            with gr.Column(scale=1):
                gr.Markdown(_("HOME_RENAME_CURRENT"))
                with gr.Row():
                    rename_name_input = gr.Textbox(placeholder=_("HOME_PLACEHOLDER_RENAME"), show_label=False, container=False, scale=3)
                    rename_btn = gr.Button(_("HOME_BTN_RENAME"), variant="secondary", scale=1)
                rename_log = gr.Textbox(label=_("HOME_STATUS"), lines=1, interactive=False)

        # --- SECTION 1: LIVE MONITOR ---
        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown(_("HOME_PROCESSOR"))
                with gr.Row():
                    cpu_box = gr.Textbox(label=_("HOME_CPU_LOAD"))
                    gpu_load_box = gr.Textbox(label=_("HOME_GPU_LOAD"))
                with gr.Row():
                    cpu_temp_box = gr.Textbox(label=_("HOME_CPU_TEMP"))
                    gpu_temp_box = gr.Textbox(label=_("HOME_GPU_TEMP"))
            
            with gr.Column():
                gr.Markdown(_("HOME_MEMORY"))
                with gr.Row():
                    ram_box = gr.Textbox(label=_("HOME_RAM_USAGE"))
                    vram_box = gr.Textbox(label=_("HOME_VRAM_USAGE"))
                swap_box = gr.Textbox(label=_("HOME_SWAP_USED"))

        # --- SECTION 2: STATIC DETAILS ---
        with gr.Row():
            with gr.Column():
                gr.Markdown(_("HOME_HARDWARE_ENV"))
                py_box = gr.Textbox(label=_("HOME_PYTHON"))
                gpu_box = gr.Textbox(label=_("HOME_GPU_ACCEL"))
                disk_box = gr.Textbox(label=_("HOME_DISK_SPACE"))
            
            with gr.Column():
                gr.Markdown(_("HOME_ACTIVE_PROJ_DATA"))
                proj_box = gr.Textbox(label=_("HOME_PROJ_DETAILS"), lines=5)
                refresh_btn = gr.Button(_("HOME_BTN_REFRESH"), variant="secondary")
                
                with gr.Accordion(_("HOME_DANGER_ZONE"), open=False):
                    gr.Markdown(_("HOME_DELETE_CONFIRM_TEXT"))
                    delete_confirm_input = gr.Textbox(placeholder=_("HOME_PLACEHOLDER_DELETE"), show_label=False)
                    delete_btn = gr.Button(_("HOME_BTN_DELETE"), variant="stop")
                    delete_log = gr.Textbox(label=_("HOME_DELETION_STATUS"), lines=1, interactive=False)

        # --- LOGIC ---
        if project_selector is not None and header_md is not None:
            create_btn.click(create_project_fn, inputs=[create_name_input], outputs=[project_selector, header_md, create_log]).success(get_static_and_project_info, outputs=[py_box, gpu_box, disk_box, proj_box])
            rename_btn.click(rename_project_fn, inputs=[rename_name_input], outputs=[project_selector, header_md, rename_log]).success(get_static_and_project_info, outputs=[py_box, gpu_box, disk_box, proj_box])
            delete_btn.click(
                delete_project_fn, 
                inputs=[delete_confirm_input], 
                outputs=[project_selector, header_md, delete_log]
            ).success(
                get_static_and_project_info, 
                outputs=[py_box, gpu_box, disk_box, proj_box]
            )
        
        else:
            fb = lambda x: (gr.update(), gr.update(), "⚠️ Run via app.py to enable")
            create_btn.click(fb, inputs=[create_name_input], outputs=[gr.Textbox(visible=False), gr.Textbox(visible=False), create_log])
            rename_btn.click(fb, inputs=[rename_name_input], outputs=[gr.Textbox(visible=False), gr.Textbox(visible=False), rename_log])
            delete_btn.click(lambda x: (gr.update(), gr.update(), "⚠️ Run via app.py to enable"), outputs=[gr.State(), gr.State(), delete_log])
        timer = gr.Timer(value=2.0)
        timer.tick(get_live_metrics, outputs=[cpu_box, cpu_temp_box, ram_box, swap_box, vram_box, gpu_load_box, gpu_temp_box, proj_box])
        
        refresh_btn.click(get_static_and_project_info, outputs=[py_box, gpu_box, disk_box, proj_box])
        demo.load(get_static_and_project_info, outputs=[py_box, gpu_box, disk_box, proj_box])
        
        with gr.Accordion(_("HOME_DIAGNOSTICS_ACCORDION"), open=False):
            gr.Markdown(_("HOME_DIAGNOSTICS_DESC"))
            
            diag_btn = gr.Button(_("HOME_DIAGNOSTICS_BTN"), variant="secondary")
            diag_output = gr.Textbox(
                label=_("HOME_DIAGNOSTICS_LOGS"), 
                lines=12, 
                max_lines=20, 
                interactive=False,
                elem_id="diag_logs"
            )
            
            diag_btn.click(
                fn=run_torch_compile_test,
                outputs=[diag_output]
            )

    return demo