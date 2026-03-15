<h1 align="center">ITTS-TR</h1> 
<div align="center"> 
  <a href="README-TR.md"><img src="img/flags/tr.svg" alt="TR" width="24"/></a> | 
  <a href="README.md"><img src="img/flags/gb.svg" alt="GB" width="24"/></a> 
</div>

---

A comprehensive Gradio-based Web User Interface for managing, training, and running inference on the Index-TTS text-to-speech model. This interface streamlines the entire ML pipeline from data preparation to final audio generation.

**ORIGINAL REPO :** [INDEX-TTS Official Repository](https://github.com/index-tts/index-tts)

**Note on Language Support:** This project is specifically designed for training the Turkish language; nevertheless, it can be used to train other Latin-based languages. For non-Latin languages, modifications to the code may be required.

## ✨ Features

This WebUI provides a modular, tabbed workflow:

* **Home:** Project management and real-time hardware monitoring (CPU, RAM, VRAM, Temps).
* **Models:** Model checkpoint selection and management.
* **Corpus & Dataset:** Audio and text data ingestion, formatting, and dataset compilation.
* **Tokenizer & Preprocessor:** Text tokenization and audio preprocessing pipelines for model ingestion.
* **Trainer:** Interface to configure and monitor Index-TTS model training/fine-tuning.
* **Inference:** Generate high-fidelity audio from text using trained checkpoints.
* **TTS:** A standalone inference engine that bypasses project settings for direct model loading, zero-shot controls, and rapid generation.

## 🧩 Prerequisites

* NVIDIA GPU (Highly recommended for Training and Inference)
* CUDA Toolkit compatible with your PyTorch installation
* Windows 10+

## 🚀 Installation

To set up the ITTS-TR environment properly, please follow these steps:

1. **Get the Repository:** Clone or download this repository to your local machine.
2. **Run the Installer:** Navigate to the **bat** folder containing the setup scripts and double-click the `install.bat` file. 
3. **Follow the On-Screen Prompts:** The batch script will guide you through the following automated setup phases:
   * **Git Installation:** You will be prompted to install a portable version of GitHub if you do not already have it.
   * **Python Setup:** Enter Python version **3.11.9** when prompted. The script will download, extract, and configure an isolated Python environment.
   * **Base Dependencies:** The script automatically installs the core Python requirements defined in `requirements.txt`.
   * **PyTorch & CUDA Configuration:** You will be asked if you want to install PyTorch with CUDA support. If you proceed, you can select your preferred CUDA version (12.6, 12.8, or 13.0) to ensure proper GPU acceleration. Version **12.8** is highly recommended.
   * **FFmpeg Installation:** You will be prompted to install FFmpeg, with options to choose either the Stable (v7.1.1) or the Latest Release.
   * **yt-dlp:** You can optionally choose to install the `yt-dlp` executable for media downloading.
   * **Core Model Cloning & Patching:** Finally, the script will ask you to clone the sparse `index-tts` repository and apply mandatory dependency fixes.

### 🔑 Hugging Face Token Configuration (`HF_TOKEN`)

The `paths.bat` configuration file contains an `HF_TOKEN` environment variable. This token is strictly required to authenticate and download certain gated models and weights from the Hugging Face Hub. 

If you do not already have an `HF_TOKEN` configured as a global environment variable on your Windows system, you must open `paths.bat` in a text editor and manually insert your Hugging Face access token before attempting to download models in the WebUI.

---

## 📚 Usage

To launch the interface, run the webui batch script from the root directory:

```bat
webui.bat
```

The application will generate a `projects/` directory to store your workspace data and a `wui.json` file for your global UI preferences (like language settings). Open the local URL provided in your terminal (typically `http://127.0.0.1:7860`) in your browser.

To launch the tensorboard, run the tensorboard batch script located inside the bat folder:

```bat
bat\tensorboard.bat
```

---

## ⚡ Triton

To achieve maximum training and inference speed by utilizing dynamically compiled GPU kernels, you can enable OpenAI's Triton. Since Triton compiles kernels natively at runtime, Windows users must configure a strict build environment.

**System Requirements for Triton:** 1. **Visual Studio C++ Build Tools:** Download the Visual Studio Installer and install the **"Desktop development with C++"** workload. This provisions the essential MSVC compiler (`cl.exe`).

2. **NVIDIA CUDA Toolkit:** Install the official standalone CUDA Toolkit. The version must exactly match the CUDA version you selected for PyTorch during the `install.bat` phase (e.g., 12.6, 12.8, or 13.0). 

3. **Strict Path Configuration:** The dynamic compiler relies on hardcoded system paths. Ensure that your `paths.bat` file is configured so that its directory variables strictly match your local MSVC and CUDA Toolkit installation paths. If `nvcc` or `cl.exe` cannot be located by the batch script's internal routing, Triton will fail to compile the kernels.

---

## 🛡️ License & Legal Disclaimer

This repository utilizes a dual-licensing structure:

**1. User Interface & Wrapper Code (Apache 2.0)**
The overarching Gradio interface, project management logic, and utility scripts located in the root directory are licensed under the **Apache License 2.0**. See the `LICENSE` file in the root directory for full details.

**2. Index-TTS Core Model (Bilibili Model Use License Agreement)**
The core text-to-speech model, model weights, and specific training code located within the `indextts/` directory are owned by Bilibili and are strictly governed by the Bilibili Model Use License Agreement. You can find this agreement in the [official gitHub repository](https://github.com/index-tts/index-tts) of index-tts. By using this software, you agree to comply with its terms, including prohibitions on high-risk deployment.

### Required Disclaimer

*Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.*