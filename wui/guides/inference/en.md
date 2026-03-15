### 👁️ Overview

The Inference module is the final stage of the pipeline, where your trained acoustic model generates speech from text. This interface allows you to test your custom fine-tuned voices, compare them against the official base model, and deeply manipulate the emotional delivery of the generated audio.

#### 📂 1. Model & Text Configuration

Before generating speech, you must select the model and provide the target text.

- **Folder Selection:** Choose your trained project from the dropdown. The system will automatically load the latest unwrapped checkpoint (`gpt.pth` or `latest.pth`) from your project's `trains/` directory.
- **Use Original Model:** Bypasses your custom training folder and loads the official, unmodified base model. This is excellent for comparing the baseline quality against your fine-tuned results.
- **Language:** Forces a specific language token (`TR` or `EN`) at the beginning of the text to guide the model's pronunciation rules. Setting this to `Auto` relies on the model's internal language guessing.
- **Input Text:** The text you want the model to read. 

#### 📢 2. Voice & Emotion Control

The UnifiedVoice architecture separates the speaker's acoustic identity (Voice) from their delivery (Emotion).

- **Reference Audio (Voice):** Upload a short (3-10 second) clean audio clip. The model will clone the acoustic characteristics (timbre, pitch, environment) of this speaker.
- **Control Mode:** Dictates how the model determines the emotional delivery of the text:
  - **Same as Reference:** The model extracts and copies the emotion directly from your Voice Reference Audio.
  - **Reference Audio:** Allows you to upload a *second* audio file specifically to extract emotion. (e.g., Voice of Person A, but crying like Person B).
  - **Emotion Vectors:** Unlocks manual sliders (Happy, Angry, Sad, Melancholy, etc.) allowing you to dial in exact emotional weights.
  - **Description Text:** Leverages a built-in LLM (Qwen) to interpret a text prompt (e.g., "whispering urgently") and mathematically convert it into emotion vectors.
- **Emotion Intensity:** A multiplier (0.0 to 1.0) that scales how strongly the selected emotion overrides the neutral baseline.

#### ⚙️ 3. Advanced Parameters

These settings control the underlying mathematical generation process.

- **Enable BigVGAN CUDA Kernel:** If your environment supports it (requires `ninja` and a C++ compiler), this significantly speeds up the final waveform generation.
- **Enable Sampling:** When checked, the model introduces slight randomness, making repeated generations sound slightly different and more natural. 
- **Temperature & Top P:** Controls the randomness of the sampling. Lower values make the speech highly predictable and stable, while higher values make it more expressive but prone to artifacts.
- **Max Tokens:** Hard limits to prevent infinite generation loops on exceptionally long segments.
