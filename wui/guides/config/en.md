### 👁️ Overview

Welcome to the heart of the TTS engine. The `config.yaml` file dictates the exact architectural dimensions, sequence limits, and audio processing math used during model training and inference. **Warning:** Changing advanced parameters mid-training will break your checkpoint. Only tweak these settings *before* starting a fresh training run.

#### 🧠 1. Core Hyperparameters

These are the most critical settings. They directly control the "brain size" of your model and determine how much GPU VRAM you will need.

- **Vocab Size (number_text_tokens):** This dictates the maximum number of unique tokens the GPT model can understand. This number *must* exactly match the vocabulary size of the Tokenizer (`.model` file) you will train in Phase 4.
- **Audio Sample Rate:** The target frequency for your audio processing. Standard values are 22050 Hz or 24000 Hz. Higher rates produce crisper audio but require significantly more computing power.
- **Max Text/Mel Tokens:** The maximum length of the input text or generated audio sequences. If you set text to 600, anything longer is truncated. Increasing these lengths will exponentially increase your GPU VRAM usage.
- **Model Dimension (model_dim):** The internal hidden size of the GPT transformer. 1024 or 1280 are standard for high-quality models.
- **Layers & Heads:** The number of transformer blocks stacked together (Layers) and parallel attention mechanisms (Heads). More layers mean better reasoning and prosody, but slower training.

#### 🔤 2. Advanced Tokenizer & Text Front-End

This section configures the text processing pipeline, determining how raw input text is sanitized, normalized, and converted into linguistic tokens before entering the acoustic model.

- **Language:** The target language code (e.g., `en`, `tr`, `es`). This dynamically routes text through the correct language-specific normalizer to apply the proper character whitelists and punctuation splitting rules.
- **Tokenizer Type & Vocab Type:** Defines the algorithmic approach used to split text into digestible network inputs (such as `bpe` for Byte-Pair Encoding or character-level parsing).
- **Case Format:** Determines if the text should be standardized to a specific casing (e.g., forcing all characters to lowercase or uppercase) to match the exact casing state the Tokenizer was trained on.
- **Wordify:** When enabled, the system expands numbers, dates, times, currencies, and mathematical symbols into their full spoken-word equivalents (e.g., "$5" becomes "five dollars").
- **Abbreviations:** Toggles the expansion of common, language-specific abbreviations (e.g., converting "Dr." to "Doctor" or "St." to "Street"). 
- **Extract (Grapheme Extraction):** When enabled, this function forcefully inserts spaces between every single character (e.g., "hello" becomes "h e l l o"). This is often required for specific character-level acoustic alignment models that do not use BPE sub-word merging.

#### 🎛️ 3. Dataset & Mel Setup

This section controls how your raw `.wav` audio files are mathematically converted into spectrograms that the neural network can read.

- **BPE Model:** The exact filename of the tokenizer model (e.g., `bpe.model`) that the dataset loader should look for.
- **N FFT (n_fft):** The size of the Fast Fourier Transform window. 1024 is the industry standard for 22kHz-24kHz audio.
- **Hop Length:** The number of audio samples between successive STFT frames. A value of 256 means the model takes a "snapshot" of the audio every 256 samples.
- **Win Length:** The size of the window function applied to the audio. Usually matches `n_fft` (1024).
- **N Mels:** The number of Mel-frequency bands to generate. 80 or 100 are standard.
- **Normalize Mel:** Whether to statistically normalize the spectrogram values. Keep this `False` unless your specific training script explicitly requires normalized inputs.

#### 🧩 4. GPT Token Logic

This section defines the structural boundaries and conditioning logic for the generative text-to-audio process.

- **Use Mel Codes as Input:** When `True`, the model feeds acoustic tokens back into itself autoregressively during training.
- **Train Solo Embeddings:** A specialized flag for isolating embedding layers during specific fine-tuning stages.
- **Condition Type:** Defines the architectural module used to bridge text and audio. `conformer_perceiver` is a highly advanced, efficient cross-attention mechanism.
- **Start/Stop Tokens:** These are strict ID numbers that tell the model when an audio sequence or text sequence begins and ends (e.g., `start_text_token` defaults to 0).
- **Number Mel Codes:** The total vocabulary size of your semantic audio codec.

#### 🔗 5. Checkpoints & Vocoder

Paths and definitions for the external weights and components your pipeline relies on.

- **Checkpoints (gpt.pth, s2mel.pth):** The relative filenames where the system will save (or resume from) the GPT and Semantic-to-Mel model weights.
- **W2V Stat & Matrices:** Paths to pre-calculated statistical tensors (like `wav2vec2bert_stats.pt`, `feat1.pt`) used for speaker and emotion conditioning.
- **Qwen Emo Path:** Directory path for the underlying LLM used for emotion extraction.
- **Vocoder Type & Name:** The Vocoder is the neural network responsible for taking the AI-generated spectrogram and turning it back into an audible `.wav` file. `bigvgan` is a state-of-the-art vocoder that produces exceptionally crisp, human-like voice artifacts.

#### ⬡ → ◯  Model Resizing & Weight Preservation in Index-TTS

When generating a new configuration for the UnifiedVoice architecture, the system employs an intelligent weight-transfer algorithm to ensure that your pre-trained weights are never unnecessarily lost. This guide explains the mathematical mechanics behind this process.

### 1. The Core Slicing Mechanism

The preservation of trained weights relies on calculating the overlapping mathematical boundary between the original pre-trained tensor and your newly configured tensor.

The system achieves this using a dynamic slicing operation that evaluates the minimum dimension size between the old and new shapes (`min(ds, ts)`). This ensures that we only transfer the exact intersection of data that fits perfectly into the new computational graph.

### 2. Resizing Scenarios

Depending on how you adjust the parameters in the configuration UI, the model handles the pre-trained weights in three distinct ways:

- **Identical Layers (No Change):** If core structural parameters like `model_dim`, `layers`, and `heads` remain untouched, the deep network components (Transformer attention blocks, feed-forward layers, and normalization layers) map 1:1. The pre-trained weights are copied perfectly without alteration.

- **Expansion (e.g., Increasing Vocabulary):** If you increase the `number_text_tokens` (e.g., from 10,000 to 12,000), the original 10,000 trained embeddings are copied directly into the new tensor. The newly added 2,000 slots are initialized with random, untrained weights, ensuring the model retains its foundational knowledge of the base characters.

- **Truncation (e.g., Decreasing Context):** If you shrink a parameter, the script mathematically crops the tensor. It preserves the trained weights from index 0 up to your new cutoff limit. Because tokenizers typically sort by frequency, this safely retains the most critical and highly-trained tokens.

### 3. The Exception: Dimension Mismatches

The only scenario where trained weights are discarded is if you drastically alter a core architectural dimension, such as changing `model_dim` from 1280 to 512, which creates an incompatible shape.

In such cases, the system relies on a graceful degradation fallback:

- It detects the dimension mismatch.

- It safely skips that specific layer.

- It logs a precise warning in the UI (e.g., `Skipped layer... Dimension mismatch`).

- It initializes only that incompatible layer from scratch while salvaging the rest of the compatible network components.
