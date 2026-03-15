### 👁️ Overview

The Dataset Preparation module is strictly responsible for acquiring, standardizing, and structuring audio-text pairs into a unified format for acoustic model training. Regardless of the input source, this module generates a standardized output consisting of a `wavs/` directory containing audio clips and a structured `metadata.csv` mapping each file to its normalized text transcript.

#### 🤗 Method 1: Hugging Face Dataset

This method allows you to directly pull and process pre-existing speech datasets from the Hugging Face Hub.

- **Hugging Face Dataset:** The repository ID (e.g., `erenfazlioglu/turkishvoicedataset`). The system automatically downloads the `train` split and parses the internal structure.
- **Target Folder Name:** Defines the local directory name where the dataset will be stored under `datasets/<language>/<target_folder_name>`.
- **Automated Extraction:** The pipeline safely extracts text across varying dataset schemas (looking for keys like `transcription`, `text`, or `sentence`) and extracts the raw audio bytes directly into `.wav` files.

#### ✂️ Method 2: Custom Audio Slicer

This method processes a single, long-form audio file (e.g., a podcast or audiobook) into thousands of short, transcribed training clips.

- **Upload Audio:** Select your local long-form audio file.
- **Max Clip Duration (sec):** Defines the hard limit for a single audio clip. If a speech segment exceeds this duration, the system intelligently slices it further based on timestamp boundaries provided by Whisper.
- **VAD & Diarization:** Behind the scenes, the system initializes `pyannote/speaker-diarization-3.1` to perform Voice Activity Detection, isolating actual speech and ignoring long silences. *(Note: This requires a valid `HF_TOKEN` environment variable).*
- **Whisper Transcription:** Each isolated speech segment is fed into the `large-v3` Whisper model to generate a highly accurate text transcription in your target language.

#### ⚙️ Core Configuration & Controls

Both processing methods share critical parameters for standardizing your audio data.

- **Language:** Assigns the language tag (e.g., `tr`, `en`) which dictates both the output directory structure and the language forced upon the Whisper transcription model.
- **Resample To:** Forces the audio into a specific sample rate (16kHz, 22.05kHz, 24kHz, 44.1kHz, or 48kHz). If set to `None`, the original sample rate of the source audio is preserved.
- **Save Every X Clips:** Controls how often the `metadata.csv` is flushed to disk. Lower numbers provide safety against crashes, while higher numbers slightly improve processing speed.

#### 🔄 Processing & Normalization Pipeline

To ensure the acoustic model receives mathematically clean data, all text and audio pass through a strict pipeline before saving:

1. **Audio Formatting:** Audio arrays are converted and saved as single-channel (mono) `.wav` files at the designated sample rate.
2. **Text Wordification:** Raw text is passed through the `TurkishWordifier` to expand numbers, dates, and symbols into their spoken word equivalents (e.g., "1919" becomes "bin dokuz yüz on dokuz").
3. **Text Normalization:** The expanded text is then fed into the `TurkishWalnutNormalizer` to handle casing, punctuation standardization, and language-specific orthography rules.
4. **State Management:** The module tracks processed filenames. If interrupted, clicking **♻️ Resume Process** will scan the existing `metadata.csv` and automatically skip audio clips that have already been formatted, preventing duplicated effort.
