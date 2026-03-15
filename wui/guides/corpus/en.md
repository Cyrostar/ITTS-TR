### 👁️ Overview

The Language Corpus module is the foundational data-gathering phase of the TTS pipeline. A high-quality acoustic model requires a tokenizer that has seen millions of words to understand the target language's structure. This module allows you to aggregate raw text from PDFs and TXT files, and provides a powerful suite of audio extraction and transcription tools to generate new text data from raw audio sources.

#### 📘 1. Workspace & Repositories

This section is dedicated to building the master `corpus.txt` file, which will later be fed into the SentencePiece Tokenizer training phase.

- **Upload PDF/Text:** Drop your raw text documents here. The system parses the text, optionally filters for unique words to maximize vocabulary efficiency without bloating the file, and stores them in the project repositories.
- **Repositories:** Displays the successfully processed PDF and TXT files currently residing in your project's workspace.
- **Combine All Mix Files:** Compiles every processed document in your repositories into a single, massive `corpus.txt` file.

#### 🧰 2. Audio Acquisition & Cleaning

If you lack raw text but have access to speech, these tools help you extract and prepare audio for transcription.

- **YouTube Downloader:** Paste a URL to immediately fetch and extract the audio track from a video. Excellent for gathering podcast or interview data.
- **Audio Cleaner (Demucs):** Raw audio often contains background music or noise that ruins transcription and acoustic training. This tool uses the `Demucs` neural network to mathematically isolate the human vocal track and discard the background noise.

#### 🎙️ 3. Transcription & Diarization

Convert your clean audio into usable text and separated speaker files.

- **Audio Transcriptor (Whisper):** Feeds your audio into OpenAI's Whisper model to generate highly accurate, punctuated text.
  - *Walnut Normalizer:* Automatically passes the Whisper output through our custom normalizer so the text is perfectly formatted for TTS training.
- **Diarization (Speaker Separation):** Uses the `pyannote/speaker-diarization-3.1` model to detect multiple speakers in a single audio file.
  - *Trim Silence:* Automatically stitches the detected segments together, removing long pauses.
  - *Speaker Files:* Exports an independent `.wav` file for every unique speaker detected, completely isolating their speech for targeted dataset creation.

#### 🏷️ 4. File Standardization

Machine learning pipelines require strict, predictable file paths.

- **Document Namer & Audiobook Namer:** These utilities sanitize your input strings (e.g., converting Turkish characters like 'ç' to 'c', replacing spaces with underscores) and enforce a strict naming convention (`Genre_Author_Title.txt` or `Source_Narrator_Genre_Author_Title.wav`). Use these *before* uploading files to keep your workspace structurally sound.
