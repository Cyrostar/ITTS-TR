### 👁️ Overview

The Language Corpus module is the foundational data-gathering phase of the TTS pipeline. A high-quality acoustic model requires a tokenizer that has seen millions of words to understand the target language's structure. This module has been massively upgraded to utilize a high-performance, multi-core SQLite database (`corpus.db`) to aggregate raw text from PDFs and TXT files, alongside a powerful suite of audio extraction and transcription tools to generate new text data.

#### 🗄️ 1. Database Processing Engine

This section walks through the core tabs used to build your vocabulary database, which replaces the legacy `corpus.txt` system.

- **PDF Corpus Builder:** Point the system to a folder of PDFs. It utilizes all available CPU cores to extract text in parallel, breaking them down into readable chunks and safely storing them into the database.
- **Text Normalizer:** Reads the raw chunks using fault-tolerant B-Tree pagination and processes them through the Multilingual Normalizer via a persistent CPU pool. It saves unique chunks and aggregates their exact occurrence counts.
- **Word Extractor:** Scans the normalized database to extract distinct individual words, calculating their exact frequencies across your entire dataset.
- **Syllabifier:** Processes the normalized text through the Turkish Syllabifier (or language equivalents) to break down text into phonetic syllables. It multiplies frequencies by chunk occurrence to find the absolute most common phonetic sounds.
- **Vocabulary Statistics:** An analytical view to check the Top 10 words and syllables in your database. You can also export the Top 2000 lists directly to JSON files for external use.
- **Tokenizer:** Trains a Byte Pair Encoding (BPE) SentencePiece model directly from the pre-normalized text chunks in your database. It automatically forces the top 1000 highest-frequency words and syllables into the vocabulary to ensure phonetic stability.

#### 🧰 2. Workspace & Upload Utilities

- **Add Documents:** Drop your raw text documents here. You can either save them to your local project folders or use the "Process & Merge to DB" button to instantly chunk, normalize, and inject them directly into your database.
- **File Repositories:** Displays the successfully processed PDF and TXT files currently residing in your project's workspace.

#### 🧽 3. Audio Acquisition & Cleaning

If you lack raw text but have access to speech, these tools help you extract and prepare audio for transcription.

- **YouTube Downloader:** Paste a URL to immediately fetch and extract the highest quality audio track from a video. Excellent for gathering podcast or interview data.
- **Audio Cleaner (Demucs):** Raw audio often contains background music or noise that ruins transcription and acoustic training. This tool uses the `htdemucs` neural network to mathematically isolate the human vocal track and discard the background noise.

#### 🎙️ 4. Transcription & Diarization

Convert your clean audio into usable text and separated speaker files.

- **Audio Transcriptor (Whisper):** Feeds your audio into OpenAI's Whisper model (up to `large-v3`) to generate highly accurate, punctuated text. 
  - *Normalizer Toggle:* Automatically passes the Whisper output through the normalizer so the text is perfectly formatted for TTS training.
- **Diarization (Speaker Separation):** Uses the `pyannote/speaker-diarization-3.1` model to detect multiple speakers (up to a user-defined max) in a single audio file.
  - *Trim Silence:* Automatically stitches the detected segments together based on your gap settings.
  - *Speaker Files:* Exports an independent `.wav` file for every unique speaker detected, completely isolating their speech for targeted dataset creation.

#### 🏷️ 5. File Standardization

Machine learning pipelines require strict, predictable file paths.

- **Document Namer & Audiobook Namer:** These utilities sanitize your input strings (e.g., converting Turkish characters like 'ç' to 'c', replacing hyphens with spaces, and enforcing underscores) to create a strict naming convention (`Genre-Author-Title` or `Audiobook-Source-Narrator-Genre-Author-Title`). Use these *before* uploading files to keep your workspace structurally sound.