### 👁️ Overview

The Preprocessor (Feature Extraction) module is a critical bridge between your raw dataset and the acoustic training phase. Raw audio and text cannot be fed directly into the TTS model. Instead, this module passes your data through multiple pre-trained neural networks to extract high-dimensional mathematical representations (features) and saves them as `.npy` arrays.

#### 📂 1. Data Source & Core Settings

Before extracting features, you must define the target dataset and how the data should be structured for training.

- **Target Dataset:** The specific dataset folder (created in Phase 3) containing your `metadata.csv` and `wavs/` directory.
- **Folder Language:** Filters the dataset dropdown to show datasets belonging to a specific language tag.
- **Inject Language Marker:** Determines if a specific language ID token should be prepended to the text tokens.
  - **None:** No language ID is injected.
  - **TR (ID-3) / EN (ID-4):** Forces the model to recognize the language explicitly, which is crucial for multi-lingual models.
- **Validation Split (%):** Determines the percentage of your dataset to withhold from training. This withheld data (`val.jsonl`) is used to test the model's accuracy on unseen data during the training phase.

#### ⚡ 2. Performance Settings

Feature extraction is highly resource-intensive. These settings allow you to balance speed against your hardware's limits.

- **Batch Size:** The number of audio clips processed simultaneously by the GPU. Lower this value if you encounter CUDA Out-Of-Memory (OOM) errors.
- **CPU Workers:** The number of parallel CPU threads dedicated to loading audio files from your disk. Higher values speed up the data pipeline but consume more system RAM.

#### ⚙️ 3. Advanced Configuration

- **Use Relative Paths:** When checked, the generated manifest files (`.jsonl`) will store relative paths instead of absolute paths. This is highly recommended as it allows you to move your project folder to another drive or machine without breaking the dataset links.
- **Use Merged Tokenizer:** Instructs the extractor to use the `_bpe_merged.model` instead of the standard tokenizer. Use this only if you have explicitly merged multiple tokenizers.
- **Torch Compile:** Uses PyTorch 2.0+ `torch.compile()` to optimize the extraction models. This significantly speeds up the extraction process but requires an initial "warm-up" period where the process will appear frozen.

#### 🧲 4. Under the Hood: What is Extracted?

For every valid audio-text pair, the system extracts four specific features and saves them in the `extractions/<dataset_name>` folder:

1. **Text IDs (`text_ids/`):** The raw text is tokenized using your custom SentencePiece model into a sequence of integers.
2. **Semantic Codes (`codes/`):** The audio is passed through the `W2V-BERT 2.0` feature extractor and quantized using the `MaskGCT` semantic codec to create discrete audio tokens.
3. **Conditioning (`condition/`):** High-level acoustic features processed by the UnifiedVoice GPT model.
4. **Emotion Vectors (`emo_vec/`):** Emotional and prosodic embeddings extracted from the semantic features.

Finally, the module generates two manifest files (`train.jsonl` and `val.jsonl`) which map the original audio paths to these newly generated `.npy` feature arrays.
