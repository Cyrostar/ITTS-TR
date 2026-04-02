### 👁️ Overview

The Tokenizer module is strictly responsible for training a Byte Pair Encoding (BPE) model using SentencePiece, which translates your raw text into numerical sequences understandable by the TTS acoustic model. It handles vocabulary generation, character coverage, text normalization, and special token injection.

#### 📂 1. Data Selection

Before training, you must define the textual foundation the tokenizer will learn from.

- **Select Language & Dataset:** Choose your target language and the specific dataset folder. The UI will automatically parse the `metadata.csv` associated with this dataset.

- **Metadata Coverage:** A slider (10% - 100%) that allows you to sample a specific percentage of your dataset's metadata for training. Useful for rapid prototyping on massive datasets.

- **Include Unified Corpus Text:** Appends the contents of your `corpus.db` database to your dataset metadata.

- **Train Only With Corpus Text:** Ignores the dataset metadata entirely and trains exclusively on the `corpus.db` database.

#### 🧠 2. Vocabulary & Coverage Configuration

- **Vocabulary Size:** Determines the maximum number of unique tokens (subwords/words) the model can memorize. The slider ranges from 2,000 to 30,000, with a default of 12,000. *Engineering Note: Selecting a dataset will automatically attempt to sync this value with the `number_text_tokens` defined in your project's `config.yaml`.*

- **Character Coverage:** Defines the percentage of raw character variations to encompass within the model. The default is `1.0` (100%).

#### 🏷️ 3. Special Tokens & Tags
- **Style & Emotion Tags:** Automatically injects predefined conversational, narrative, and emotional state tags (e.g., `[happy]`, `[whisper]`, `[podcast]`) to teach the acoustic model highly expressive delivery boundaries.
- **Alphabet Extensions:** Checkboxes to explicitly force the tokenizer to memorize standard English letters, Turkic extended characters, Turkish long vowels (e.g., `â`, `î`), and standard punctuation marks.
- **Custom Special Tokens:** A text field where you can define specific symbols, currencies, or characters (separated by `|`) to manually lock into the vocabulary.

#### 💉 4. Phonetic & Linguistic Injections
These core features bypass the statistical BPE algorithm. They force the Tokenizer to permanently lock specific linguistic units into its vocabulary matrix, guaranteeing perfect acoustic alignment during TTS synthesis.

- **Inject High-Frequency Syllables:** Queries your compiled `corpus.db` for the absolute most common syllables across your datasets (governed by the **Syllable Count** value) and locks them in. This provides strict phonetic anchors that drastically reduce slurring and skipped audio artifacts.
- **Inject High-Frequency Words:** Queries the database for the most frequent whole words (governed by the **Word Count** value). Hardcoding frequent words allows the model to learn their distinct, natural prosody (rhythm and intonation) as a single acoustic embedding, rather than stitching them together robotically.
- **Vocabulary Capacity Engine:** The system dynamically calculates if your forced injections (tags + syllables + words) leave enough mandatory space (at least 256 slots) for the basic linguistic alphabet and control bytes. It will safely halt the pipeline if it detects a vocabulary overflow risk.

#### ⚙️ 5. Advanced Training Rules
- **Normalization & Casing:** Determine whether to bypass the SPM internal normalizer (`identity` rule) and whether to force your vocabulary into strict uppercase or lowercase formats.
- **Max Sentences (Sample Size):** Limits RAM usage on massive datasets by capping the parsed lines. Set to 0 to use all available sentences.
- **Train Extremely Large Corpus:** Engages C++ memory optimizations for parsing multi-gigabyte training streams.
- **Shuffle Corpus:** Randomizes the parsed input streams to ensure a uniform linguistic distribution.
- **Hard Vocab Limit:** Strictly enforces the requested vocabulary size without padding the trailing space.

### 🧰 Utilities

These diagnostic tools allow you to validate your text processing pipeline before committing to acoustic model training.

#### 🎗️ Tokenizer Safety Check

An automated validation suite to check tokenizer suitability for TTS.

- Upload your trained SentencePiece `.model` file to verify it successfully captures standard characters.
- It tests explicit mapping of special characters based on your normalization and casing rules.
- It checks for the presence of detrimental byte-fallback tokens.
- It executes sample tokenization on complex words to ensure no `<unk>` tokens are generated.

#### 💱 Tokenizer Tester

A direct inference tool to visualize how your model breaks down text.

- Input raw text to see exactly how the active project's tokenizer splits it into subwords.
- Test both Standard (trained) and Merged model states.
- Outputs the total token count and a detailed array of `[ID] Piece` pairs.

#### 📚 Multilingual Wordifier

Tests numeric/date expansion and unique word extraction logic.

- Input text containing complex structures like numbers, dates, or abbreviations (e.g., "19.05.1919" or "2.500").
- **Return Format:** Toggle between "Full Block" (the expanded sentence) or "Word List" (a comma-separated array of the extracted words).

#### 🫧 Multilingual Normalizer

Tests the preprocessing logic on raw text.

- Input messy text containing mixed casing, punctuation errors, or special symbols.
- The output reveals exactly how the acoustic model will "read" the text after normalization rules and abbreviation expansions are applied.

#### ✂️ Turkish Syllabifier

Tests the Turkish syllabification, stress marking, and harmony algorithms.

- Input Turkish text to see how the system programmatically breaks it down into distinct phonetic syllables.
- Toggle advanced linguistic checks like stress markers, vowel harmony validation, and a detailed word-by-word analysis mode.

#### 🎨 Design Custom Model

A powerful visual interface to design a custom model from the original (Source) and the trained (Target) tokenizer models.

- Upload a target model to merge into the official baseline model.
- **Source Configuration:** Strip out unwanted elements from the base model like language markers, CJK tokens, English tokens, or source punctuation. You can also force lowercase conversion and inject required structural tokens.
- **Target Configuration:** Define exactly how the new model merges into the base. Choose whether to preserve standalone letters and punctuation, and strictly enforce the token casing rules.
- Provides a direct side-by-side output of the processed Source Vocab, processed Target Vocab, and the final Merged Output.