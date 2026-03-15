### 👁️ Overview

The Tokenizer module is strictly responsible for training a Byte Pair Encoding (BPE) model using SentencePiece, which translates your raw text into numerical sequences understandable by the TTS acoustic model. It handles vocabulary generation, character coverage, text normalization, and special token injection.

#### 📂 1. Data Selection

Before training, you must define the textual foundation the tokenizer will learn from.

- **Select Language & Dataset:** Choose your target language and the specific dataset folder. The UI will automatically parse the `metadata.csv` associated with this dataset.

- **Metadata Coverage:** A slider (10% - 100%) that allows you to sample a specific percentage of your dataset's metadata for training. Useful for rapid prototyping on massive datasets.

- **Include Unified Corpus Text:** Appends the contents of `corpus/corpus.txt` to your dataset metadata.

- **Train Only With Corpus Text:** Ignores the dataset metadata entirely and trains exclusively on the `corpus.txt` file.

#### 🧠 2. Vocabulary & Coverage Configuration

- **Vocabulary Size:** Determines the maximum number of unique tokens (subwords/words) the model can memorize. The slider ranges from 2,000 to 30,000, with a default of 12,000. *Engineering Note: Selecting a dataset will automatically attempt to sync this value with the `number_text_tokens` defined in your project's `config.yaml`.*

- **Character Coverage:** Defines the percentage of characters to retain from the raw text (0.99 to 1.0). Setting this to 1.0 ensures all rare characters (like Q, W, X in Turkish datasets) are kept.

#### 🔣 3. Special Tokens & Characters

A robust TTS tokenizer requires explicit awareness of punctuation, structural tags, and specific alphabetic edge cases.

- **Special Tokens:** A text field to manually introduce custom symbols (e.g., currency, math operators) separated by the pipe `|` character.

- **Character Injection Checkboxes:** Explicitly force the tokenizer to recognize specific character sets to prevent them from becoming `<unk>` (unknown) tokens:
  
  - Turkish characters (ç, ğ, ö, ş, ü).
  
  - English characters (q, w, x).
  
  - Turkic characters (ə, ұ, қ).
  
  - Long vowels (â, î, û).
  
  - Punctuation (. , ? ! ' : ; ...).

- **Automated Tag Injection:** Note that behind the scenes, the system automatically injects a comprehensive suite of style tags (e.g., `[casual]`, `[podcast]`) and emotion tags (e.g., `[happy]`, `[whisper]`) directly into the tokenizer vocabulary.

#### ⚙️ 4. Normalization & Augmentation

- **Normalization Rule:** Configures SentencePiece's internal normalizer (`nmt_nfkc`, `nfkc`, `none`, etc.). *Note: The text is already heavily normalized by the internal `TurkishWalnutNormalizer` prior to reaching SentencePiece.*

- **💉 Inject Word Samples:** Activating this injects common Turkish words into the training data to heavily influence the BPE algorithm toward creating whole-word tokens rather than sub-word fragments. The severity is controlled by the **Injection Multiplier** slider (1-100).

- **♻️ Deduplicate Data:** Removes identical duplicate lines from the training pool, preventing the tokenizer from heavily biasing toward repeated phrases.

### 🧰 Utilities

These diagnostic tools allow you to validate your text processing pipeline before committing to acoustic model training.

#### 🖊️ Turkish Tokenizer Safety Check

An automated validation suite for trained SentencePiece `.model` files.

- Upload your trained model to verify it successfully captures standard `a-z` characters.

- It tests explicit mapping of special Turkish characters (e.g., `Ç` → `ç`) based on your normalization rules.

- It checks for the presence of detrimental byte-fallback tokens.

- It executes sample tokenization on complex words to ensure no `<unk>` tokens are generated.

#### 💱 Tokenizer Tester

A direct inference tool to visualize how your model breaks down text.

- Input raw text to see exactly how the active project's tokenizer splits it into subwords.

- Outputs the total token count and a detailed array of `[ID] Piece` pairs.

#### 🫧 Turkish Walnut Normalizer

Tests the raw text preprocessing logic.

- Input messy text containing mixed casing, punctuation errors, or special symbols.

- The output reveals exactly how the acoustic model will "read" the text after normalization rules are applied.

#### 📚 Turkish Textblock Wordifier

Validates numerical and date expansion.

- Input text containing numbers (e.g., "19.05.1919" or "2.500").

- **Return Format:** Toggle between "Full Block" (the expanded sentence) or "Word List" (a comma-separated array of the extracted words).
