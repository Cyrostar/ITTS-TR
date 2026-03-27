import gradio as gr
import os
import sys
import re
import json
import string
import random
import yaml
import pandas as pd
import sentencepiece as spm
from datetime import datetime

from core import core
from core.core import _
from core import injection
from indextts.utils.front import TextTokenizer
from core.spice import SentencePieceTrainerWrapper
from core.spice import GenericSpiceTokenizer
from core.spice import JsonToModelConverter
from core.normalizer import MultilingualNormalizer, MultilingualWordifier
from core.database import SQLiteManager

# --- HELPER FUNCTIONS ---

def list_datasets(lang):
    if not lang:
        return []
    ds_path = os.path.join(core.path_base, "datasets", lang)
    if not os.path.exists(ds_path):
        return []
    
    # Return only directories inside the language folder
    return sorted([
        d for d in os.listdir(ds_path) 
        if os.path.isdir(os.path.join(ds_path, d))
    ])
    
def load_text_from_corpus():
    """
    Reads unique text chunks from the unified corpus database.
    """
    texts = []
    db_path = os.path.join(core.corpus_directory(), "corpus.db")

    if not os.path.exists(db_path):
        print(f"[Corpus] Warning: {db_path} not found.")
        return texts

    try:
        db = SQLiteManager(db_path)
        records = db.fetch_all("SELECT text FROM normalized_chunks WHERE text IS NOT NULL")
        for row in records:
            chunk = row["text"].strip()
            if chunk:
                texts.append(chunk)
    except Exception as e:
        print(f"[Corpus] Failed to read from {db_path}: {e}")

    return texts

def load_text_from_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return []
    
    texts = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 5:
                texts.append(parts[4])
    return texts
       
def on_dataset_select(lang, dataset_name):
    """
    Reads config.yaml from project/configs/ to return the 
    number_text_tokens value for the tokenizer UI slider.
    """
    logs = []
    new_vocab_size = 12000 # Default fallback
    
    if not dataset_name:
        return gr.update(value=new_vocab_size), ""

    try:
        # 1. Target Path
        target_dir = core.configs_directory()
        target_yaml = os.path.join(target_dir, "config.yaml")
            
        # 2. Read YAML for Vocab Size (from the project-specific file)
        if os.path.exists(target_yaml):
            with open(target_yaml, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                
                # Look inside the 'gpt' section for the key
                if config_data and "gpt" in config_data and "number_text_tokens" in config_data["gpt"]:
                    new_vocab_size = int(config_data["gpt"]["number_text_tokens"])
                    logs.append(f"📊 Updated Vocab Size slider to: {new_vocab_size} (found in project config.yaml)")
                else:
                    logs.append("⚠️ 'number_text_tokens' not found inside 'gpt' section of config.yaml. Using default 12000.")
        else:
            logs.append("ℹ️ No config.yaml found in project. Using default vocab size (12000).")

    except Exception as e:
        logs.append(f"❌ System Error reading config: {str(e)}")

    # Return the new slider value and the log string
    return gr.update(value=new_vocab_size), "\n".join(logs)

# --- CORE: Train Function ---
def train_tokenizer_ui(
    lang,
    dataset_name,
    vocab_size, 
    data_coverage,
    char_coverage,
    include_corpus,
    special_tokens_str,
    style_chk,
    emotion_chk,
    tr_spec_chk,
    tr_seng_chk, 
    tr_turk_chk,    
    tr_long_chk, 
    tr_punc_chk,
    use_only_corpus,
    norm_rule,
    case_rule,
    inject_words,
    multiplier,
    deduplicate,
    sentence_size,
    train_extremely,
    shuffle_sentences,
    hard_vocab,
    progress=gr.Progress()
):
    logs = []
    def log(msg): 
        logs.append(msg)
        return "\n".join(logs)

    # 1. Validate Inputs
    if not use_only_corpus and not dataset_name:
        return log("❌ Error: No dataset selected!")

    # 2. Setup Paths & Sync Config
    out_dir = core.tokenizer_directory()
    base_name = core.project_name if core.project_name else "myproject"
    model_prefix = os.path.join(out_dir, f"{base_name}_bpe")
    
    config_yaml_path = os.path.join(core.configs_directory(), "config.yaml")
    
    # 📝 Update the config.yaml with the user's selected vocab_size
    if os.path.exists(config_yaml_path):
        try:
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                
            if config_data and "gpt" in config_data:
                old_vocab = config_data["gpt"].get("number_text_tokens", "unknown")
                if old_vocab != int(vocab_size):
                    config_data["gpt"]["number_text_tokens"] = int(vocab_size)
                    with open(config_yaml_path, "w", encoding="utf-8") as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    yield log(f"📝 Synced config.yaml: Vocab Size updated from {old_vocab} to {vocab_size}")
                else:
                    yield log(f"✅ Config.yaml Vocab Size is already synced ({vocab_size})")
        except Exception as e:
            yield log(f"⚠️ Could not sync config.yaml: {e}")
    else:
        yield log(f"⚠️ No config.yaml found in project! Training with {vocab_size}, but config was not updated.")

    # Debug logs
    if dataset_name:
        yield log(f"🔍 Selected Dataset: {dataset_name}")
    else:
        yield log("ℹ️ No Dataset Selected (Using generic name)")
        
    yield log(f"💾 Output Model: {model_prefix}")

    metadata_texts = []
    corpus_texts = []

    # 3. Branching Logic for Data Loading
    if use_only_corpus:
        yield log("⚠️ Mode: ONLY CORPUS (Ignoring Dataset Metadata)")
        
        # Force load corpus
        yield log("📚 Reading corpus/corpus.txt ...")
        corpus_texts = load_text_from_corpus()
        yield log(f"   - Corpus lines: {len(corpus_texts)}")
        
    else:
        # Standard Mode: Metadata + Optional Corpus
        metadata_path = os.path.join(core.path_base, "datasets", lang, dataset_name, "metadata.csv")
        yield log(f"📂 Metadata Path: {metadata_path}")

        if not os.path.exists(metadata_path):
            return log("❌ Error: metadata.csv not found in that dataset folder!")

        yield log("⏳ Reading texts from metadata.csv ...")
        metadata_texts = load_text_from_metadata(metadata_path)
        yield log(f"   - Metadata lines: {len(metadata_texts)}")
        
        if data_coverage < 100 and len(metadata_texts) > 0:
            sample_size = int(len(metadata_texts) * (data_coverage / 100.0))
            # Randomly select the percentage of lines
            metadata_texts = random.sample(metadata_texts, sample_size)
            yield log(f"   - Metadata lines (Sampled {data_coverage}%): {len(metadata_texts)}")
        
        if include_corpus:
            yield log("📚 Reading corpus/corpus.txt ...")
            corpus_texts = load_text_from_corpus()
            yield log(f"   - Corpus lines: {len(corpus_texts)}")
        else:
            yield log("⏭️ Skipping corpus file.")
        
    # 5. Normalization (Replaces simple lowercase)
    yield log("⬇️ Normalizing all text (MultilingualNormalizer)...")
    
    # Determine the uppercase flag based on user selection
    is_upper = (case_rule == "uppercase")

    # Initialize the normalizer (handles Lowercase, Punctuation, Ellipsis)
    normalizer = MultilingualNormalizer(lang=lang, wordify=True, abbreviations=True, upper=is_upper)
    
    # 1. Process Metadata Texts (Wordify & Case integrated via Normalizer)
    if metadata_texts:
        metadata_texts = [normalizer.normalize(t) for t in metadata_texts]
        
    # 2. Process Corpus Texts (Wordify & Case integrated via Normalizer)
    if corpus_texts:
        corpus_texts = [normalizer.normalize(t) for t in corpus_texts]
        
    if inject_words:
        yield log(f"💉 Injecting extra words to Training & Corpus (Multiplier: {multiplier})...")
        extra_text = injection.tr_corpus(multiplier,True)
        extra_text_processed = normalizer.normalize(extra_text)
        try:
            db_path = os.path.join(core.corpus_directory(), "corpus.db")
            if os.path.exists(db_path):
                db = SQLiteManager(db_path)
                upsert_query = """
                    INSERT INTO normalized_chunks (text, occurrence_count) 
                    VALUES (?, ?) 
                    ON CONFLICT(text) DO UPDATE SET 
                    occurrence_count = normalized_chunks.occurrence_count + excluded.occurrence_count
                """
                db.execute_write(upsert_query, (extra_text_processed, 1))
        except Exception as e:
            yield log(f"⚠️ Warning: Could not append injected text to corpus DB: {e}")

        corpus_texts.append(extra_text_processed)
        
    raw_combined = metadata_texts + corpus_texts
    
    # 6. Combine + deduplicate
    if deduplicate:
        train_data = list(dict.fromkeys(raw_combined))
        yield log(f"✅ Deduplication ON: Reduced {len(raw_combined)} to {len(train_data)} unique lines.")
    else:
        train_data = raw_combined
        yield log(f"ℹ️ Deduplication OFF: Using all {len(train_data)} lines (including repeats).")
    
    if not train_data:
        return log("❌ Error: No training data found!")
    
    # 7. Add Language Tags 

    user_symbols = []
    
    lang_tags = [f"▁[{l}]" for l in core.language_list()]

    for tag in lang_tags:
        if tag not in user_symbols:
            user_symbols.append(tag)        
        
    # 8. Add style and emotions
    
    style_tags = [
        # Conversational
        "[casual]", "[friendly]", "[intimate]", "[chatty]",
        
        # Narrative & Content
        "[storytelling]", "[narration]", "[documentary]",
        
        # Podcast / Broadcast
        "[podcast]", "[radio]", "[host]",
        
        # Informative / Authority
        "[educational]", "[explainer]", "[instructional]",
        "[professional]", "[authoritative]",
        
        # Expressive / Performance
        "[dramatic]", "[theatrical]", "[comedic]",
        "[motivational]", "[inspirational]",
        
        # Reading / Scripted
        "[audiobook]", "[scripted]", "[monologue]"
    ]
        
    emotion_tags = [
        # Emotions (High Energy/Negative)
        "[angry]", "[furious]", "[shouting]", "[fearful]", "[disgusted]", "[afraid]",
        
        # Emotions (Positive/High Energy)
        "[happy]", "[elated]", "[cheerful]", "[excited]", "[joyful]", "[suprised]", 
        
        # Emotions (Low Energy/Sad)
        "[sad]", "[depressed]", "[resigned]", "[melancholic]",
        
        # Delivery Styles
        "[whisper]", "[softly]", "[sarcastic]", "[dryly]", 
        "[matter-of-fact]", 
        
        # Neutral/Reset
        "[neutral]", "[calm]", "[normal]", 
        
        # Vocalizations/Sounds
        "[laughs]", "[giggles]", "[chuckles]", "[laughing]",
        "[sighs]", "[clears-throat]", "[gasps]", "[breathing-heavily]",
        
        # Pacing & Timing
        "[pause]", "[long-pause]", "[hesitates]", 
        "[fast]", "[slow]"
    ]
    
    if style_chk:
        for tag in style_tags:
            if tag not in user_symbols:
                user_symbols.append(tag)
                
        yield log(f"🎭 Added {len(style_tags)} Style Tags to Tokenizer")
    
    if emotion_chk:
        for tag in emotion_tags:
            if tag not in user_symbols:
                user_symbols.append(tag)
                
        yield log(f"🎭 Added {len(emotion_tags)} Emotion Tags to Tokenizer")

    # 9. Parse Special Tokens
      
    tr_spec = ["ç", "▁ç", "ğ", "▁ğ", "ö", "▁ö", "ş", "▁ş", "ü", "▁ü"]
    
    tr_long = ["â", "▁â", "î", "▁î", "û", "▁û"]
    
    tr_seng = ["q", "▁q", "w", "▁w", "x", "▁x"]
    
    tr_turk = ["ə", "▁ə", "x", "▁x", "q", "▁q", "ә", "▁ә", "ғ", "▁ғ", "қ", "▁қ", "ң", "▁ң", "ө", "▁ө", "ұ", "▁ұ", "ү", "▁ү", "җ", "▁җ", "ä", "▁ä", "ž", "▁ž", "ň", "▁ň", "ý", "▁ý"]
    
    tr_punc = [".", "▁.", ",", "▁,", "?", "▁?", "!", "▁!", "'", "▁'", ":", "▁:", ";", "▁;", "...", "▁...", "\"", "▁\""]
                          
    if tr_spec_chk:
        for tag in tr_spec:
            if tag not in user_symbols:
                user_symbols.append(tag)
                
    if tr_long_chk:
        for tag in tr_long:
            if tag not in user_symbols:
                user_symbols.append(tag)
                
    if tr_seng_chk:
        for tag in tr_seng:
            if tag not in user_symbols:
                user_symbols.append(tag)
    
    if tr_turk_chk:
        for tag in tr_turk:
            if tag not in user_symbols:
                user_symbols.append(tag)
                
    if tr_punc_chk:
        for tag in tr_punc:
            if tag not in user_symbols:
                user_symbols.append(tag)
    
    if special_tokens_str:
        user_symbols.extend([x.strip() for x in special_tokens_str.split("|") if x.strip()])
        yield log(f"✨ Found {len(user_symbols)} Special Tokens")

    # 10. Apply Casing
    if case_rule == "uppercase":
        user_symbols = [sym.upper() for sym in user_symbols]
        yield log(f"🔠 Converted {len(user_symbols)} Special Tokens to uppercase")

    # 11. Stream Directly from RAM (Bypassing Disk I/O)
    yield log("🌊 Streaming data directly from RAM to C++ backend via iterator...")
    
    # 12. Train SentencePiece
    yield log(f"🧠 Training BPE Tokenizer (Vocab: {vocab_size}, Coverage: {char_coverage})...")
        
    try:
        # 1. Initialize Trainer
        trainer = SentencePieceTrainerWrapper(
            vocab_size=int(vocab_size),
            model_type="bpe",
            character_coverage=float(char_coverage),
            user_defined_symbols=user_symbols,
            split_digits=True,
            normalization_rule_name="identity" if norm_rule == "none" else norm_rule,
            hard_vocab_limit=bool(hard_vocab),
            train_extremely_large_corpus=bool(train_extremely),
            input_sentence_size=int(sentence_size),
            shuffle_input_sentence=bool(shuffle_sentences)
        )
        
        # 2. Execute training using the Iterator
        wrapper_logs = trainer.train(
            model_prefix=model_prefix,
            sentence_iterator=iter(train_data)
        )
        
        # 3. Yield the structured logs back to the Gradio UI
        yield log(f"\n--- WRAPPER EXECUTION LOG ---\n{wrapper_logs}")
    
    except Exception as e:
        yield log(f"❌ Training Failed: {e}")
        return

    yield log("\n🎉 Done! Ready for Training.")
    
def turkish_tokenizer_safety_check(model_file, case_rule="lowercase", tokenizer_type="indextts"):
    
    if model_file is None:
        return "❌ No tokenizer uploaded."
        
    if hasattr(model_file, "name"):
        model_file = model_file.name

    try:
        if tokenizer_type == "itts-tr":
            tokenizer = GenericSpiceTokenizer(vocab_file=model_file, normalizer=None, cjk=False)
        else:
            tokenizer = TextTokenizer(vocab_file=model_file, normalizer=None)
            
        sp = tokenizer.sp_model
        
        # Determine the uppercase flag based on user selection
        is_upper = (case_rule == "uppercase")
        
        # Instantiate Normalizer to match training conditions
        normalizer = MultilingualNormalizer(lang="tr", upper=is_upper)

        report = []
        ok = True

        report.append("## 🔍 TURKISH TOKENIZER SAFETY CHECK")
        report.append(f"### **📦 Vocab Size:** `{sp.get_piece_size()}`")
        report.append(f"### **🔲 UNK ID:** `{sp.unk_id()}` | **BOS ID:** `{sp.bos_id()}` | **EOS ID:** `{sp.eos_id()}` | **PAD ID:** `{sp.pad_id()}`")
        report.append(f"### **⚙️ Normalization Mode:** `{'UPPERCASE' if is_upper else 'LOWERCASE'}`\n")

        # 1. Define Combined Alphabet (Turkish + q, w, x)
        combined_alphabet = [
            "a", "b", "c", "ç", "d", "e", "f", "g", "ğ", "h", "ı", "i", 
            "j", "k", "l", "m", "n", "o", "ö", "p", "q", "r", "s", "ş", 
            "t", "u", "ü", "v", "w", "x", "y", "z"
        ]

        # --- CHECK 1: Alphabet Coverage Grid (Markdown Table) ---
        report.append("### 🔤 Alphabet Coverage Grid\n")
        
        upper_line = []
        upper_status_line = []
        lower_line = []
        lower_status_line = []
        missing_chars = []
        
        for char in combined_alphabet:
            # Proper Turkish uppercase for display
            if char == "i": upper_char = "İ"
            elif char == "ı": upper_char = "I"
            else: upper_char = char.upper()
            
            lower_char = char
            
            upper_line.append(upper_char)
            lower_line.append(lower_char)
            
            # Check UPPERCASE literal presence in vocab
            ids_up = sp.encode(upper_char, out_type=int)
            if sp.unk_id() in ids_up:
                upper_status_line.append("❌")
                # Only fail safety if we expect uppercase to be present
                if is_upper:
                    missing_chars.append(upper_char)
                    ok = False
            else:
                upper_status_line.append("✅")
                
            # Check LOWERCASE literal presence in vocab
            ids_low = sp.encode(lower_char, out_type=int)
            if sp.unk_id() in ids_low:
                lower_status_line.append("❌")
                # Only fail safety if we expect lowercase to be present
                if not is_upper:
                    if lower_char not in missing_chars:
                        missing_chars.append(lower_char)
                    ok = False
            else:
                lower_status_line.append("✅")

        # Format as Markdown Table
        report.append("| " + " | ".join(upper_line) + " |")
        report.append("|" + "|".join(["---"] * len(upper_line)) + "|")
        report.append("| " + " | ".join(upper_status_line) + " |")
        report.append("| " + " | ".join(lower_line) + " |")
        report.append("| " + " | ".join(lower_status_line) + " |\n")
        
        if missing_chars:
            report.append(f"⚠️ **Missing letters:** {', '.join(missing_chars)}")
        else:
            report.append("✅ **All alphabet letters are present.**")

        # --- CHECK 2: Extended Characters (Long Vowels) Grid with Prefixes ---
        report.append("\n### 🔤 Extended Characters (Long Vowels)\n")
        
        # Included prefixed variants (▁) to match your tr_long training list
        ext_pairs = [
            ("Â", "â"), ("▁Â", "▁â"), 
            ("Î", "î"), ("▁Î", "▁î"), 
            ("Û", "û"), ("▁Û", "▁û")
        ]
        
        ext_upper_line = []
        ext_upper_status = []
        ext_lower_line = []
        ext_lower_status = []
        missing_ext = []
        
        for up_char, low_char in ext_pairs:
            ext_upper_line.append(up_char)
            ext_lower_line.append(low_char)
            
            # Check literal presence of Uppercase in vocab
            ids_up = sp.encode(up_char, out_type=int)
            if sp.unk_id() in ids_up:
                ext_upper_status.append("❌")
                if is_upper:
                    missing_ext.append(up_char)
                    ok = False
            else:
                ext_upper_status.append("✅")
                
            # Check literal presence of Lowercase in vocab
            ids_low = sp.encode(low_char, out_type=int)
            if sp.unk_id() in ids_low:
                ext_lower_status.append("❌")
                if not is_upper:
                    if low_char not in missing_ext:
                        missing_ext.append(low_char)
                    ok = False
            else:
                ext_lower_status.append("✅")

        # Format as Markdown Table for perfect alignment
        report.append("| " + " | ".join(ext_upper_line) + " |")
        report.append("|" + "|".join(["---"] * len(ext_upper_line)) + "|")
        report.append("| " + " | ".join(ext_upper_status) + " |")
        report.append("| " + " | ".join(ext_lower_line) + " |")
        report.append("| " + " | ".join(ext_lower_status) + " |\n")

        if missing_ext:
            report.append(f"⚠️ **Missing extended letters:** {', '.join(missing_ext)}")
        else:
            report.append("✅ **All required extended letters are present.**")
            
        # --- CHECK 3: Punctuation Coverage Grid ---
        report.append("### 🔤 Punctuation Coverage\n")
        
        punctuations = [".", "▁.", ",", "▁,", "!", "▁!", "?", "▁?", ":", "▁:", ";", "▁;", "'", "▁'", "▁"]
        punct_line = []
        punct_status = []
        missing_punct = []
        
        for p in punctuations:
            punct_line.append(f"`{p}`")
            ids_p = sp.encode(p, out_type=int)
            if sp.unk_id() in ids_p:
                punct_status.append("❌")
                missing_punct.append(p)
                ok = False
            else:
                punct_status.append("✅")
                
        report.append("| " + " | ".join(punct_line) + " |")
        report.append("|" + "|".join(["---"] * len(punctuations)) + "|")
        report.append("| " + " | ".join(punct_status) + " |\n")

        if missing_punct:
            report.append(f"⚠️ **Missing punctuation:** {', '.join(missing_punct)}")
        else:
            report.append("✅ **All essential punctuation marks are present.**")
            
        # --- CHECK 4: Spaced Characters Grid ---
        report.append("### 🌌 Spaced Characters\n")
        
        spaced_upper_line = []
        spaced_upper_status = []
        spaced_lower_line = []
        spaced_lower_status = []
        missing_spaced = []
        
        for char in combined_alphabet:
            # Proper Turkish uppercase
            if char == "i": upper_char = "İ"
            elif char == "ı": upper_char = "I"
            else: upper_char = char.upper()
            
            lower_char = char
            
            # Prefixed versions
            prefixed_up = f"▁{upper_char}"
            prefixed_low = f"▁{lower_char}"
            
            # Use <small> tags to shrink the table contents to fit the UI
            spaced_upper_line.append(f"<small>{prefixed_up}</small>")
            spaced_lower_line.append(f"<small>{prefixed_low}</small>")
            
            # Check UPPERCASE literal presence in vocab
            ids_up = sp.encode(prefixed_up, out_type=int)
            if sp.unk_id() in ids_up:
                spaced_upper_status.append("<small>❌</small>")
                if is_upper:
                    missing_spaced.append(prefixed_up)
                    ok = False
            else:
                spaced_upper_status.append("<small>✅</small>")
                
            # Check LOWERCASE literal presence in vocab
            ids_low = sp.encode(prefixed_low, out_type=int)
            if sp.unk_id() in ids_low:
                spaced_lower_status.append("<small>❌</small>")
                if not is_upper:
                    if prefixed_low not in missing_spaced:
                        missing_spaced.append(prefixed_low)
                    ok = False
            else:
                spaced_lower_status.append("<small>✅</small>")

        # Format as Markdown Table
        report.append("| " + " | ".join(spaced_upper_line) + " |")
        report.append("|" + "|".join(["---"] * len(combined_alphabet)) + "|")
        report.append("| " + " | ".join(spaced_upper_status) + " |")
        report.append("| " + " | ".join(spaced_lower_line) + " |")
        report.append("| " + " | ".join(spaced_lower_status) + " |\n")
        
        if missing_spaced:
             report.append(f"⚠️ **Missing spaced letters:** {', '.join(missing_spaced)}")
        else:
             report.append("✅ **All spaced alphabet letters are present.**")

        # 3️⃣ Check for byte tokens
        byte_tokens = []
        for i in range(sp.get_piece_size()):
            if sp.is_byte(i):
                byte_tokens.append(sp.id_to_piece(i))

        if byte_tokens:
            report.append("\n### 🚨 BYTE TOKENS DETECTED (BAD FOR TTS)\n")
            report.extend([f"- ❌ `{b}`" for b in byte_tokens[:20]])
            report.append("- ... *(truncated)*")
            ok = False
        else:
            report.append("\n✅ **No byte tokens detected.**")

        # 4️⃣ Check sample Turkish words
        samples = ["Merhaba", "Dünya", "Işık", "Gölge", "Kâr", "Hâl", "X-Ray", "WhatsApp"]
        report.append("\n### 🧪 Sample Tokenization (Normalized)\n")
        for w in samples:
            w_norm = normalizer.normalize(w)
            pieces = sp.encode(w_norm, out_type=str)
            report.append(f"- `{w}` *(norm: {w_norm})* → `{pieces}`")
            if "<unk>" in pieces:
                ok = False

        # Final verdict
        report.append("### " + ("🎉 TOKENIZER PASSED SAFETY CHECK"
                                if ok else
                                "❌ TOKENIZER FAILED SAFETY CHECK"))

        return "\n".join(report)

    except Exception as e:
        return f"❌ Safety check failed:\n```text\n{e}\n```"
        
def test_tokenizer_inference(text, lang, model_state="trained", case_rule="lowercase", tokenizer_type="indextts"):

    if not text: return ""
    
    try:
        # Resolve Project Name
        p_name = core.project_name if core.project_name else "myproject"

        # 🔀 Switch between trained and merged models
        if model_state == "merged":
            model_filename = f"{p_name}_m_bpe.model"
        else:
            model_filename = f"{p_name}_bpe.model"
        
        model_path = os.path.join(core.tokenizer_directory(), model_filename)
        
        if not os.path.exists(model_path):
            return f"❌ Model file not found:\n{model_path}\n\nPlease train (Standard) or pack (Merged) the model first."
            
        # Determine the uppercase flag based on user UI selection
        is_upper = (case_rule == "uppercase")
        
        # 1. Normalize text
        normalizer = MultilingualNormalizer(lang=lang, wordify=True, abbreviations=True, upper=is_upper)
        normalized_text = normalizer.normalize(text)
        
        # 2. Load tokenizer based on user selection
        if tokenizer_type == "itts-tr":
            tokenizer = GenericSpiceTokenizer(vocab_file=model_path, normalizer=None, cjk=False)
        else:
            # Fallback to IndexTTS TextTokenizer
            tokenizer = TextTokenizer(vocab_file=model_path, normalizer=None)
        
        # 3. Tokenize using your custom methods
        ids = tokenizer.encode(normalized_text)
        pieces = tokenizer.tokenize(normalized_text)
        
        # 4. Format Report
        lines = []
        lines.append(f"📂 Model: {model_filename}")
        lines.append(f"⚙️ Tokenizer Type: {tokenizer_type}")
        lines.append(f"📝 Normalized Text: {normalized_text}")
        lines.append(f"🔢 Token Count: {len(ids)}")
        lines.append("-" * 30)
        
        # Pretty print tokens: [ID] Piece
        token_strs = [f"[{i}] {p}" for i, p in zip(ids, pieces)]
        lines.append(" | ".join(token_strs))
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Error: {e}"
    
def test_normalizer_ui(text, lang, case_rule, extract_flag):
    if not text:
        return ""
    try:
        is_upper = (case_rule == "uppercase")
        
        normalizer = MultilingualNormalizer(
            lang=lang, 
            wordify=True, 
            abbreviations=True, 
            upper=is_upper, 
            extract=extract_flag
        )
        return normalizer.normalize(text)
    except Exception as e:
        return f"Normalization Error: {e}"
        
def test_wordifier_ui(text, return_words_flag, lang, use_abbrev):
    if not text:
        return ""
    try:
        # Initialize the router with the UI parameters
        wordifier_router = MultilingualWordifier(text, language_code=lang, abbreviations=use_abbrev)
        
        # 1. Word List Mode
        if return_words_flag:
            words_list = wordifier_router.get_words()
            return ", ".join(words_list)
            
        # 2. Full Block Mode
        if hasattr(wordifier_router.processor, 'normalized_text'):
            return wordifier_router.processor.normalized_text
        elif hasattr(wordifier_router.processor, 'get_text'):
            return wordifier_router.processor.get_text()
        else:
            return text
            
    except Exception as e:
        return f"Wordifier Error: {e}"
        
def process_design_vocab(
    uploaded_model, 
    strip_cjk, 
    strip_en_tokens, 
    preserve_source_punct,
    apply_injections, 
    convert_lowercase,
    target_pres_letters,
    target_pres_punct,
    target_token_case,
    target_letter_case
):
        
    logs = []
    model_path_source = os.path.join(core.path_base, "indextts", "checkpoints", "bpe.model")
    
    logs.append(f"📥 Pulling Source BPE model from: {model_path_source}")
    
    try:
        if not os.path.exists(model_path_source):
            return "", "", "", "❌ Error: bpe.model not found. Cannot design vocabulary."
            
        logs.append("✅ bpe.model located. Unpacking model...")
         
        sp = spm.SentencePieceProcessor()
        sp.load(model_path_source)
    
        data = []
        
        for i in range(sp.get_piece_size()):
            data.append({
                "id": i,
                "piece": sp.id_to_piece(i),
                "score": sp.get_score(i),
                "is_control": sp.is_control(i),
                "is_unknown": sp.is_unknown(i),
                "is_unused": sp.is_unused(i),
                "is_byte": sp.is_byte(i),
            })

        # --- FILTERING SETUP ---
        target_tags = ["[ZH]", "[EN]", "[JA]", "[KO]"]
        
        u2581_char = "▁"
        allowed_pattern = f'^[a-zA-ZçğıİöşüÇĞÖŞÜâîûÂÎÛ\\s\\W{u2581_char}]+$'
        allowed_chars_re = re.compile(allowed_pattern)
        
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        punct_pattern = re.compile(r'^[\W_]+$')

        cleaned_data = []
        removed_pieces = []
        removed_count = 0
        invalid_chars_count = 0
        en_removed_count = 0
        punct_removed_count = 0
        lowered_count = 0
        
        # 🛡️ CAPTURE BASE EN TOKENS: Trap them so the Target Model cannot resurrect them
        base_en_tokens = set()
        for entry in data:
            p_id = entry.get("id", -1)
            if 10204 <= p_id <= 11998:
                p_piece = entry.get("piece", "")
                core_chars = p_piece.replace("▁", "")
                if len(core_chars) > 1:
                    base_en_tokens.add(p_piece)
        
        for entry in data:
            piece = entry.get("piece", "")
            piece_id = entry.get("id", -1)
            
            # Rule 1: Strip language markers
            if any(tag in piece for tag in target_tags):
                removed_pieces.append(piece)
                removed_count += 1
                continue
          
            # Rule 2: Preserve special tokens
            if piece in ["<s>", "</s>", "<unk>"]:
                cleaned_data.append(entry)
                continue
                
            # Rule 3: Preserve special space characters
            if piece in [" ", "▁"]:
                cleaned_data.append(entry)
                continue
                
            # Rule 4: Source Punctuation Filtration
            core_chars = piece.replace("▁", "")
            if core_chars and bool(punct_pattern.match(core_chars)):
                if not preserve_source_punct:
                    punct_removed_count += 1
                    continue
                
            # Rule 5: Dynamic EN Token Handling (Base EN block spans 10204-11998)
            if 10204 <= piece_id <= 11998:
                if strip_en_tokens and piece in base_en_tokens:
                    en_removed_count += 1
                    continue
                else:
                    cleaned_data.append(entry)
                    continue

            # Rule 6: Dynamic CJK Handling
            if cjk_pattern.search(piece):
                if strip_cjk:
                    invalid_chars_count += 1
                    continue
                else:
                    cleaned_data.append(entry)
                    continue

            # Rule 7: Remove if piece contains characters NOT in the allowed list
            if not allowed_chars_re.match(piece):
                invalid_chars_count += 1
                continue
                
            # Rule 8: Catch-all for remaining valid tokens
            if piece_id != 11999:
                cleaned_data.append(entry)

        # Rule 9: Inject Custom Language Markers at IDs 3 and 4
        cleaned_data = [e for e in cleaned_data if e.get("id") not in [3, 4]]
        
        cleaned_data.extend([
            {
                "id": 3, "piece": "▁[TR]", "score": 0.0,
                "is_control": False, "is_unknown": False, "is_unused": False, "is_byte": False
            },
            {
                "id": 4, "piece": "▁[EN]", "score": 0.0,
                "is_control": False, "is_unknown": False, "is_unused": False, "is_byte": False
            }
        ])

        # Rule 10: Enforce ID 11999 as <pad>
        cleaned_data.append({
            "id": 11999, "piece": "<pad>", "score": 0.0,
            "is_control": True, "is_unknown": False, "is_unused": False, "is_byte": False
        })
        
        # Rule 11: Convert Remaining EN Tokens to Lowercase
        if convert_lowercase:
            for entry in cleaned_data:
                if 10204 <= entry.get("id", -1) <= 11998:
                    orig_piece = entry.get("piece", "")
                    lowered_piece = orig_piece.lower()
                    if orig_piece != lowered_piece:
                        entry["piece"] = lowered_piece
                        lowered_count += 1
        
        # Rule 12: Dynamic Injections based on UI Master Flag
        required_injections = []
        
        if apply_injections:
            inject_puncts_normal = [".", ",", "!", "?", ":", ";", "(", ")", "'", "-", "..."]
            inject_puncts_spaced = ["▁.", "▁,", "▁!", "▁?", "▁:", "▁;", "▁(", "▁)", "▁'", "▁-", "▁..."]
            
            inject_turkish_special = ["Ç", "▁Ç", "Ğ", "▁Ğ", "İ", "▁İ", "Ö", "▁Ö", "Ş", "▁Ş", "Ü", "▁Ü"]
            inject_turkish_long_vowels = ["Â", "Ê", "Î", "Ô", "Û"]
            
            inject_turkic_characters = ["Ə", "Ñ", "Ŋ", "Ä", "Ë", "Ž", "Ň", "Ý", "Ū", "Ă", "Ĕ", "Ś", "Ÿ"]
            inject_turkic_characters_spaced = ["▁Ə", "▁Ñ", "▁Ŋ", "▁Ä", "▁Ë", "▁Ž", "▁Ň", "▁Ý", "▁Ū", "▁Ă", "▁Ĕ", "▁Ś", "▁Ÿ"]
            
            inject_eng_upper = list(string.ascii_uppercase)
            inject_eng_upper_spaced = [f"▁{c}" for c in string.ascii_uppercase]
            
            required_injections = (inject_puncts_normal + inject_puncts_spaced + 
                                   inject_turkish_special + inject_turkish_long_vowels + 
                                   inject_turkic_characters + inject_turkic_characters_spaced +
                                   inject_eng_upper + inject_eng_upper_spaced)
        
        existing_pieces = {e.get("piece") for e in cleaned_data}
        existing_ids = {e.get("id") for e in cleaned_data}
        
        missing_injections = [p for p in required_injections if p not in existing_pieces]
        injected_items_log = []
        
        if missing_injections:
            available_id = 10203  # Start searching downwards from just below the English block
            for item in missing_injections:
                while available_id in existing_ids:
                    available_id -= 1
                
                cleaned_data.append({
                    "id": available_id,
                    "piece": item,
                    "score": 0.0,
                    "is_control": False,
                    "is_unknown": False,
                    "is_unused": False,
                    "is_byte": False
                })
                existing_ids.add(available_id)
                injected_items_log.append(f"{item} (ID: {available_id})")
                available_id -= 1

        # Sort sequentially to maintain model integrity
        cleaned_data = sorted(cleaned_data, key=lambda x: x["id"])
        
        # Output Logging Matrix
        logs.append(f"1. Language markers removed: {removed_count} (Tokens: {', '.join(removed_pieces)})")
        logs.append("2. <s>, </s> and <unk> preserved (with ids 0,1,2)")
        
        if preserve_source_punct:
            logs.append("3. Source punctuation preserved.")
        else:
            logs.append(f"3. Source punctuation stripped ({punct_removed_count} base punctuation tokens removed).")
            
        logs.append("4. Special space character is preserved (with id 10201)")
        
        if strip_en_tokens:
            logs.append(f"5. English sub-words stripped: {en_removed_count} tokens removed (Single chars preserved).")
        else:
            logs.append("5. All english pieces are preserved (10204 - 11998).")
            
        if strip_cjk:
            logs.append(f"6. CJK and invalid characters removed via Regex: {invalid_chars_count}")
        else:
            logs.append(f"6. Invalid characters removed via Regex: {invalid_chars_count} (CJK preserved)")
            
        logs.append("7. Injected ▁[TR] (ID: 3) and ▁[EN] (ID: 4)")
        logs.append("8. Last piece that is 11999 is set as <pad>")
        
        if convert_lowercase:
            logs.append(f"9. Converted {lowered_count} remaining English tokens to lowercase.")
        else:
            logs.append("9. English token casing preserved.")
            
        if missing_injections:
            logs.append(f"10. Applied {len(missing_injections)} Required Injections: {', '.join(injected_items_log)}")
        elif apply_injections:
            logs.append("10. All Required Injections were already present in the source.")
        else:
            logs.append("10. Required Injections bypassed by user.")
            
        # 🟢 SNAPSHOT 1: The Processed Source Data
        source_json_output = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
        
        # ==========================================
        # Rule 13: Merge Target Model to Fill Gaps
        # ==========================================
        p_name = core.project_name if core.project_name else "myproject"
        
        if uploaded_model is not None:
            model_path_target = uploaded_model.name
            logs.append(f"📥 Pulling Target BPE model from UPLOAD: {os.path.basename(model_path_target)}")
        else:
            model_path_target = os.path.join(core.tokenizer_directory(), f"{p_name}_bpe.model")
            logs.append(f"📥 Pulling Target BPE model from PROJECT: {model_path_target}")
            
        target_merged_items = []
            
        if not os.path.exists(model_path_target):
            logs.append("⚠️ Target bpe.model not found. Skipping merge phase.")
        else:
            logs.append("✅ Target bpe.model located. Merging into gaps...")
            sp_target = spm.SentencePieceProcessor()
            sp_target.load(model_path_target)
            
            existing_pieces = {e.get("piece") for e in cleaned_data}
            existing_ids = {e.get("id") for e in cleaned_data}
            
            max_id = 11999
            available_gaps = [i for i in range(max_id) if i not in existing_ids]
            
            merged_count = 0
            skipped_count = 0
            
            for i in range(sp_target.get_piece_size()):
                raw_piece = sp_target.id_to_piece(i)
                core_chars = raw_piece.replace("▁", "")
                
                if raw_piece in ["<s>", "</s>", "<unk>", "<pad>"]:
                    skipped_count += 1
                    continue
                    
                if strip_en_tokens and raw_piece in base_en_tokens:
                    skipped_count += 1
                    continue
                    
                if strip_cjk and cjk_pattern.search(raw_piece):
                    skipped_count += 1
                    continue
                
                is_tag = core_chars.startswith("[") and core_chars.endswith("]")
                is_punct = bool(punct_pattern.match(core_chars)) if core_chars else False
                is_single_letter = len(core_chars) == 1 and core_chars.isalpha()
                
                # Target Preservation Filters
                if is_punct and not target_pres_punct:
                    skipped_count += 1
                    continue
                    
                if is_single_letter and not target_pres_letters:
                    skipped_count += 1
                    continue
                    
                # Target Casing Rules
                if is_tag:
                    # Emotion/Style/Language tags bypass casing logic and remain EXACTLY the same
                    t_piece = raw_piece
                elif is_single_letter:
                    # Apply standalone letter casing rule
                    t_piece = raw_piece.lower() if target_letter_case == "lowercase" else raw_piece.upper()
                elif core_chars == "":
                    # It is purely the space marker "▁" or completely empty
                    t_piece = raw_piece 
                else:
                    # It is a multi-character sub-word (e.g., "▁hello", "ing")
                    t_piece = raw_piece.lower() if target_token_case == "lowercase" else raw_piece.upper()
                    
                # Final Merge Logic
                if t_piece not in existing_pieces:
                    if available_gaps:
                        new_id = available_gaps.pop(0)
                        
                        new_item = {
                            "id": new_id,
                            "piece": t_piece,
                            "score": sp_target.get_score(i),
                            "is_control": sp_target.is_control(i),
                            "is_unknown": sp_target.is_unknown(i),
                            "is_unused": sp_target.is_unused(i),
                            "is_byte": sp_target.is_byte(i),
                        }
                        
                        cleaned_data.append(new_item)
                        target_merged_items.append(new_item) # Track it!
                        
                        existing_pieces.add(t_piece)
                        merged_count += 1
                    else:
                        skipped_count += 1 
                        
            logs.append(f"11. Merged {merged_count} new pieces from Target. Skipped {skipped_count} (filtered, duplicates, or no capacity).")

        # Final Sort to maintain absolute model integrity
        cleaned_data = sorted(cleaned_data, key=lambda x: x["id"])

        # ==========================================
        # Export Merged Model & Vocab
        # ==========================================
        out_dir = core.tokenizer_directory()
        os.makedirs(out_dir, exist_ok=True)
        
        ds_name = f"{p_name}_m_bpe"
        vocab_path = os.path.join(out_dir, f"{ds_name}.vocab")
        bpe_path = os.path.join(out_dir, f"{ds_name}.model")
        
        try:
            with open(vocab_path, "w", encoding="utf-8") as vf:
                for item in cleaned_data:
                    piece = item.get("piece", "")
                    score = item.get("score", 0.0)
                    vf.write(f"{piece}\t{score}\n")
            logs.append(f"✅ Saved merged vocab to: {vocab_path}")
        except Exception as ve:
            logs.append(f"⚠️ Warning: Failed to save .vocab file: {ve}")
            
        try:
            converter = JsonToModelConverter(ds_name, out_dir)
            convert_msg = converter.convert(cleaned_data)
            logs.append(f"✅ Saved merged BPE model to: {bpe_path}")
        except Exception as ce:
            logs.append(f"❌ Protobuf Packing Error: {str(ce)}")

        # 🟢 SNAPSHOT 2 & 3: Target Data and Final Data
        target_json_output = json.dumps(target_merged_items, ensure_ascii=False, indent=2)
        final_json_output = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
        
        return source_json_output, target_json_output, final_json_output, "\n".join(logs)
        
    except Exception as e:
        return "", "", "", f"❌ Execution Failed: {str(e)}"
              
def open_tokenizer_folder():
    folder_path = core.tokenizer_directory()

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."
        
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    lang_options = core.language_list()

    # =========================
    # Phase 2: Train Tokenizer
    # =========================
    
    with gr.Blocks() as demo:
        gr.Markdown(_("TOKENIZER_HEADER"))
        gr.Markdown(_("TOKENIZER_DESC"))

        with gr.Group():
            
            with gr.Row():
                with gr.Column():               
                    # --- DATASET SELECTOR ---
                    dataset_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_DATASET"),
                        choices=list_datasets("tr"),
                        value=None,
                        interactive=True
                    )
                with gr.Column(): 
                    # --- LANGUAGE SELECTOR ---
                    lang_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_LANG"),
                        choices=lang_options,
                        value="tr",
                        interactive=True
                    )
                    
            with gr.Row():    
                refresh_btn = gr.Button(_("TOKENIZER_BTN_REFRESH"))
                    
            with gr.Row():
                include_corpus_chk = gr.Checkbox(
                    label=_("TOKENIZER_CHK_CORPUS"),
                    value=False,
                    info=_("TOKENIZER_INFO_CORPUS")
                )
                use_only_corpus_chk = gr.Checkbox(
                    label=_("TOKENIZER_CHK_ONLY_CORPUS"),
                    value=False,
                    info=_("TOKENIZER_INFO_ONLY_CORPUS")
                )            
                inject_words_chk = gr.Checkbox(
                    label=_("TOKENIZER_CHK_INJECT"), 
                    value=False,
                    info=_("TOKENIZER_INFO_INJECT")
                )
                multiplier_slider = gr.Slider(
                    label=_("TOKENIZER_SLIDER_MULTIPLIER"), 
                    minimum=1, maximum=100, step=1, value=10
                )               
                deduplicate_chk = gr.Checkbox(
                    label=_("TOKENIZER_CHK_DEDUP"), 
                    value=False, 
                    info=_("TOKENIZER_INFO_DEDUP")
                )
                
            with gr.Row():
                with gr.Column():                
                    data_coverage_slider = gr.Slider(
                        label=_("TOKENIZER_SLIDER_META_COV"),
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=100,
                        info=_("TOKENIZER_INFO_META_COV")
                    )
                with gr.Column():                
                    char_coverage_slider = gr.Slider(
                        label=_("TOKENIZER_SLIDER_CHAR_COV"),
                        minimum=0.99,
                        maximum=1.0,
                        step=0.0001,
                        value=1.0,
                        info=_("TOKENIZER_INFO_CHAR_COV")
                    )
                    
            with gr.Row():   
                vocab_slider = gr.Slider(
                    minimum=2000,
                    maximum=30000,
                    value=12000,
                    step=1000,
                    label=_("TOKENIZER_SLIDER_VOCAB")
                )
                
            with gr.Group(): 
                gr.Markdown(_("TOKENIZER_HEADER_NORM"))                
                norm_rule_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_NORM"),
                    show_label=False,
                    choices=["nmt_nfkc", "nmt_nfkc_cf", "nfkc", "nfkc_cf", "none"],
                    value="none",
                    scale=4,
                    interactive=True         
                )
                
            with gr.Group(): 
                gr.Markdown(_("TOKENIZER_HEADER_CASE"))                
                case_rule_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_CASE"),
                    show_label=False,
                    choices=["lowercase", "uppercase"],
                    value="lowercase",
                    scale=4,
                    interactive=True         
                )
                
            with gr.Group():
                gr.Markdown(_("TOKENIZER_HEADER_EMOTIONS"))
                style_chk = gr.Checkbox(label=_("TOKENIZER_CHK_STYLE"), value=False)
                emotion_chk = gr.Checkbox(label=_("TOKENIZER_CHK_EMOTION"), value=False)
                
            with gr.Group():
                gr.Markdown(_("TOKENIZER_HEADER_TR_SPECIAL"))
                tr_spec_chk = gr.Checkbox(label=_("TOKENIZER_CHK_TR_SPEC"), value=False)
                tr_seng_chk = gr.Checkbox(label=_("TOKENIZER_CHK_SENG"), value=False)
                tr_turk_chk = gr.Checkbox(label=_("TOKENIZER_CHK_TURK"), value=False)
                tr_long_chk = gr.Checkbox(label=_("TOKENIZER_CHK_LONG"), value=False)
                tr_punc_chk = gr.Checkbox(label=_("TOKENIZER_CHK_PUNC"), value=False)
                
            with gr.Group():
                gr.Markdown(_("TOKENIZER_HEADER_SPECIAL"))
                special_input = gr.Textbox(
                    label=_("TOKENIZER_LABEL_SPECIAL"),                   
                    value="",
                    placeholder="€ | £ | ¥ | ₺ | ₿ | ± | × | ÷ | ≠ | ≤ | ≥ | ∞ | √ | ∑ | ∏ | π | ∆ | ∂ | µ | Ω"
                )
            
            with gr.Group():
                gr.Markdown(_("TOKENIZER_HEADER_ADVANCED")) 
                tok_sentence_size = gr.Number(
                    label=_("TOKENIZER_LABEL_ADV_SENT_SIZE"), 
                    value=0, 
                    precision=0, 
                    info=_("TOKENIZER_INFO_ADV_SENT_SIZE")
                )
                tok_train_ext = gr.Checkbox(
                    label=_("TOKENIZER_CHK_ADV_TRAIN_EXT"), 
                    value=False, 
                    info=_("TOKENIZER_INFO_ADV_TRAIN_EXT")
                )                
                tok_shuffle = gr.Checkbox(
                    label=_("TOKENIZER_CHK_ADV_SHUFFLE"), 
                    value=False, 
                    info=_("TOKENIZER_INFO_ADV_SHUFFLE")
                )
                tok_hard_vocab = gr.Checkbox(
                    label=_("TOKENIZER_CHK_ADV_HARD_VOCAB"), 
                    value=False, 
                    info=_("TOKENIZER_INFO_ADV_HARD_VOCAB")
                )                  
                      
            train_btn = gr.Button(_("TOKENIZER_BTN_TRAIN"), variant="primary")

            with gr.Row():
                log_box = gr.Textbox(label=_("TOKENIZER_LABEL_LOGS"), lines=5, max_lines=15)
                
            with gr.Row():    
                tok_folder_btn = gr.Button(_("COMMON_FOLDER_OPEN"))

        # --- EVENTS ---
        
        lang_dd.change(
            fn=lambda l: gr.Dropdown(choices=list_datasets(l), value=None),
            inputs=[lang_dd],
            outputs=[dataset_dd]
        )
                
        dataset_dd.change(
            fn=on_dataset_select,
            inputs=[lang_dd, dataset_dd],
            outputs=[vocab_slider, log_box]
        )
        
        inputs = [
            lang_dd, 
            dataset_dd, 
            vocab_slider,
            data_coverage_slider,            
            char_coverage_slider, 
            include_corpus_chk, 
            special_input,
            style_chk,             
            emotion_chk,           
            tr_spec_chk,           
            tr_seng_chk,
            tr_turk_chk,
            tr_long_chk,
            tr_punc_chk,
            use_only_corpus_chk,
            norm_rule_dd,
            case_rule_dd,
            inject_words_chk,
            multiplier_slider,
            deduplicate_chk,
            tok_sentence_size,
            tok_train_ext,
            tok_shuffle,
            tok_hard_vocab
        ]
       
        # Train Click
        train_btn.click(
            fn=train_tokenizer_ui,
            inputs=inputs,
            outputs=[log_box]
        )
        
        # Refresh Click
        refresh_btn.click(
            fn=lambda l: gr.Dropdown(choices=list_datasets(l)),
            inputs=[lang_dd],
            outputs=[dataset_dd]
        )
        
        # Folder Click
        tok_folder_btn.click(
            fn=open_tokenizer_folder,
            inputs=[],
            outputs=[]
        )
                  
        gr.HTML("<div style='height:10px'></div>")

        # ==========
        # UTILITIES
        # ==========
        with gr.Group():
            gr.Markdown(_("TOKENIZER_HEADER_UTILS"), elem_classes="wui-markdown")

        # =========================
        # Turkish Tokenizer Safety Check
        # =========================
        with gr.Accordion(_("TOKENIZER_ACC_SAFETY"), open=False, elem_classes="wui-accordion"):            
            gr.Markdown(_("TOKENIZER_DESC_SAFETY"))
        
            with gr.Row():
                safety_file = gr.File(
                    label=_("TOKENIZER_LABEL_UPLOAD"),
                    file_types=[".model"]
                )
            with gr.Group(): 
                gr.Markdown(_("TOKENIZER_HEADER_TTYPE"))
                safety_tok_type_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_TTYPE"),
                    choices=["indextts", "itts-tr"],
                    value="indextts",
                    scale=4,
                    interactive=True
                )   
            with gr.Group(visible=False) as safety_case_group: 
                gr.Markdown(_("TOKENIZER_HEADER_SCASE"))                
                safety_case_rule_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_SCASE"),
                    show_label=False,
                    choices=["lowercase", "uppercase"],
                    value="lowercase",
                    scale=4,
                    interactive=True         
                )           
            with gr.Row():
                safety_btn = gr.Button(_("TOKENIZER_BTN_SAFETY"), variant="primary")
            
            safety_report = gr.Markdown(
                value=_("TOKENIZER_LABEL_REPORT"),
                elem_classes="wui-markdown"
            )
            
            safety_btn.click(
                fn=turkish_tokenizer_safety_check,
                inputs=[safety_file, safety_case_rule_dd, safety_tok_type_dd],
                outputs=safety_report
            )
            
            safety_tok_type_dd.change(
                fn=lambda t: gr.update(visible=(t == "itts-tr")),
                inputs=[safety_tok_type_dd],
                outputs=[safety_case_group]
            )            

        # =================
        # Tokenizer Tester
        # =================
        with gr.Accordion(_("TOKENIZER_ACC_TESTER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("TOKENIZER_DESC_TESTER"))
            
            with gr.Row():
                with gr.Column(scale=3):
                    tok_check_input = gr.Textbox(
                        label=_("TOKENIZER_LABEL_INPUT"), 
                        lines=2, 
                        value="Merhaba Dünya! Bu bir test.",
                        placeholder=_("TOKENIZER_PLACEHOLDER_TEST_INPUT")
                    )
                with gr.Column(scale=1):
                    tok_lang_dd = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_LANG"),
                        choices=lang_options,
                        value="tr"
                    )
            with gr.Group(): 
                gr.Markdown(_("TOKENIZER_HEADER_TMTYPE"))
                tok_mtype_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_TMTYPE"),
                    choices=["trained", "merged"],
                    value="trained",
                    scale=4,
                    interactive=True
                )                   
            with gr.Group(): 
                gr.Markdown(_("TOKENIZER_HEADER_TTYPE"))
                tok_type_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_TTYPE"),
                    choices=["indextts", "itts-tr"],
                    value="indextts",
                    scale=4,
                    interactive=True
                )
            with gr.Group(visible=False) as tok_case_group: 
                gr.Markdown(_("TOKENIZER_HEADER_TCASE"))                
                tok_case_rule_dd = gr.Dropdown(
                    label=_("TOKENIZER_LABEL_TCASE"),
                    show_label=False,
                    choices=["lowercase", "uppercase"],
                    value="lowercase",
                    scale=4,
                    interactive=True         
                )
            with gr.Row():    
                tok_check_btn = gr.Button(_("TOKENIZER_BTN_RUN_TEST"), variant="primary")
            
            tok_check_output = gr.Textbox(
                label=_("TOKENIZER_LABEL_TOK_OUT"), 
                lines=4, 
                interactive=False
            )
            
            tok_check_btn.click(
                fn=test_tokenizer_inference,
                inputs=[
                    tok_check_input, 
                    tok_lang_dd, 
                    tok_mtype_dd,       # (Trained/Merged)
                    tok_case_rule_dd,  # (Lowercase/Uppercase)
                    tok_type_dd       # (indextts/itts-tr)
                ],
                outputs=tok_check_output
            )
            
            tok_type_dd.change(
                fn=lambda t: gr.update(visible=(t == "itts-tr")),
                inputs=[tok_type_dd],
                outputs=[tok_case_group]
            )
                  
        # ===========================
        # Wordifier Tester
        # ===========================        
        with gr.Accordion(_("TOKENIZER_ACC_WORDIFIER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("TOKENIZER_DESC_WORDIFIER"))
            
            with gr.Row():
                with gr.Column(scale=3):
                    wordify_input = gr.Textbox(
                        label=_("TOKENIZER_LABEL_INPUT"), 
                        lines=12, 
                        value="19.05.1919'da 2.500 kişi Samsun'a çıktı.",
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        wordify_lang = gr.Dropdown(
                            label=_("TOKENIZER_LABEL_LANG"),
                            choices=lang_options,
                            value="tr"
                        )
                    with gr.Row():
                        gr.Column(scale=1)
                        with gr.Column(scale=1, min_width=150):
                            wordify_abbrev = gr.Checkbox(
                                label=_("TOKENIZER_CHK_ABBREV"), 
                                value=True
                            )
                        gr.Column(scale=1)
                    with gr.Row():
                        wordify_mode = gr.Radio(
                            choices=[_("TOKENIZER_CHOICE_FULL_BLOCK"), _("TOKENIZER_CHOICE_WORD_LIST")],
                            value=_("TOKENIZER_CHOICE_FULL_BLOCK"),
                            label=_("TOKENIZER_LABEL_RETURN_FMT"),
                            type="index"
                        )
            with gr.Row():
                wordify_output = gr.Textbox(
                    label=_("TOKENIZER_LABEL_RESULT"), 
                    lines=4, 
                    interactive=False
                )
            with gr.Row():
                wordify_btn = gr.Button(_("TOKENIZER_BTN_WORDIFY"), variant="primary")
            
            wordify_btn.click(
                fn=test_wordifier_ui,
                inputs=[wordify_input, wordify_mode, wordify_lang, wordify_abbrev],
                outputs=wordify_output
            )
            
        # =========================
        # Normalizer Tester
        # =========================
        with gr.Accordion(_("TOKENIZER_ACC_NORM"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("TOKENIZER_DESC_NORM"))
            
            with gr.Row():
                with gr.Column(scale=3):
                    norm_input = gr.Textbox(
                        label=_("TOKENIZER_LABEL_INPUT"), 
                        lines=12, 
                        value="""TARİH: 19.05.1919! (Saat: 08:00)... İSTANBUL'dan SAMSUN'a uzanan o "eşsiz" yolculuk; [happy] %14,5 ihtimalle değil, %100 İNANÇLA başladı. Rakamlar(3,14 veya 1.250.000 TL) önemsizdi: Tek hedef "BAĞIMSIZLIK"tı. O gün;herkes(ve her şey) değişti. Sonuç:CUMHURİYET! [sad] Ama bedeli ağırdı... (v2.0) sürümü yükleniyor: HAZIR MISIN?""",
                        placeholder=_("TOKENIZER_PLACEHOLDER_NORM_INPUT")
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        norm_lang = gr.Dropdown(
                            label=_("TOKENIZER_LABEL_LANG"),
                            choices=lang_options,
                            value="tr"
                        )
                    with gr.Row():
                        norm_case = gr.Dropdown(
                            label=_("TOKENIZER_LABEL_CASE"),
                            choices=["lowercase", "uppercase"],
                            value="lowercase"
                        )
                    with gr.Row():
                        gr.Column(scale=1)
                        with gr.Column(scale=1, min_width=150):
                            norm_extract = gr.Checkbox(
                                label=_("CONFIG_CHK_EXTRACT"),
                                value=False
                            )
                        gr.Column(scale=1)
                        
            with gr.Row():

                norm_output = gr.Textbox(
                    label=_("TOKENIZER_LABEL_NORM_OUT"), 
                    lines=3, 
                    interactive=False
                )
            
            norm_btn = gr.Button(_("TOKENIZER_BTN_NORM"), variant="primary")
            
            norm_btn.click(
                fn=test_normalizer_ui,
                inputs=[norm_input, norm_lang, norm_case, norm_extract],
                outputs=norm_output
            )
            
        # =====================
        # Custom Vocab Designer
        # =====================          
        with gr.Accordion(_("TOKENIZER_ACC_DESIGN"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("TOKENIZER_DESC_DESIGN"))
            
            with gr.Row():
                with gr.Column(scale=2):
                    design_model_upload = gr.File(
                        label=_("TOKENIZER_LABEL_DESIGN_UPLOAD"), 
                        file_types=[".model"]
                    )
                with gr.Column(scale=1):
                    gr.Markdown(_("TOKENIZER_HEADER_DESIGN_SOURCE"))
                    design_source_strip_cjk = gr.Checkbox(label=_("TOKENIZER_CHK_STRIP_CJK"), value=True)
                    design_source_strip_en = gr.Checkbox(label=_("TOKENIZER_CHK_STRIP_ENG"), value=True)
                    design_source_pres_punct = gr.Checkbox(label=_("TOKENIZER_CHK_PRES_SRC_PUNCT"), value=True) 
                    design_source_apply_injections = gr.Checkbox(label=_("TOKENIZER_CHK_APPLY_INJ"), value=True)
                    design_source_conv_lower = gr.Checkbox(label=_("TOKENIZER_CHK_CONV_LOWER"), value=False)
                with gr.Column(scale=1):
                    gr.Markdown(_("TOKENIZER_HEADER_DESIGN_TARGET"))
                    design_target_pres_letters = gr.Checkbox(label=_("TOKENIZER_CHK_PRES_TGT_LETTERS"), value=True)
                    design_target_pres_punct = gr.Checkbox(label=_("TOKENIZER_CHK_PRES_TGT_PUNCT"), value=True)
                    design_target_token_case = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_TGT_TOK_CASE"),
                        choices=["lowercase", "uppercase"],
                        value="lowercase"
                    )
                    design_target_letter_case = gr.Dropdown(
                        label=_("TOKENIZER_LABEL_TGT_LET_CASE"),
                        choices=["lowercase", "uppercase"],
                        value="uppercase"
                    )
                    
            with gr.Row():
                design_vocab_btn = gr.Button(_("TOKENIZER_BTN_DESIGN"), variant="primary")
                
            design_log_output = gr.Textbox(label=_("TOKENIZER_LABEL_DESIGN_LOGS"), interactive=False, lines=8)
            
            # --- 3-COLUMN SIDE-BY-SIDE TEXT BLOCKS ---
            with gr.Row():
                design_source_output = gr.Textbox(
                    label=_("TOKENIZER_LABEL_OUT_SOURCE"), 
                    interactive=False, 
                    lines=15
                )
                design_target_output = gr.Textbox(
                    label=_("TOKENIZER_LABEL_OUT_TARGET"), 
                    interactive=False, 
                    lines=15
                )
                design_final_output = gr.Textbox(
                    label=_("TOKENIZER_LABEL_OUT_FINAL"), 
                    interactive=False, 
                    lines=15
                )
            
            # --- EVENT BINDING ---
            design_vocab_btn.click(
                fn=process_design_vocab,
                inputs=[
                    design_model_upload, 
                    # Source Flags
                    design_source_strip_cjk, 
                    design_source_strip_en, 
                    design_source_pres_punct,
                    design_source_apply_injections, 
                    design_source_conv_lower,
                    # Target Flags
                    design_target_pres_letters,
                    design_target_pres_punct,
                    design_target_token_case,
                    design_target_letter_case
                ],
                outputs=[
                    design_source_output, 
                    design_target_output, 
                    design_final_output, 
                    design_log_output
                ]
            )
                   
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("tokenizer"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")