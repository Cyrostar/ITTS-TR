import gradio as gr
import os
import sys
import shutil
import subprocess
import re
import string
import datetime
import concurrent.futures
import multiprocessing
from collections import defaultdict
from pypdf import PdfReader
import sentencepiece as spm

import gc
import torch
import torchaudio
import whisper
import soundfile as sf
from pyannote.audio import Pipeline
from demucs.pretrained import get_model
from demucs.apply import apply_model

from core import core
from core.core import _
from core.database import SQLiteManager
from core.spice import SentencePieceTrainerWrapper
from core.normalizer import MultilingualNormalizer, MultilingualWordifier
from core.syllabify import TurkishSyllabifier

# --- HELPER FUNCTIONS ---

def refresh_lists():
    """Manual refresh handler returning formatted strings."""
    return list_files_formatted("pdf", ".pdf"), list_files_formatted("txt", ".txt")
    
def list_files_formatted(subfolder, extension):
    """Lists files in a specific corpus subfolder with icons, matching models.py format."""
    target_dir = os.path.join(core.corpus_directory(), subfolder)
    
    if not os.path.exists(target_dir):
        return "📂 Directory not created yet."
    
    try:
        items = [f for f in os.listdir(target_dir) if f.lower().endswith(extension)]
        if not items:
            return f"📂 No {extension.upper()} files found."
            
        formatted_list = [f"📄 {item}" for item in sorted(items)]
        return "\n".join(formatted_list)
    except Exception as e:
        return f"Error: {str(e)}"

# --- DATABASE & TOKENIZER MANAGER ---

def get_db():
    db_dir = core.corpus_directory()
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "corpus.db")
    return SQLiteManager(db_path)

def init_db():
    """Ensures the db folder exists and initializes relational SQLite tables."""
    db = get_db()
    
    # 1. Parent Table: Tracks processed PDFs & their source language
    db.create_table(
        "processed_pdfs", 
        "id INTEGER PRIMARY KEY AUTOINCREMENT, pdf_name TEXT UNIQUE, lang TEXT"
    )
    
    # 2. Child Table: Original schema + the new flag + lang marker
    db.create_table(
        "pdf_chunks", 
        "pdf_id INTEGER, page_number INTEGER, lang TEXT, text TEXT, is_normalized INTEGER DEFAULT 0, FOREIGN KEY(pdf_id) REFERENCES processed_pdfs(id)"
    )
        
    # 3. Normalized Table: Tracks chunks by unique Language AND Text pair
    db.create_table(
        "normalized_chunks",
        "id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT, text TEXT, occurrence_count INTEGER DEFAULT 1, is_syllabified INTEGER DEFAULT 0, UNIQUE(lang, text)"
    )
    
    # 4. Syllables Table: Tracks distinct syllables and their frequencies per language
    db.create_table(
        "syllables",
        "id INTEGER PRIMARY KEY AUTOINCREMENT, lang TEXT, syllable TEXT, frequency INTEGER DEFAULT 1, UNIQUE(lang, syllable)"
    )
    return db

def is_actual_pdf(file_path):
    """Reads the first 5 bytes of a file to verify it has the %PDF- magic number."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            return header == b'%PDF-'
    except Exception:
        return False

def _extract_pdf_worker(args):
    """
    Top-level worker function that extracts text from a PDF on a separate CPU core.
    """
    pdf_file, pdf_path, chunk_size = args
    try:
        reader = PdfReader(pdf_path, strict=False)
        extracted_chunks = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                extracted_text = page.extract_text()
                if extracted_text:
                    words = extracted_text.split()
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        if chunk.strip():
                            extracted_chunks.append((page_num, chunk))
            except Exception:
                pass 
                
        return (pdf_file, True, extracted_chunks)
    except Exception as e:
        return (pdf_file, False, str(e))

def _normalize_worker(args):
    """
    Top-level worker function that normalizes a batch of text chunks on a separate CPU core.
    """
    lang_code, raw_texts = args
    local_counts = defaultdict(int)
    
    # Instantiate the normalizer locally inside the worker process
    normalizer = MultilingualNormalizer(lang=lang_code, wordify=True, abbreviations=True)
    
    for raw_text in raw_texts:
        if raw_text and raw_text.strip():
            try:
                norm_text = normalizer.normalize(raw_text) if hasattr(normalizer, 'normalize') else normalizer(raw_text)
                if norm_text and str(norm_text).strip():
                    # Increment the local counter
                    local_counts[str(norm_text).strip()] += 1
            except Exception:
                continue
                
    return dict(local_counts) # Return as standard dictionary to pass back to main thread

def process_pdfs(folder_path, lang_input, chunk_size, max_workers, progress=gr.Progress()):
    """Iterates through PDFs, processes them in parallel across CPU cores, and saves to SQLite."""
    if not folder_path or not os.path.exists(folder_path):
        return "❌ Error: The specified folder does not exist."
    
    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')])
    if not pdf_files:
        return "⚠️ No PDF files found in the specified directory."
    
    db = init_db()
    
    total_chunks_inserted = 0
    success_count = 0
    ui_logs = []
    
    existing_rows = db.fetch_all("SELECT pdf_name FROM processed_pdfs")
    processed_set = {row["pdf_name"] for row in existing_rows}
    
    pending_pdfs = []
    for pdf_file in pdf_files:
        if pdf_file in processed_set:
            ui_logs.append(f"⏩ Skipped: {pdf_file} (Already in database)")
            continue
            
        pdf_path = os.path.join(folder_path, pdf_file)
        
        if not is_actual_pdf(pdf_path):
            ui_logs.append(f"🛡️ Security Rejection: {pdf_file} is not a valid PDF file.")
            continue
            
        pending_pdfs.append((pdf_file, pdf_path, int(chunk_size)))
        
    if not pending_pdfs:
        final_msg = "✅ All PDFs are already in the database. No new files to process."
        if ui_logs: final_msg += "\n\n--- LOGS ---\n" + "\n".join(ui_logs)
        return final_msg

    safe_workers = max(1, int(max_workers))
    ui_logs.append(f"🚀 Utilizing {safe_workers} CPU cores for parallel extraction...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        futures = {executor.submit(_extract_pdf_worker, args): args[0] for args in pending_pdfs}
        
        for future in progress.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs..."):
            pdf_file = futures[future]
            try:
                _, success, result = future.result()
                
                if success:
                    db.execute_write("INSERT INTO processed_pdfs (pdf_name, lang) VALUES (?, ?)", (pdf_file, lang_input))
                    pdf_record = db.fetch_one("SELECT id FROM processed_pdfs WHERE pdf_name = ?", (pdf_file,))
                    pdf_id = pdf_record["id"]
                    
                    db_ready_chunks = [(pdf_id, page_num, lang_input, chunk) for page_num, chunk in result]
                    
                    if db_ready_chunks:
                        db.execute_many(
                            "INSERT INTO pdf_chunks (pdf_id, page_number, lang, text) VALUES (?, ?, ?, ?)",
                            db_ready_chunks
                        )
                        total_chunks_inserted += len(db_ready_chunks)
                        
                    success_count += 1
                else:
                    ui_logs.append(f"❌ Error on {pdf_file}: {result}")
                    
            except Exception as e:
                ui_logs.append(f"❌ Fatal process error on {pdf_file}: {str(e)}")
                
    final_status = f"✅ Processed {success_count}/{len(pending_pdfs)} new PDFs. Inserted {total_chunks_inserted} raw text chunks.\n"
    if ui_logs:
        final_status += "\n--- LOGS ---\n" + "\n".join(ui_logs)
        
    return final_status

def normalize_database(lang_code, max_workers, progress=gr.Progress()):
    """Reads unnormalized chunks using B-Tree pagination, processes via a persistent CPU pool, and commits incrementally."""
    try:
        db = init_db()
       
        count_record = db.fetch_one("SELECT COUNT(*) as total FROM pdf_chunks WHERE text IS NOT NULL AND is_normalized = 0")
        total_remaining = count_record["total"] if count_record else 0
        
        if total_remaining == 0:
            return "✅ All chunks are already normalized! No new text to process."
            
        safe_workers = max(1, int(max_workers))
        db_batch_size = 50000 
        worker_chunk_size = 10000
        
        total_processed = 0
        total_inserted = 0
        last_rowid = 0
        
        progress(0, desc="Starting fault-tolerant batch normalization...")
        
        # ⚡ ARCHITECTURAL FIX: Keep the OS process pool open for the entire duration of the task
        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
            
            while True:
                # 1. Fetch using B-Tree pagination (teleports past completed rows)
                records = db.fetch_all(f"SELECT rowid as id, text FROM pdf_chunks WHERE rowid > {last_rowid} AND text IS NOT NULL AND is_normalized = 0 LIMIT {db_batch_size}")
                
                if not records:
                    break
                    
                last_rowid = records[-1]["id"]
                chunk_ids = [row["id"] for row in records]
                raw_texts = [row["text"] for row in records]
                
                # 2. Split into smaller batches for the persistent CPU workers
                batches = [(lang_code, raw_texts[i:i + worker_chunk_size]) for i in range(0, len(raw_texts), worker_chunk_size)]
                
                del records
                del raw_texts
                
                batch_counts = defaultdict(int)
                
                # 3. Dispatch to the ALREADY RUNNING worker pool
                futures = [executor.submit(_normalize_worker, b) for b in batches]
                del batches
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        local_counts = future.result()
                        for text, count in local_counts.items():
                            batch_counts[text] += count
                    except Exception as e:
                        print(f"⚠️ Worker error skipped: {e}")
                            
                # 4. UPSERT the normalized results directly into the database
                if batch_counts:
                    insert_data = [(lang_code, text, count) for text, count in batch_counts.items()]
                    upsert_query = """
                        INSERT INTO normalized_chunks (lang, text, occurrence_count) 
                        VALUES (?, ?, ?) 
                        ON CONFLICT(lang, text) DO UPDATE SET 
                        occurrence_count = normalized_chunks.occurrence_count + excluded.occurrence_count
                    """
                    for i in range(0, len(insert_data), 10000):
                        db.execute_many(upsert_query, insert_data[i:i + 10000])
                    total_inserted += len(insert_data)
                    
                # 5. Checkpoint: Mark this specific block as normalized
                update_data = [(cid,) for cid in chunk_ids]
                db.execute_many("UPDATE pdf_chunks SET is_normalized = 1 WHERE rowid = ?", update_data)
                
                total_processed += len(chunk_ids)
                progress(total_processed / total_remaining, desc=f"Normalized {total_processed}/{total_remaining} chunks...")
            
        return f"✅ Normalization complete! Safely processed {total_processed:,} raw chunks."
        
    except Exception as e:
        return f"❌ Error during normalization: {e}"

def truncate_database():
    """Empties all records from the database tables and resets auto-incrementing IDs."""
    try:
        db = init_db()
        chunks_cleared = db.truncate_table("pdf_chunks")
        pdfs_cleared = db.truncate_table("processed_pdfs")
        norm_cleared = db.truncate_table("normalized_chunks")
        syllables_cleared = db.truncate_table("syllables")
        
        if chunks_cleared and pdfs_cleared and norm_cleared and syllables_cleared:
            return "🗑️ ✅ Database truncated successfully. All tables have been reset."
        else:
            return "❌ Error: Failed to truncate one or more tables."
    except Exception as e:
        return f"❌ Error during truncation: {e}"
            
def _syllabify_worker(args):
    """
    Top-level worker function that extracts syllables from normalized text chunks.
    Multiplies syllable frequency by the chunk's true corpus occurrence count.
    """
    lang_code, text_count_pairs = args
    local_counts = defaultdict(int)
    
    # Instantiate the syllabifier locally (it does not take a lang argument)
    # If supporting other languages later, add a dynamic class selector here
    if lang_code == "tr":
        syllabifier = TurkishSyllabifier() 
    else:
        # Fallback if non-Turkish language is selected but class is Turkish-only
        return dict(local_counts)
    
    for text, count in text_count_pairs:
        if text and text.strip():
            try:
                # Use process_phrase to handle multi-word text chunks
                # It returns a list of: (word, ('syl1', 'syl2'))
                phrase_results = syllabifier.process_phrase(text) 
                
                for word, syllables in phrase_results:
                    for idx, syl in enumerate(syllables):
                        clean_syl = str(syl).strip()
                        if clean_syl:
                            # Prepend the IndexTTS boundary marker to the first syllable of the word
                            if idx == 0:
                                clean_syl = "▁" + clean_syl
                        if clean_syl:
                            # Multiply by the chunk's occurrence count
                            local_counts[clean_syl] += count 
            except Exception:
                continue
                
    return dict(local_counts)
    
def syllabify_database(lang_code, max_workers, progress=gr.Progress()):
    """Reads normalized chunks, processes syllables via CPU pool, and UPSERTs frequencies."""
    try:
        db = init_db()
       
        count_record = db.fetch_one("SELECT COUNT(*) as total FROM normalized_chunks WHERE is_syllabified = 0")
        total_remaining = count_record["total"] if count_record else 0
        
        if total_remaining == 0:
            return "✅ All normalized chunks are already syllabified!"
            
        safe_workers = max(1, int(max_workers))
        db_batch_size = 50000 
        worker_chunk_size = 10000
        
        total_processed = 0
        total_inserted = 0
        last_rowid = 0
        
        progress(0, desc="Starting batch syllabification...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
            
            while True:
                # 1. Fetch using B-Tree pagination
                records = db.fetch_all(f"SELECT rowid as id, text, occurrence_count FROM normalized_chunks WHERE rowid > {last_rowid} AND is_syllabified = 0 LIMIT {db_batch_size}")
                
                if not records:
                    break
                    
                last_rowid = records[-1]["id"]
                chunk_ids = [row["id"] for row in records]
                text_count_pairs = [(row["text"], row["occurrence_count"]) for row in records]
                
                # 2. Split into smaller batches for CPU workers
                batches = [(lang_code, text_count_pairs[i:i + worker_chunk_size]) for i in range(0, len(text_count_pairs), worker_chunk_size)]
                
                batch_counts = defaultdict(int)
                
                # 3. Dispatch to worker pool
                futures = [executor.submit(_syllabify_worker, b) for b in batches]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        local_counts = future.result()
                        for syl, count in local_counts.items():
                            batch_counts[syl] += count
                    except Exception as e:
                        print(f"⚠️ Syllabifier Worker error: {e}")
                            
                # 4. UPSERT syllables to database safely
                if batch_counts:
                    insert_data = [(lang_code, syl, count) for syl, count in batch_counts.items()]
                    upsert_query = """
                        INSERT INTO syllables (lang, syllable, frequency) 
                        VALUES (?, ?, ?) 
                        ON CONFLICT(lang, syllable) DO UPDATE SET 
                        frequency = syllables.frequency + excluded.frequency
                    """
                    for i in range(0, len(insert_data), 10000):
                        db.execute_many(upsert_query, insert_data[i:i + 10000])
                    total_inserted += len(insert_data)
                    
                # 5. Checkpoint: Mark normalized chunks as syllabified
                update_data = [(cid,) for cid in chunk_ids]
                db.execute_many("UPDATE normalized_chunks SET is_syllabified = 1 WHERE rowid = ?", update_data)
                
                total_processed += len(chunk_ids)
                progress(total_processed / total_remaining, desc=f"Syllabified {total_processed}/{total_remaining} chunks...")
            
        return f"✅ Syllabification complete! Processed {total_processed:,} chunks and extracted/updated {total_inserted:,} distinct syllable blocks."
        
    except Exception as e:
        return f"❌ Error during syllabification: {e}"

def train_tokenizer(vocab_size, model_prefix, lang_code, sentence_size, shuffle_sentences, train_extremely, norm_rule, hard_vocab, progress=gr.Progress()):
    """Trains SentencePiece from the pre-normalized unique text chunks for a specific language."""
    if not lang_code:
        return "❌ Error: No language specified for the Tokenizer."
        
    try:
        db = get_db()
        full_prefix_path = os.path.join(core.corpus_directory(), model_prefix)
        
        # PRE-FLIGHT CHECK: Fetch directly from the pre-cleaned table, FILTERED BY LANGUAGE
        records = db.fetch_all("SELECT text FROM normalized_chunks WHERE lang = ?", (lang_code,))
        
        if not records:
            return f"❌ Error: Database contains no normalized text chunks for language '{lang_code}'. Run the Normalizer first."
        
        # Convert to a flat list in memory to completely prevent Python Generator Exhaustion bugs
        text_stream = [row["text"] for row in records if row["text"] and row["text"].strip()]
        
        if not text_stream:
            return f"❌ Error: The extracted text stream is totally empty after formatting."
            
        # Fetch the top 1000 highest frequency syllables for the target language
        syl_records = db.fetch_all("SELECT syllable FROM syllables WHERE lang = ? ORDER BY frequency DESC LIMIT 1000", (lang_code,))
        forced_syllables = [row["syllable"] for row in syl_records if row["syllable"]]
        
        if forced_syllables:
            print(f"💉 Forcing {len(forced_syllables)} high-frequency syllables into the base vocabulary...")
        
        # 1. Initialize ITTS SentencePiece Trainer Wrapper
        trainer = SentencePieceTrainerWrapper(
            vocab_size=int(vocab_size),
            model_type="bpe",
            hard_vocab_limit=bool(hard_vocab),
            normalization_rule_name=norm_rule,                  # Bypass SPM's internal normalizer ("identity")
            train_extremely_large_corpus=bool(train_extremely), # Activate C++ memory optimizations
            input_sentence_size=int(sentence_size),             # Randomly sample X million chunks
            shuffle_input_sentence=bool(shuffle_sentences),     # Ensure a uniform linguistic distribution
            user_defined_symbols=forced_syllables               # ⚡ LOCKS SYLLABLES INTO VOCAB
        )
        
        progress(0.5, desc="Executing C++ SPM Training (This will take a while)...")

        # 2. Execute training using a materialized iterator
        log_output = trainer.train(
            model_prefix=full_prefix_path,
            sentence_iterator=iter(text_stream)
        )
        
        progress(0.9, desc="Cleaning up temporary files...")
        
        # 3. Garbage collection
        gc.collect()
               
        # 4. Prevent False-Positive Success Messages
        if "Failed" in log_output or "❌" in log_output or "Exception" in log_output:
            return f"⚠️ Tokenizer Training Failed!\n\n--- EXECUTION LOG ---\n{log_output}"
        
        progress(1.0, desc="Training Complete!")
        
        return f"✅ Tokenizer '{model_prefix}.model' built successfully for language '{lang_code}'!\n\n--- EXECUTION LOG ---\n{log_output}"
        
    except Exception as e:
        return f"❌ Error training tokenizer: {e}"

# --- CORE PROCESSING LOGIC ---

def save_files_ui(file_objs, corpus_name, progress=gr.Progress()):
    """
    Handles LIST of uploaded files. Saves them to corpus/pdf and corpus/txt.
    """
    logs = []
    
    def update_step(msg):
        logs.append(msg)
        return "\n".join(logs)
    
    def fail_return(msg):
        return update_step(msg), gr.update(), gr.update()

    if not file_objs:
        return fail_return("❌ No files uploaded.")
    
    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    corpus_dir = core.corpus_directory()
    pdf_dir = os.path.join(corpus_dir, "pdf")
    txt_dir = os.path.join(corpus_dir, "txt")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    total_files = len(file_objs)
    yield update_step(f"🚀 Starting batch save for {total_files} file(s)..."), gr.update(), gr.update()

    for idx, file_obj in enumerate(file_objs):
        original_filename = os.path.basename(file_obj.name)
        base_name = os.path.splitext(original_filename)[0]
        
        if total_files == 1 and corpus_name:
            final_name = corpus_name
        else:
            final_name = base_name

        file_ext = os.path.splitext(original_filename)[1].lower()
        
        progress(idx / total_files, desc=f"Saving {original_filename}")
        logs.append(f"\n--- 📄 File {idx+1}/{total_files}: {original_filename} ---")
        
        try:
            if file_ext == ".pdf":
                dest_path = os.path.join(pdf_dir, f"{final_name}.pdf")
                if os.path.exists(dest_path): os.remove(dest_path)
                shutil.move(file_obj.name, dest_path)
                logs.append(f"   💾 Saved PDF to corpus/pdf/")

            elif file_ext == ".txt":
                dest_path = os.path.join(txt_dir, f"{final_name}.txt")
                if os.path.exists(dest_path): os.remove(dest_path)
                shutil.move(file_obj.name, dest_path)
                logs.append(f"   💾 Saved TXT to corpus/txt/")
            else:
                logs.append(f"   ⚠️ Skipped: Unsupported format {file_ext}")
                continue

        except Exception as e:
            logs.append(f"   ❌ Error: {str(e)}")
            
        yield "\n".join(logs), gr.update(), gr.update()

    logs.append("\n✨ BATCH SAVE COMPLETE ✨")
    yield "\n".join(logs), list_files_formatted("pdf", ".pdf"), list_files_formatted("txt", ".txt")

def process_and_add_workspace_files(lang_code, chunk_size, progress=gr.Progress()):
    """
    Reads all PDF and TXT files from the workspace, chunks them dynamically based on user input, 
    normalizes them, and performs a direct UPSERT into the normalized_chunks DB table.
    """
    db = init_db()
    corpus_dir = core.corpus_directory()
    pdf_dir = os.path.join(corpus_dir, "pdf")
    txt_dir = os.path.join(corpus_dir, "txt")
    
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')] if os.path.exists(pdf_dir) else []
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.lower().endswith('.txt')] if os.path.exists(txt_dir) else []
    
    all_files = pdf_files + txt_files
    if not all_files:
        return "⚠️ No PDF or TXT files found in the workspace repositories."
        
    normalizer = MultilingualNormalizer(lang=lang_code, wordify=True, abbreviations=True)
    chunk_size = int(chunk_size)
    local_counts = defaultdict(int)
    
    ui_logs = [f"🚀 Processing {len(all_files)} files from workspace..."]
    progress(0, desc="Extracting and chunking text...")
    
    total_files = len(all_files)
    
    for idx, file_path in enumerate(all_files):
        ext = os.path.splitext(file_path)[1].lower()
        extracted_text = ""
        
        try:
            if ext == '.pdf':
                reader = PdfReader(file_path, strict=False)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + " "
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                    
            if extracted_text:
                words = extracted_text.split()
                # Chunk text dynamically based on specified chunk_size parameter
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if chunk.strip():
                        # Normalize the chunk
                        norm_text = normalizer.normalize(chunk) if hasattr(normalizer, 'normalize') else normalizer(chunk)
                        if norm_text and str(norm_text).strip():
                            local_counts[str(norm_text).strip()] += 1
                            
        except Exception as e:
            ui_logs.append(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
        
        progress((idx + 1) / total_files, desc=f"Processing {os.path.basename(file_path)}")
        
    if not local_counts:
        ui_logs.append("⚠️ No valid text chunks were extracted and normalized.")
        return "\n".join(ui_logs)
        
    progress(0.9, desc="Saving directly to database...")
    ui_logs.append(f"✨ Extracted {len(local_counts)} unique normalized chunks. Upserting to database...")
    
    insert_data = [(lang_code, text, count) for text, count in local_counts.items()]
    
    # Use UPSERT to elegantly aggregate duplicates based on the (lang, text) UNIQUE constraint
    upsert_query = """
        INSERT INTO normalized_chunks (lang, text, occurrence_count) 
        VALUES (?, ?, ?) 
        ON CONFLICT(lang, text) DO UPDATE SET 
        occurrence_count = normalized_chunks.occurrence_count + excluded.occurrence_count
    """
    
    batch_size = 50000
    total_inserted = 0
    
    try:
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i:i + batch_size]
            db.execute_many(upsert_query, batch)
            total_inserted += len(batch)
        ui_logs.append(f"✅ Successfully added {total_inserted} chunk blocks into the vocabulary database!")
    except Exception as e:
        ui_logs.append(f"❌ Database Error: {str(e)}")
        
    return "\n".join(ui_logs)
    
def open_tokenizer_folder():
    """Opens the project's tokenizer directory in the system file explorer."""
    folder_path = core.corpus_directory()

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."

def open_video_folder():
    folder_path = os.path.join(core.wui_outs, "video")

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."
    
def open_cleaner_folder():
    folder_path = os.path.join(core.wui_outs, "cleaner")

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."

def open_diarization_folder():
    folder_path = os.path.join(core.wui_outs, "diarization")

    if not os.path.exists(folder_path):
        return "Folder does not exist."

    os.startfile(folder_path)
    return "Folder opened."
    
def run_ytdlp(url):

    exe_path = os.getenv("ARTHA_YT_DIP_DIR")
    
    if not exe_path:
        return "❌ Error: 'ARTHA_YT_DIP_DIR' environment variable is not set."
       
    exe_file = os.path.join(exe_path, "yt-dlp_x86.exe")
    
    if not os.path.exists(exe_file):
        return f"❌ Error: Executable not found at {exe_file}"
    
    out_path = os.path.join(core.wui_outs, "video")
    os.makedirs(out_path, exist_ok=True)
            
    out_template = os.path.join(out_path, "%(title)s.%(ext)s")
    
    try:
        meta_cmd = [
            exe_file,
            "--print", "title",
            url
        ]
        meta_result = subprocess.run(meta_cmd, capture_output=True, text=True, check=True)
     
        title = meta_result.stdout.strip()
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        
        out_file = os.path.join(out_path, f"{safe_title}.mp3")
    
        command = [
            exe_file,
            "-f", "bestaudio",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            url,
            "-o", out_file
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        log = result.stdout.strip()
        return title, out_file

    except subprocess.CalledProcessError as e:
        return f"❌ Executable failed with error code {e.returncode}.\n\nError Output:\n{e.stderr}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"
        
def clean_audio_with_demucs_api(audio_input, progress=gr.Progress()):
    if not audio_input:
        return "❌ Error: No audio file provided."
        
    # Extract path from Gradio file object
    audio_path = audio_input.name if hasattr(audio_input, "name") else str(audio_input)
    
    # Setup output directory
    output_dir = os.path.join(core.wui_outs, "cleaner")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    final_vocal_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        progress(0.1, desc="Loading Demucs model to VRAM...")
        # 1. Load the model ('htdemucs' is the standard high-quality one)
        model = get_model('htdemucs')
        model.to(device)
        model.eval()
        
        progress(0.3, desc="Loading and formatting audio...")
        # 2. Load audio using torchaudio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to model's expected sample rate (44100 Hz)
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            
        # Demucs expects stereo (2 channels). 
        # If mono, duplicate it. If surround (5.1), mix it down.
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
            
        # Add a batch dimension: (channels, length) -> (1, channels, length)
        wav = wav.unsqueeze(0).to(device)
        
        progress(0.5, desc="Separating stems (this takes time)...")
        # 3. Apply the model
        # split=True chunks long files so you don't run out of VRAM!
        with torch.no_grad():
            sources = apply_model(model, wav, shifts=1, split=True, overlap=0.25)
            
        progress(0.8, desc="Extracting vocal track...")
        # sources shape is (batch, sources, channels, length)
        # Find exactly which index holds the 'vocals'
        vocal_idx = model.sources.index('vocals')
        
        # Extract just the vocals and pull it back to System RAM (CPU)
        vocals_tensor = sources[0, vocal_idx].cpu()
        
        # 4. Save the file directly
        torchaudio.save(final_vocal_path, vocals_tensor, model.samplerate)
        
        progress(0.9, desc="Flushing GPU memory...")
        # 5. CRITICAL: Clean up VRAM so Whisper has room to run next!
        del model
        del wav
        del sources
        del vocals_tensor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        progress(1.0, desc="Done!")
        return f"✅ Success!\nCleaned audio saved to:\n{final_vocal_path}"
        
    except Exception as e:
        return f"❌ Demucs Error: {str(e)}"

def transcribe_audio_ui(audio_input, model_size, use_normalizer, single_paragraph, lang, progress=gr.Progress()):
    if not audio_input:
        return "❌ Error: No audio file provided."
    
    # Extract the file path from the Gradio file object
    audio_path = audio_input.name if hasattr(audio_input, "name") else str(audio_input)
    
    try:
        # Explicitly check for CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        progress(0.1, desc=f"Loading Whisper model on {device.upper()}...")
        model = whisper.load_model(model_size, device=device)
        
        progress(0.3, desc="Transcribing (this may take a while)...")
        # Pass the dynamic language argument to Whisper
        result = model.transcribe(audio_path, language=lang)
        
        progress(0.8, desc="Formatting text...")
        processed_lines = []
        normalizer = MultilingualNormalizer(lang=lang, wordify=True) if use_normalizer else None
        
        # Whisper segments naturally correspond to sentences
        for segment in result["segments"]:
            text = segment["text"].strip()
            if not text:
                continue
                
            if use_normalizer and normalizer:
                processed_text = normalizer.normalize(text)
            else:
                wordifier = MultilingualWordifier(text, language_code=lang)
                processed_text = getattr(wordifier.processor, 'normalized_text', 
                                 getattr(wordifier.processor, 'get_text', lambda: text)())
            
            processed_lines.append(processed_text)

        progress(1.0, desc="Done!")
        
        # Toggle between Single Paragraph vs Line-by-Line
        final_text = " ".join(processed_lines) if single_paragraph else "\n".join(processed_lines)
        
        # Clean up VRAM to prevent OOM on subsequent tool usage
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        return final_text

    except Exception as e:
        return f"❌ Transcription Error: {str(e)}"
        
def diarization_audio_ui(input_file, trim_silence, gap_seconds, min_spks, max_spks):
    if input_file is None: return None
    
    HF_TOKEN = os.environ.get("HF_TOKEN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipeline.to(device)
    
    # Load audio to GPU
    waveform, fs = torchaudio.load(input_file)
    waveform = waveform.to(device)
    audio_data = {"waveform": waveform, "sample_rate": fs}
    
    print(f"--- 🚀 Multi-Speaker Engine Started ---")
    
    # Run Diarization with user-defined speaker limits
    diarization = pipeline(audio_data, min_speakers=min_spks, max_speakers=max_spks)
    
    found_speakers = sorted(diarization.labels())
    print(f"--- 🎤 Found {len(found_speakers)} Speakers: {found_speakers} ---")
    
    generated_files = []
    target_fs = 44100
    resampler = torchaudio.transforms.Resample(fs, target_fs).to(device)

    # Pre-calculate gap buffer
    silence_buffer = None
    if trim_silence and gap_seconds > 0:
        num_gap_samples = int(gap_seconds * fs)
        silence_buffer = torch.zeros((waveform.shape[0], num_gap_samples), device=device)

    # LOOP THROUGH EVERY DETECTED SPEAKER
    for spk_id in found_speakers:
        segments_to_merge = []
        has_audio = False
        
        # --- PROCESSING LOGIC ---
        if not trim_silence:
            output_waveform = torch.zeros_like(waveform)
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == spk_id:
                    start_s, end_s = int(segment.start * fs), int(segment.end * fs)
                    output_waveform[:, start_s:end_s] = waveform[:, start_s:end_s]
                    has_audio = True
            final_tensor = output_waveform
        else:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                if speaker == spk_id:
                    start_s, end_s = int(segment.start * fs), int(segment.end * fs)
                    segments_to_merge.append(waveform[:, start_s:end_s])
                    if silence_buffer is not None:
                        segments_to_merge.append(silence_buffer)
            if segments_to_merge:
                final_tensor = torch.cat(segments_to_merge, dim=-1)
                has_audio = True

        # --- SAVE EACH SPEAKER ---
        if has_audio:
            audio_resampled = resampler(final_tensor)
            audio_out = audio_resampled.t().cpu().numpy()
            
            # Precise naming for high-speed export
            timestamp = datetime.datetime.now().strftime("%H%M%S_%f")
            out_path = os.path.join(core.wui_outs, "diarization")
            os.makedirs(out_path, exist_ok=True)
            save_path = os.path.join(out_path, f"{spk_id}_{timestamp}.wav")
            
            sf.write(save_path, audio_out, target_fs, subtype='PCM_16')
            generated_files.append(save_path)
            print(f"✅ Exported {spk_id}")
    
    # Clean up Pyannote Pipeline and Tensors from VRAM
    del pipeline
    del waveform
    del resampler
    if silence_buffer is not None:
        del silence_buffer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        
    first_audio_preview = generated_files[0] if generated_files else None
    # Return the full list of files to the gr.File component
    return first_audio_preview, generated_files

def get_genre_list():
    """Returns a comprehensive list of document genres for the naming tool."""
    genres = [
        "Academic", "Anthropology", "Archaeology", "Architecture", "Art", 
        "Astrology", "Biography", "Biology", "Business", "Chemistry", 
        "Cinema", "Culinary", "Drama", "Economy", "Education", 
        "Engineering", "Essay", "Fantasy", "Fashion", "Finance", 
        "Health", "History", "Journalism", "Law", "Literature", 
        "Medicine", "Memoir", "Metaphysic", "Music", "Mythology", "Novel", 
        "Philosophy", "Physics", "Poetry", "Politics", "Psychology", 
        "Religion", "Science", "ScienceFiction", "Sociology", "Spiritual",
        "Sports", "Technology", "Theology", "Travel", "Other"
    ]
    genres.sort() # Ensure they are alphabetical
    return genres
    
# Turkish character mapping for standardized naming
TR_NAME_MAP = {
    ord('ç'): 'c', ord('Ç'): 'C',
    ord('ğ'): 'g', ord('Ğ'): 'G',
    ord('ı'): 'i', ord('I'): 'I',
    ord('İ'): 'I', 
    ord('ö'): 'o', ord('Ö'): 'O',
    ord('ş'): 's', ord('Ş'): 'S',
    ord('ü'): 'u', ord('Ü'): 'U'
}

def _clean_for_naming(text):
    """Internal helper to apply standard naming syntax."""
    if not text: return ""
    # 1. Convert Turkish characters
    t = text.translate(TR_NAME_MAP)
    # 2. Replace hyphens with spaces (prevents "Jean-Paul" becoming "JeanPaul")
    t = t.replace("-", " ")
    # 3. Remove all other punctuation (except underscores)
    chars_to_remove = string.punctuation.replace("_", "") 
    t = t.translate(str.maketrans('', '', chars_to_remove))
    return t
    
def generate_standardized_name(genre, author, title):
    if not genre or not author or not title:
        return "Please fill all fields (Genre, Author, and Title)."

    # Process Author: Clean -> Title Case -> Underscores
    c_author = _clean_for_naming(author)
    c_author = " ".join([word.capitalize() for word in c_author.split()]).replace(" ", "_")

    # Process Title: Clean -> Sentence Case -> Underscores
    c_title = _clean_for_naming(title)
    c_title = " ".join([word.capitalize() for word in c_title.split()]).replace(" ", "_")

    return f"{genre}-{c_author}-{c_title}"
    
def generate_audiobook_name(source, narrator, genre, author, title):

    if not all([source, narrator, genre, author, title]):
        return "Please fill all fields (Source, Narrator, Genre, Author, and Title)."

    # Process Source & Narrator: Clean -> Title Case -> Underscores
    c_source = _clean_for_naming(source)
    c_source = " ".join([word.capitalize() for word in c_source.split()]).replace(" ", "_")
    
    c_narrator = _clean_for_naming(narrator)
    c_narrator = " ".join([word.capitalize() for word in c_narrator.split()]).replace(" ", "_")

    # Process Author & Title (Same as Document Namer)
    c_author = _clean_for_naming(author)
    c_author = " ".join([word.capitalize() for word in c_author.split()]).replace(" ", "_")
    
    c_title = _clean_for_naming(title)
    c_title = " ".join([word.capitalize() for word in c_title.split()]).replace(" ", "_")

    return f"Audiobook-{c_source}-{c_narrator}-{genre}-{c_author}-{c_title}"
    
# ======================================================
# UI CREATION
# ======================================================

def create_demo():
    
    lang_options = core.language_list()
    
    with gr.Blocks() as demo:
        gr.Markdown(_("CORPUS_HEADER"))
        gr.Markdown(_("CORPUS_DESC"))        
        
        with gr.Tabs():            
            # --- TAB 1: PDF Corpus Builder ---
            with gr.Tab(_("CORPUS_DB_TAB_PDF")):
                gr.Markdown(_("CORPUS_DB_DESC_PDF"))
                
                with gr.Row():
                    folder_input = gr.Textbox(
                        label=_("CORPUS_DB_LABEL_FOLDER"), 
                        placeholder=_("CORPUS_DB_PLACEHOLDER_FOLDER"),
                        scale=3
                    )
                    lang_input = gr.Dropdown(
                        label=_("CORPUS_DB_LABEL_LANG_PIPE"), 
                        choices=lang_options,
                        value="tr"
                    )
                    chunk_input = gr.Number(
                        label=_("CORPUS_DB_LABEL_CHUNK"), 
                        value=10, 
                        precision=0,
                        scale=1
                    )
                    worker_input = gr.Slider(
                        label=_("CORPUS_DB_LABEL_WORKERS"), 
                        minimum=1, 
                        maximum=multiprocessing.cpu_count(), 
                        value=max(1, multiprocessing.cpu_count() // 2), 
                        step=1,
                        scale=1
                    )
                    
                with gr.Row():
                    process_btn = gr.Button(_("CORPUS_DB_BTN_PROCESS"), variant="primary")
                    truncate_btn = gr.Button(_("CORPUS_DB_BTN_TRUNCATE"), variant="stop")
                    
                with gr.Row():
                    db_output_log = gr.Textbox(label=_("CORPUS_DB_LABEL_LOGS"), interactive=False, lines=3)

            # --- TAB 2: Text Normalizer ---
            with gr.Tab(_("CORPUS_DB_TAB_NORM")):
                gr.Markdown(_("CORPUS_DB_DESC_NORM"))
                
                with gr.Row():
                    worker_input_norm = gr.Slider(
                        label=_("CORPUS_DB_LABEL_WORKERS"), 
                        minimum=1, 
                        maximum=multiprocessing.cpu_count(), 
                        value=max(1, multiprocessing.cpu_count() // 2), 
                        step=1
                    ) 
                    
                with gr.Row():
                    norm_btn = gr.Button(_("CORPUS_DB_BTN_NORMALIZE"), variant="primary")
                    
                with gr.Row():
                    norm_output_log = gr.Textbox(label=_("CORPUS_DB_LABEL_NORM_STATUS"), interactive=False, lines=2)

            # --- TAB 3: Syllabifier
            with gr.Tab(_("CORPUS_DB_TAB_SYL")):
                gr.Markdown(_("CORPUS_DB_DESC_SYL"))
                
                with gr.Row():
                    syl_lang_input = gr.Dropdown(
                        label=_("COMMON_LABEL_LANG"), 
                        choices=lang_options,
                        value="tr"
                    )
                    worker_input_syl = gr.Slider(
                        label=_("CORPUS_DB_LABEL_WORKERS"), 
                        minimum=1, 
                        maximum=multiprocessing.cpu_count(), 
                        value=max(1, multiprocessing.cpu_count() // 2), 
                        step=1
                    ) 
                    
                with gr.Row():
                    syl_btn = gr.Button(_("CORPUS_DB_BTN_SYL"), variant="primary")
                    
                with gr.Row():
                    syl_output_log = gr.Textbox(label=_("CORPUS_DB_LABEL_SYL_STATUS"), interactive=False, lines=2)
            
            # --- TAB 4: Tokenizer ---
            with gr.Tab(_("CORPUS_DB_TAB_TOK")):
                gr.Markdown(_("CORPUS_DB_DESC_TOK"))
                
                with gr.Row():
                    tok_lang_input = gr.Dropdown(label=_("COMMON_LABEL_LANG"), choices=lang_options, value="tr")
                    vocab_input = gr.Number(label=_("CORPUS_DB_LABEL_VOCAB"), value=8000, precision=0)
                    prefix_input = gr.Textbox(label=_("CORPUS_DB_LABEL_PREFIX"), value="itts_bpe")
                    
                with gr.Row():
                    tok_sentence_size = gr.Number(label="Max Sentences (Sample Size)", value=5000000, precision=0, info="Limits RAM usage on massive datasets. Set to 0 to use all.")
                    tok_norm_rule = gr.Dropdown(label="Normalization Rule", choices=["identity", "nmt_nfkc", "nfkc", "nfkc_cf"], value="identity", info="Bypass SPM normalizer with 'identity' if DB is pre-normalized.")
                                       
                with gr.Row():
                    tok_train_ext = gr.Checkbox(label="Train Extremely Large Corpus", value=True, info="Activates C++ memory optimizations for multi-gigabyte files.")                   
                    tok_shuffle = gr.Checkbox(label="Shuffle Corpus", value=True, info="Randomly sample to ensure vocabulary diversity.")
                    tok_hard_vocab = gr.Checkbox(label="Hard Vocab Limit", value=False, info="Strictly enforce the requested vocabulary size without padding.")                    
                    
                with gr.Row():
                    train_btn = gr.Button(_("CORPUS_DB_BTN_TRAIN"), variant="primary")
                    
                with gr.Row():
                    tok_output_log = gr.Textbox(label=_("CORPUS_DB_LABEL_TOK_STATUS"), interactive=False, lines=2)
                
                with gr.Row():    
                    tok_folder_btn = gr.Button(_("COMMON_FOLDER_OPEN"))
                
        gr.HTML("<div style='height:10px'></div>")
        
        # ==========
        # UTILITIES
        # ==========
        with gr.Group():
            gr.Markdown(_("CORPUS_HEADER_UTILS"), elem_classes="wui-markdown")
            
        # --- SECTION: UPLOAD DOCUMENTS ---
        with gr.Accordion(_("CORPUS_HEADER_ADD"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_ADD"))   
            with gr.Row():
                file_input = gr.File(
                    label=_("COMMON_LABEL_UPLOAD"),
                    file_types=[".pdf", ".txt"],
                    file_count="multiple" 
                )
            with gr.Row():
                corpus_name = gr.Textbox(
                    label=_("CORPUS_LABEL_NAME"),
                    placeholder=_("CORPUS_PLACEHOLDER_NAME"),
                    scale=3
                )
                corpus_lang = gr.Dropdown(
                    label=_("COMMON_LABEL_LANG"),
                    choices=lang_options, 
                    value="tr",
                    scale=1
                )
                corpus_chunk = gr.Number(
                    label=_("CORPUS_DB_LABEL_CHUNK"),
                    value=10, 
                    precision=0,
                    scale=1
                )
            with gr.Row():
                save_btn = gr.Button(_("CORPUS_BTN_DSAVE"), variant="secondary", elem_classes="wui-button-blue")
                process_and_add_btn = gr.Button(_("CORPUS_BTN_DMERGE"), variant="primary", elem_classes="wui-button-green")
            with gr.Row():
                corpus_log = gr.Textbox(
                    label=_("COMMON_LABEL_LOGS"),
                    lines=6,
                    max_lines=6
                )
            
        # --- SECTION: FILE REPOSITORIES ---
        with gr.Accordion(_("CORPUS_ACC_REPO"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_REPO"))   
            with gr.Row():
                with gr.Column():
                    gr.Markdown(_("CORPUS_HEADER_PDF"))
                    pdf_files = gr.Textbox(
                        label=_("CORPUS_LABEL_PDF"), 
                        value=lambda: list_files_formatted("pdf", ".pdf"), 
                        lines=10,
                        max_lines=10,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown(_("CORPUS_HEADER_TXT"))
                    txt_files = gr.Textbox(
                        label=_("CORPUS_LABEL_TXT"), 
                        value=lambda: list_files_formatted("txt", ".txt"), 
                        lines=10,
                        max_lines=10,
                        interactive=False
                    )
                    
            gr.HTML("<br>")
            with gr.Row():
                refresh_btn = gr.Button(_("COMMON_BTN_REFRESH"), variant="secondary")
            
        # --- SECTION: YOUTUBE DOWNLOADER ---
        with gr.Accordion(_("CORPUS_ACC_YT"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_YT"))   
            with gr.Row():
                yt_url = gr.Textbox(label=_("CORPUS_LABEL_URL"))
            
            yt_run_btn = gr.Button(_("CORPUS_BTN_FETCH"), variant="primary")
            
            yt_log = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=1, max_lines=6, interactive=False)
            
            yt_aud = gr.Audio(label=_("CORPUS_LABEL_PREVIEW"), type="filepath")
            
            yt_folder_btn = gr.Button(_("COMMON_FOLDER_OPEN"))
            
        # --- SECTION: AUDIO CLEANER ---
        with gr.Accordion(_("CORPUS_ACC_CLEANER"), open=False, elem_classes="wui-accordion"):    
            gr.Markdown(_("CORPUS_DESC_CLEANER"))   
            with gr.Row():
                clean_audio_input = gr.File(label=_("CORPUS_LABEL_UPLOAD_AUDIO"), file_types=[".wav", ".mp3"])
            
            clean_btn = gr.Button(_("CORPUS_BTN_ISOLATE"), variant="primary")
            
            clean_output = gr.Textbox(label=_("COMMON_LABEL_LOGS"), lines=5)
            
            clean_folder_btn = gr.Button(_("COMMON_FOLDER_OPEN"))
           
        # --- SECTION: AUDIO TRANSCRIPTOR ---
        with gr.Accordion(_("CORPUS_ACC_WHISPER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_WHISPER"))
            with gr.Row():
                transcribe_audio_input = gr.File(
                    label=_("CORPUS_LABEL_WHISPER_AUDIO"), 
                    file_types=[".wav", ".mp3"]
                )
                with gr.Column():
                    transcribe_model_size = gr.Dropdown(
                        label=_("CORPUS_LABEL_WHISPER_MODEL"), 
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], 
                        value="large-v3",
                        info=_("CORPUS_INFO_WHISPER_MODEL")
                    )
                    transcribe_lang = gr.Dropdown(
                        label=_("COMMON_LABEL_LANG"), 
                        choices=lang_options, 
                        value="tr",
                        info=_("CORPUS_INFO_WHISPER_LANG")
                    )
                with gr.Column():
                    with gr.Row():
                        transcribe_use_normalizer = gr.Checkbox(
                            label=_("CORPUS_CHK_NORMALIZER"), 
                            value=False,
                            info=_("CORPUS_INFO_NORMALIZER")
                        )
                    with gr.Row():
                        transcribe_single_paragraph = gr.Checkbox(
                            label=_("CORPUS_CHK_PARAGRAPH"), 
                            value=True,
                        )
            
            transcribe_btn = gr.Button(_("CORPUS_BTN_TRANSCRIBE"), variant="primary")
            transcribe_output = gr.Textbox(
                label=_("CORPUS_LABEL_TRANSCRIBE_OUT"), 
                lines=8, 
                placeholder=_("CORPUS_PLACEHOLDER_TRANSCRIBE"),
                interactive=True,
                buttons=["copy"]
            )

        # --- SECTION: DIARIZATION ---
        with gr.Accordion(_("CORPUS_ACC_DIARIZATION"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_DIARIZATION"))           
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(type="filepath", label=_("CORPUS_LABEL_DIA_AUDIO"))
                    
                    with gr.Group():
                        gr.Markdown(_("CORPUS_HEADER_DIA_SETTINGS"))
                        trim_toggle = gr.Checkbox(label=_("CORPUS_CHK_TRIM"), value=False)
                        gap_input = gr.Number(label=_("CORPUS_LABEL_GAP"), value=0.0, step=0.1)
                        
                    with gr.Row():
                        min_s = gr.Slider(1, 10, value=1, step=1, label=_("CORPUS_LABEL_MIN_SPK"))
                        max_s = gr.Slider(1, 20, value=10, step=1, label=_("CORPUS_LABEL_MAX_SPK"))
                        
                    diarize_btn = gr.Button(_("CORPUS_BTN_DIA_START"), variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown(_("CORPUS_HEADER_DIA_FILES"))
                    first_speaker_audio = gr.Audio(label=_("CORPUS_LABEL_DIA_PREVIEW"), type="filepath", interactive=False)
                    file_output = gr.File(label=_("CORPUS_LABEL_DIA_DOWNLOAD"), file_count="multiple")
                    dia_folder_btn = gr.Button(_("COMMON_FOLDER_OPEN"))

        # --- SECTION: DOCUMENT NAMER ---
        with gr.Accordion(_("CORPUS_ACC_NAMER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_NAMER"))
            
            with gr.Row():
                namer_genre = gr.Dropdown(
                    label=_("CORPUS_LABEL_GENRE"),
                    choices=get_genre_list(), 
                    value="Novel",
                    filterable=True
                )
                namer_author = gr.Textbox(
                    label=_("CORPUS_LABEL_AUTHOR"), 
                    placeholder=_("CORPUS_PLACEHOLDER_AUTHOR")
                )
                namer_title = gr.Textbox(
                    label=_("CORPUS_LABEL_TITLE"), 
                    placeholder=_("CORPUS_PLACEHOLDER_TITLE")
                )
            
            with gr.Row():
                namer_btn = gr.Button(_("CORPUS_BTN_GEN_NAME"), variant="secondary")
            
            namer_output = gr.Textbox(
                label=_("CORPUS_LABEL_RESULT"),
                interactive=True
            )
        
        # --- SECTION: AUDIOBOOK NAMER ---
        with gr.Accordion(_("CORPUS_ACC_AB_NAMER"), open=False, elem_classes="wui-accordion"):
            gr.Markdown(_("CORPUS_DESC_AB_NAMER"))
            
            with gr.Row():
                ab_source = gr.Textbox(label=_("CORPUS_LABEL_SOURCE"), placeholder=_("CORPUS_PLACEHOLDER_SOURCE"))
                ab_narrator = gr.Textbox(label=_("CORPUS_LABEL_NARRATOR"), placeholder=_("CORPUS_PLACEHOLDER_NARRATOR"))
            
            with gr.Row():
                ab_genre = gr.Dropdown(
                    label=_("CORPUS_LABEL_GENRE"),
                    choices=get_genre_list(),
                    value="Novel",
                    filterable=True
                )
                ab_author = gr.Textbox(label=_("CORPUS_LABEL_AUTHOR"), placeholder=_("CORPUS_PLACEHOLDER_AB_AUTHOR"))
                ab_title = gr.Textbox(label=_("CORPUS_LABEL_TITLE"), placeholder=_("CORPUS_PLACEHOLDER_AB_TITLE"))
            
            with gr.Row():
                ab_btn = gr.Button(_("CORPUS_BTN_GEN_AB"), variant="secondary")
            
            ab_output = gr.Textbox(label=_("CORPUS_LABEL_RESULT"), interactive=True)
             
        # ==========================================
        # ACTIONS
        # ==========================================
        
        # Database Manager
        process_btn.click(fn=process_pdfs, inputs=[folder_input, lang_input, chunk_input, worker_input], outputs=db_output_log)
        truncate_btn.click(fn=truncate_database, inputs=None, outputs=db_output_log)
        norm_btn.click(fn=normalize_database, inputs=[lang_input, worker_input_norm], outputs=norm_output_log)
        syl_btn.click(fn=syllabify_database, inputs=[syl_lang_input, worker_input_syl], outputs=syl_output_log)
        train_btn.click(
            fn=train_tokenizer, 
            inputs=[vocab_input, prefix_input, tok_lang_input, tok_sentence_size, tok_shuffle, tok_train_ext, tok_norm_rule, tok_hard_vocab], 
            outputs=tok_output_log
        )
        
        tok_folder_btn.click(
            fn=open_tokenizer_folder,
            inputs=None,
            outputs=None
        )
        
        # Tools
        save_btn.click(
            fn=save_files_ui,
            inputs=[file_input, corpus_name],
            outputs=[corpus_log, pdf_files, txt_files] 
        )
        
        process_and_add_btn.click(
            fn=process_and_add_workspace_files,
            inputs=[corpus_lang, corpus_chunk],
            outputs=[corpus_log]
        )
        
        refresh_btn.click(
            fn=refresh_lists,
            inputs=[],
            outputs=[pdf_files, txt_files]
        )
        
        yt_run_btn.click(
            fn=run_ytdlp,
            inputs=[yt_url],
            outputs=[yt_log, yt_aud]
        )
        
        yt_folder_btn.click(
            fn=open_video_folder,
            outputs=None
        )
        
        clean_btn.click(
            fn=clean_audio_with_demucs_api,
            inputs=[clean_audio_input],
            outputs=[clean_output]
        )
        
        clean_folder_btn.click(
            fn=open_cleaner_folder,
            outputs=None
        )
        
        transcribe_btn.click(
            fn=transcribe_audio_ui,
            inputs=[transcribe_audio_input, transcribe_model_size, transcribe_use_normalizer, transcribe_single_paragraph, transcribe_lang],
            outputs=[transcribe_output]
        )
        
        diarize_btn.click(
            fn=diarization_audio_ui, 
            inputs=[audio_input, trim_toggle, gap_input, min_s, max_s], 
            outputs=[first_speaker_audio, file_output]
        )
        
        dia_folder_btn.click(
            fn=open_diarization_folder,
            outputs=None
        )
        
        namer_btn.click(
            fn=generate_standardized_name,
            inputs=[namer_genre, namer_author, namer_title],
            outputs=[namer_output]
        )
        
        ab_btn.click(
            fn=generate_audiobook_name,
            inputs=[ab_source, ab_narrator, ab_genre, ab_author, ab_title],
            outputs=[ab_output]
        )
        
        # =============
        # DOCUMENTATION
        # =============
        with gr.Group():
            gr.Markdown(_("COMMON_HEADER_DOCS"), elem_classes="wui-markdown") 
        
        with gr.Accordion(_("COMMON_ACC_GUIDE"), open=False, elem_classes="wui-accordion"):
            guide_markdown = gr.Markdown(
                value=core.load_guide_text("corpus"), elem_classes="wui-markdown"
            )
            
        gr.HTML("<div style='height:10px'></div>")
        
    return demo