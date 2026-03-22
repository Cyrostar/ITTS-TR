import os
import re
from typing import List, Union
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.message_factory import MessageFactory

class GenericSpiceTokenizer:

    def __init__(self, vocab_file: str, normalizer=None, cjk: bool = False):
        if not os.path.exists(vocab_file):
            raise ValueError(f"❌ BPE model file not found at: {vocab_file}")
        
        self.vocab_file = vocab_file
        self.normalizer = normalizer
        self.cjk = cjk
        
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)
        
        if self.cjk:
            self.cjk_pattern = re.compile(
                r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
            )
            
    def _apply_cjk_spacing(self, text: str) -> str:
        """Isolates CJK characters with spaces to prevent BPE token collision."""
        chars = self.cjk_pattern.split(text.strip())
        return " ".join([w.strip() for w in chars if w.strip()])

    @property
    def vocab_size(self) -> int:
        return self.sp_model.GetPieceSize()
        
    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def pad_token(self) -> str:
        return None

    @property
    def bos_token(self) -> str:
        return "<s>"

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def special_tokens_map(self) -> dict:
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    @property
    def pad_token_id(self) -> int:
        return -1

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def unk_token_id(self) -> int:
        return self.sp_model.unk_id()

    def tokenize(self, text: str) -> List[str]:
        if self.normalizer:
            text = self.normalizer.normalize(text) if hasattr(self.normalizer, 'normalize') else self.normalizer(text)
        if self.cjk:
            text = self._apply_cjk_spacing(text) 
        return self.sp_model.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def encode(self, text: str, **kwargs) -> List[int]:
        if self.normalizer:
            text = self.normalizer.normalize(text) if hasattr(self.normalizer, 'normalize') else self.normalizer(text)
        if self.cjk:
            text = self._apply_cjk_spacing(text)
        return self.sp_model.EncodeAsIds(text)

    def decode(self, ids: Union[int, List[int]], **kwargs) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return self.sp_model.DecodeIds(ids)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> List[str]:
        if isinstance(ids, int):
            ids = [ids]
        return [self.sp_model.IdToPiece(i) for i in ids]

    def split_segments(self, tokenized: List[str], max_text_tokens_per_segment=120, quick_streaming_tokens=0, **kwargs) -> List[List[str]]:
        """
        Intelligently splits token arrays into chunks, prioritizing sentence boundaries 
        and punctuation to prevent words from being cut in half mid-speech.
        """
        if not tokenized:
            return []
            
        # SentencePiece uses U+2581 instead of standard spaces!
        sp_space = '\u2581' 
        punct_markers = {
            '.', ',', '!', '?', ';', ':', 
            sp_space + '.', sp_space + ',', sp_space + '!', 
            sp_space + '?', sp_space + ';', sp_space + ':'
        }
        
        segments = []
        current_idx = 0
        total_len = len(tokenized)

        while current_idx < total_len:
            # If the remaining tokens fit in one segment, just take them all
            if total_len - current_idx <= max_text_tokens_per_segment:
                segments.append(tokenized[current_idx:])
                break

            # Calculate the hard mathematical limit for this chunk
            limit_idx = current_idx + max_text_tokens_per_segment
            
            # Start scanning backwards from the limit to find a punctuation mark
            best_cut_idx = limit_idx
            found_punct = False
            
            # Scan back up to 40% of the max_segment size.
            scan_floor = max(current_idx + (max_text_tokens_per_segment // 2), current_idx + 10)
            
            for scan_idx in range(limit_idx - 1, scan_floor - 1, -1):
                token = tokenized[scan_idx]
                if token in punct_markers:
                    # We found a punctuation mark! Cut immediately AFTER it.
                    best_cut_idx = scan_idx + 1
                    found_punct = True
                    break
                    
            # If no punctuation was found, search for a word boundary using the special SP space
            if not found_punct:
                for scan_idx in range(limit_idx - 1, scan_floor - 1, -1):
                    token = tokenized[scan_idx]
                    if token.startswith(sp_space):
                        # We found the start of a new word! Cut right BEFORE it.
                        best_cut_idx = scan_idx
                        break

            # Append the calculated chunk and move the index forward
            segments.append(tokenized[current_idx:best_cut_idx])
            current_idx = best_cut_idx

        return segments

class JsonToModelConverter:
    """
    Converts a JSON vocabulary list into a valid .model file using 
    SentencePiece training and internal Protobuf surgery.
    """
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.reserved_ids = {0, 1, 2}
        self._sp_model_proto = self._get_proto_class()

    def _get_proto_class(self):
        """
        Loads the ModelProto class by reading and patching the user's 
        sentencepiece_model_pb2.py file. Supports Protobuf 4.x/5.x.
        """
        # 1. Locate the file (Check root 'wui' and 'wui/core')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(os.path.dirname(current_dir), "sentencepiece_model_pb2.py"), # wui/sentencepiece_model_pb2.py
            os.path.join(current_dir, "sentencepiece_model_pb2.py"),                   # wui/core/sentencepiece_model_pb2.py
        ]
        
        target_file = None
        for p in possible_paths:
            if os.path.exists(p):
                target_file = p
                break
        
        if not target_file:
            raise FileNotFoundError(f"Missing 'sentencepiece_model_pb2.py'. Please ensure it is in {os.path.dirname(current_dir)}.")

        # 2. Robust Patching
        new_lines = []
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                if "import builder as _builder" in line: continue
                if "_builder.Build" in line: break
                new_lines.append(line)
        
        code = "\n".join(new_lines)

        # 3. Execute in isolated scope
        local_scope = {}
        try:
            exec(code, {}, local_scope)
        except Exception as e:
            raise RuntimeError(f"Failed to patch sentencepiece_model_pb2.py: {e}")
        
        descriptor = local_scope.get('DESCRIPTOR')
        if not descriptor:
             raise ValueError("Could not find DESCRIPTOR object in sentencepiece_model_pb2.py")
             
        # 4. Create class via MessageFactory
        factory = MessageFactory()
        model_proto_desc = descriptor.message_types_by_name.get('ModelProto')
        return factory.GetPrototype(model_proto_desc)

    def convert(self, json_data):
        """Builds a model from JSON by training a dummy base and injecting JSON tokens."""
        try:
            sorted_data = sorted(json_data, key=lambda x: x["id"])
            user_symbols = [item["piece"] for item in sorted_data if item["id"] not in self.reserved_ids]
            
            model_prefix = os.path.join(self.output_dir, self.dataset_name)
            
            # Use simple dummy data to minimize "required_chars" conflicts
            dummy_data = ["a"] * 100 
            
            vocab_size = len(sorted_data)

            # Common training arguments
            train_args = dict(
                model_prefix=model_prefix,
                model_type="bpe",
                user_defined_symbols=user_symbols,
                bos_id=0, eos_id=1, unk_id=2, pad_id=-1,
                character_coverage=0.9995,  # Relaxed coverage
                byte_fallback=False,        # Disable byte tokens
                train_extremely_large_corpus=False,
                shuffle_input_sentence=False
            )

            # 1. Dummy Train (with Retry Logic for Off-by-One errors)
            try:
                # Pass a fresh iterator directly into the train call
                spm.SentencePieceTrainer.train(sentence_iterator=iter(dummy_data), vocab_size=vocab_size, **train_args)
            except RuntimeError as e:
                if "Vocabulary size is smaller than required_chars" in str(e):
                    print(f"⚠️ Protobuf packing size mismatch. Retrying with {vocab_size + 1}...")
                    # Pass a NEW fresh iterator for the retry
                    spm.SentencePieceTrainer.train(sentence_iterator=iter(dummy_data), vocab_size=vocab_size + 1, **train_args)
                else:
                    raise e
            
            # 2. Surgery
            self._apply_surgery(f"{model_prefix}.model", sorted_data)
            
            # 3. Clean up the .vocab file overwritten by the C++ backend
            vocab_path = f"{model_prefix}.vocab"
            try:
                with open(vocab_path, "w", encoding="utf-8") as vf:
                    for item in sorted_data:
                        piece = item.get("piece", "")
                        score = item.get("score", 0.0)
                        vf.write(f"{piece}\t{score}\n")
            except Exception as e:
                print(f"⚠️ Warning: Failed to rewrite .vocab file: {e}")
            
            return f"✅ Protobuf conversion successful: {model_prefix}.model"
            
        except Exception as e:
            return f"❌ Conversion Error: {str(e)}"

    def _apply_surgery(self, model_path, sorted_json):
        """Directly modifies binary model pieces using embedded Proto classes."""
        proto_cls = self._sp_model_proto
        proto = proto_cls()
        
        with open(model_path, "rb") as f:
            proto.ParseFromString(f.read())

        # CONSTANTS from sentencepiece_model.proto
        TYPE_NORMAL = 1
        TYPE_UNKNOWN = 2  
        TYPE_CONTROL = 3  
        TYPE_USER_DEFINED = 4
        TYPE_BYTE = 6
        TYPE_UNUSED = 5

        for i, json_item in enumerate(sorted_json):
            if i < len(proto.pieces):
                p = proto.pieces[i]
                p.piece = json_item["piece"]
                p.score = json_item.get("score", 0.0)
                
                if json_item.get("is_unknown"): 
                    p.type = TYPE_UNKNOWN
                elif json_item.get("is_control"): 
                    p.type = TYPE_CONTROL
                elif json_item.get("is_unused"): 
                    p.type = TYPE_UNUSED
                elif json_item.get("is_byte"): 
                    p.type = TYPE_BYTE
                else: 
                    p.type = TYPE_NORMAL
                    
        if len(proto.pieces) > len(sorted_json):
            del proto.pieces[len(sorted_json):]

        proto.trainer_spec.vocab_size = len(proto.pieces)
        with open(model_path, "wb") as f:
            f.write(proto.SerializeToString())