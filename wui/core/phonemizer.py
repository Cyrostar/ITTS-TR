"""
Turkish Text-to-Speech Tokenizer and Phonemizer

This module converts Turkish text to phonemes with TTS-specific features:
- Stress marking
- Prosody (pauses, intonation)
- Text normalization
- Customizable abbreviations via file
- Batch processing
- Multi-speaker support
- Emotion markers
"""

import os
import re
import json
from typing import List, Dict, Union, Optional, Tuple
from collections import Counter

class TurkishTTSTokenizer:
    """
    Main tokenizer class for Turkish TTS with phonemization.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, abbreviations_file=None, cache_size=10000):
        """
        Initialize the TTS tokenizer.
        
        Args:
            abbreviations_file: Path to abbreviations file (optional)
            cache_size: Size of LRU cache for syllabification (default: 10000)
        """
        # Initialize syllabifier
        self.syllabifier = TurkishSyllabifier(cache_size=cache_size)
        
        # ASCII-friendly phoneme mappings
        self.phoneme_map = {
            'a': 'a',
            'b': 'b',
            'c': 'j',      # like 'j' in 'jam'
            'ç': 'ch',     # like 'ch' in 'church'
            'd': 'd',
            'e': 'e',
            'f': 'f',
            'g': 'g',
            'ğ': 'gh',     # soft g - lengthens/softens
            'h': 'h',
            'ı': 'uh',     # undotted i
            'i': 'i',
            'j': 'zh',     # like 's' in 'measure'
            'k': 'k',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'o': 'o',
            'ö': 'oe',
            'p': 'p',
            'r': 'r',
            's': 's',
            'ş': 'sh',
            't': 't',
            'u': 'u',
            'ü': 'ue',
            'v': 'v',
            'y': 'y',
            'z': 'z',
            'w': 'v',      # Common in loanwords
        }
        
        # Build phoneme vocabulary (all unique phonemes)
        self.phoneme_vocab = self.build_phoneme_vocab()
        
        # Create phoneme to index mapping
        self.phoneme_to_idx = {p: idx for idx, p in enumerate(self.phoneme_vocab)}
        self.idx_to_phoneme = {idx: p for p, idx in self.phoneme_to_idx.items()}
        
        # Initialize abbreviations and exceptions
        self.abbreviations = {}
        self.pronunciation_variants = {}  # Words with multiple pronunciations
        
        # Load abbreviations from file if provided
        if abbreviations_file:
            self.load_abbreviations_from_file(abbreviations_file)
        else:
            # Try to load from default location
            self.load_abbreviations_from_file()
        
        # Number words in Turkish
        self.ones = ['', 'bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz']
        self.tens = ['', 'on', 'yirmi', 'otuz', 'kırk', 'elli', 'altmış', 'yetmiş', 'seksen', 'doksan']
        self.scale_names = [
            ('', 1),
            ('bin', 1000),
            ('milyon', 1000000),
            ('milyar', 1000000000),
            ('trilyon', 1000000000000)
        ]
        
        # Turkish months
        self.months = {
            '01': 'ocak', '02': 'şubat', '03': 'mart', '04': 'nisan',
            '05': 'mayıs', '06': 'haziran', '07': 'temmuz', '08': 'ağustos',
            '09': 'eylül', '10': 'ekim', '11': 'kasım', '12': 'aralık'
        }
        
        # Common Turkish contractions
        self.contractions = {
            "ne'dir": "nedir",
            "ne'ye": "neye",
            "yapıyo": "yapıyor",
            "geliyo": "geliyor",
        }
        
        # Compile regex patterns for performance
        self.compile_patterns()
    
    def compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.number_pattern = re.compile(r'\b(\d+)\b')
        self.punctuation_pattern = re.compile(r'[^\w\sğçışöü]')
        self.date_pattern = re.compile(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b')
        self.time_pattern = re.compile(r'\b(\d{1,2}):(\d{2})\b')
        self.currency_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*₺')
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.percentage_pattern = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')
    
    def load_abbreviations_from_file(self, filepath=None):
        """
        Load abbreviations from a file.
        
        Args:
            filepath: Path to abbreviations file. If None, tries default location.
            
        Returns:
            Number of abbreviations loaded
        """
        count = 0
        
        if filepath is None:
            script_dir = os.path.dirname(__file__)
            filepath = os.path.join(script_dir, "abbreviations.txt")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try both = and : as separators
                    if '=' in line:
                        abbr, expansion = line.split('=', 1)
                    elif ':' in line:
                        abbr, expansion = line.split(':', 1)
                    else:
                        continue
                    
                    abbr = abbr.strip()
                    expansion = expansion.strip()
                    
                    if abbr and expansion:
                        self.abbreviations[abbr] = expansion
                        count += 1
            
            print(f"✓ Loaded {count} abbreviations from {filepath}")
            return count
        
        except FileNotFoundError:
            print(f"⚠ Abbreviations file not found: {filepath}")
            return 0
        
        except Exception as e:
            print(f"⚠ Error loading abbreviations: {str(e)}")
            return 0
    
    def expand_contractions(self, text):
        """
        Expand common Turkish contractions.
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def normalize_dates(self, text):
        """
        Convert dates to spoken form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized dates
        """
        def date_to_words(match):
            day, month, year = match.groups()
            day_word = self.number_to_words(int(day))
            month_word = self.months.get(month.zfill(2), month)
            year_word = self.number_to_words(int(year))
            return f"{day_word} {month_word} {year_word}"
        
        return self.date_pattern.sub(date_to_words, text)
    
    def normalize_time(self, text):
        """
        Convert time to spoken form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized time
        """
        def time_to_words(match):
            hour, minute = match.groups()
            hour_word = self.number_to_words(int(hour))
            minute_word = self.number_to_words(int(minute))
            return f"saat {hour_word} {minute_word}"
        
        return self.time_pattern.sub(time_to_words, text)
    
    def normalize_currency(self, text):
        """
        Convert currency to spoken form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized currency
        """
        def currency_to_words(match):
            amount = match.group(1).replace(',', '.')
            amount_float = float(amount)
            
            if amount_float == int(amount_float):
                # Whole number
                words = self.number_to_words(int(amount_float))
                return f"{words} lira"
            else:
                # Decimal number
                lira_part = int(amount_float)
                kurus_part = int(round((amount_float - lira_part) * 100))
                
                lira_words = self.number_to_words(lira_part)
                result = f"{lira_words} lira"
                
                if kurus_part > 0:
                    kurus_words = self.number_to_words(kurus_part)
                    result += f" {kurus_words} kuruş"
                
                return result
        
        return self.currency_pattern.sub(currency_to_words, text)
    
    def normalize_percentage(self, text):
        """
        Convert percentages to spoken form.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized percentages
        """
        def percentage_to_words(match):
            number = match.group(1).replace(',', '.')
            number_float = float(number)
            
            if number_float == int(number_float):
                words = self.number_to_words(int(number_float))
            else:
                words = str(number).replace('.', ' nokta ')
            
            return f"yüzde {words}"
        
        return self.percentage_pattern.sub(percentage_to_words, text)
    
    def normalize_urls_emails(self, text):
        """
        Remove or normalize URLs and emails.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized URLs/emails
        """
        # Replace URLs with a marker
        text = self.url_pattern.sub('<URL>', text)
        # Replace emails with a marker
        text = self.email_pattern.sub('<EMAIL>', text)
        return text
    
    def normalize_text(self, text):
        """
        Normalize text for TTS (numbers, abbreviations, dates, etc.)
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Normalize dates and times
        text = self.normalize_dates(text)
        text = self.normalize_time(text)
        
        # Normalize currency and percentages
        text = self.normalize_currency(text)
        text = self.normalize_percentage(text)
        
        # Handle URLs and emails
        text = self.normalize_urls_emails(text)
        
        # Convert numbers to words
        text = self.number_pattern.sub(
            lambda m: self.number_to_words(int(m.group(1))), 
            text
        )
        
        # Replace abbreviations (longest first to handle multi-word abbreviations)
        for abbr in sorted(self.abbreviations.keys(), key=len, reverse=True):
            expansion = self.abbreviations[abbr]
            # Use word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def number_to_words(self, num):
        """
        Convert a number to Turkish words.
        
        Args:
            num: Integer number
            
        Returns:
            Turkish word representation
        """
        if num == 0:
            return 'sıfır'
        if num < 0:
            return 'eksi ' + self.number_to_words(abs(num))
        
        # Handle numbers larger than trillion
        if num >= 1000000000000000:
            return str(num)  # Keep very large numbers as digits
        
        result = []
        
        # Process each scale (trillion, billion, million, thousand, hundreds)
        for scale_name, scale_value in reversed(self.scale_names[1:]):  # Skip the empty one
            if num >= scale_value:
                scale_part = num // scale_value
                num = num % scale_value
                
                # Special case: 1000 is "bin" not "bir bin"
                if scale_value == 1000 and scale_part == 1:
                    result.append('bin')
                else:
                    # Convert the scale part (0-999) to words
                    scale_words = self.convert_hundreds(scale_part)
                    result.append(scale_words + ' ' + scale_name)
        
        # Handle remaining hundreds, tens, ones
        if num > 0:
            result.append(self.convert_hundreds(num))
        
        return ' '.join(result)
    
    def convert_hundreds(self, num):
        """
        Convert a number (0-999) to Turkish words.
        
        Args:
            num: Integer from 0 to 999
            
        Returns:
            Turkish word representation
        """
        if num == 0:
            return ''
        
        result = []
        
        # Hundreds
        hundreds = num // 100
        if hundreds > 0:
            if hundreds == 1:
                result.append('yüz')
            else:
                result.append(self.ones[hundreds] + ' yüz')
        
        # Tens and ones
        remainder = num % 100
        tens_digit = remainder // 10
        ones_digit = remainder % 10
        
        if tens_digit > 0:
            result.append(self.tens[tens_digit])
        if ones_digit > 0:
            result.append(self.ones[ones_digit])
        
        return ' '.join(result)
    
    def detect_prosody(self, text):
        """
        Detect prosody markers (pauses, intonation).
        
        Args:
            text: Input text
            
        Returns:
            Text with prosody markers
        """
        if not text:
            return ""
        
        # Add pause markers
        text = text.replace('.', ' <PAUSE_LONG>')
        text = text.replace(',', ' <PAUSE_SHORT>')
        text = text.replace(';', ' <PAUSE_MED>')
        text = text.replace('!', ' <PAUSE_LONG> <EMPHASIS>')
        text = text.replace('?', ' <PAUSE_LONG> <RISING>')
        text = text.replace(':', ' <PAUSE_MED>')
        
        return text
    
    def add_emotion_markers(self, text, emotion='neutral'):
        """
        Add emotion markers for emotional speech.
        
        Args:
            text: Input text
            emotion: Emotion type (neutral, happy, sad, angry, excited)
            
        Returns:
            Text with emotion markers
        """
        valid_emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
        
        if emotion not in valid_emotions:
            emotion = 'neutral'
        
        if emotion != 'neutral':
            return f"<EMOTION_{emotion.upper()}> {text} <EMOTION_END>"
        
        return text
    
    def add_speaker_id(self, sequence, speaker_id):
        """
        Add speaker ID to sequence for multi-speaker TTS.
        
        Args:
            sequence: Phoneme sequence (list of indices)
            speaker_id: Speaker identifier (integer)
            
        Returns:
            Tuple of (sequence, speaker_id)
        """
        return (sequence, speaker_id)
    
    def text_to_phonemes(self, text, include_prosody=True, emotion='neutral'):
        """
        Convert text to phonemes with TTS features.
        
        Args:
            text: Input text
            include_prosody: Include prosody markers
            emotion: Emotion for the text
            
        Returns:
            Dictionary with phoneme information
        """
        if not text:
            return {
                'original': text,
                'normalized': '',
                'tokens': [],
                'full_phonemes': []
            }
        
        # Add emotion markers
        if emotion != 'neutral':
            text = self.add_emotion_markers(text, emotion)
        
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Add prosody markers
        if include_prosody:
            prosody_text = self.detect_prosody(normalized)
        else:
            prosody_text = normalized
        
        # Tokenize into words
        tokens = prosody_text.split()
        
        result = {
            'original': text,
            'normalized': normalized,
            'tokens': [],
            'full_phonemes': []
        }
        
        for token in tokens:
            # Handle prosody and emotion markers
            if token.startswith('<') and token.endswith('>'):
                result['tokens'].append({
                    'word': token,
                    'type': 'marker',
                    'marker': token
                })
                result['full_phonemes'].append(token)
                continue
            
            # Clean word (remove punctuation)
            clean_word = self.punctuation_pattern.sub('', token.lower())
            
            if not clean_word:
                continue
            
            # Syllabify using syllabifier
            syllables = self.syllabifier.syllabify(clean_word)
            
            # Add stress using syllabifier
            stressed_syllables = self.syllabifier.add_stress(syllables, clean_word)
            
            # Convert to phonemes
            phonemes = []
            for syllable in syllables:
                syl_phonemes = []
                for char in syllable:
                    if char in self.phoneme_map:
                        syl_phonemes.append(self.phoneme_map[char])
                phonemes.append(''.join(syl_phonemes))
            
            # Convert stressed syllables to phonemes
            stressed_phonemes = []
            for syllable in stressed_syllables:
                syl_phonemes = []
                if syllable.startswith("'"):
                    syl_phonemes.append("'")
                    syllable = syllable[1:]
                for char in syllable:
                    if char in self.phoneme_map:
                        syl_phonemes.append(self.phoneme_map[char])
                stressed_phonemes.append(''.join(syl_phonemes))
            
            token_info = {
                'word': token,
                'clean': clean_word,
                'syllables': list(syllables),
                'stressed_syllables': stressed_syllables,
                'phonemes': phonemes,
                'stressed_phonemes': stressed_phonemes,
                'full_phoneme': '-'.join(stressed_phonemes)
            }
            
            result['tokens'].append(token_info)
            result['full_phonemes'].append('-'.join(stressed_phonemes))
        
        return result

    def build_phoneme_vocab(self):
        """
        Build vocabulary of all unique phonemes.
        Includes special tokens for padding, unknown, and prosody.
        Order is fixed and deterministic for consistent indexing.
        
        Returns:
            List of all phonemes and special tokens in fixed order
        """
        # Define phonemes in a fixed order (sorted alphabetically)
        unique_phonemes = [
            'a', 'b', 'ch', 'd', 'e', 'f', 'g', 'gh', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'oe', 'p', 'r', 's',
            'sh', 't', 'u', 'ue', 'uh', 'v', 'y', 'z', 'zh'
        ]
        
        # Add special tokens at the beginning (fixed positions)
        special_tokens = [
            '<PAD>',           # index 0 - Padding token
            '<UNK>',           # index 1 - Unknown token
            '<SOS>',           # index 2 - Start of sequence
            '<EOS>',           # index 3 - End of sequence
            ' ',               # index 4 - Space/word boundary
            '-',               # index 5 - Syllable boundary
            "'",               # index 6 - Stress marker
            '<PAUSE_SHORT>',   # index 7 - Short pause
            '<PAUSE_MED>',     # index 8 - Medium pause
            '<PAUSE_LONG>',    # index 9 - Long pause
            '<EMPHASIS>',      # index 10 - Emphasis
            '<RISING>',        # index 11 - Rising intonation
            '<URL>',           # index 12 - URL marker
            '<EMAIL>',         # index 13 - Email marker
            '<EMOTION_HAPPY>',    # index 14
            '<EMOTION_SAD>',      # index 15
            '<EMOTION_ANGRY>',    # index 16
            '<EMOTION_EXCITED>',  # index 17
            '<EMOTION_CALM>',     # index 18
            '<EMOTION_END>',      # index 19
        ]
        
        # Combine: special tokens first, then phonemes
        vocab = special_tokens + unique_phonemes
        
        return vocab
    
    def get_vocab_size(self):
        """
        Return the size of the phoneme vocabulary.
        
        Returns:
            Integer vocabulary size
        """
        return len(self.phoneme_vocab)
    
    def get_pad_token_id(self):
        """Get the ID of the padding token"""
        return self.phoneme_to_idx['<PAD>']
    
    def get_sos_token_id(self):
        """Get the ID of the start-of-sequence token"""
        return self.phoneme_to_idx['<SOS>']
    
    def get_eos_token_id(self):
        """Get the ID of the end-of-sequence token"""
        return self.phoneme_to_idx['<EOS>']
    
    def save_vocab(self, filepath=None):
        """
        Save vocabulary to JSON file in Hugging Face format.
        
        Args:
            filepath: Path to save vocabulary. If None, uses default location.
        """
        if filepath is None:
            script_dir = os.path.dirname(__file__)
            filepath = os.path.join(script_dir, "vocab.json")
        
        # Hugging Face standard format: just token-to-index mapping
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.phoneme_to_idx, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Vocabulary saved to {filepath} (Hugging Face format)")
        print(f"  Vocabulary size: {self.get_vocab_size()}")
    
    def load_vocab(self, filepath=None):
        """
        Load vocabulary from JSON file in Hugging Face format.
        
        Args:
            filepath: Path to vocabulary file. If None, uses default location.
        """
        if filepath is None:
            script_dir = os.path.dirname(__file__)
            filepath = os.path.join(script_dir, "vocab.json")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.phoneme_to_idx = json.load(f)
        
        # Generate reverse mapping
        self.idx_to_phoneme = {int(idx): phoneme for phoneme, idx in self.phoneme_to_idx.items()}
        
        # Regenerate phoneme_vocab from loaded data
        self.phoneme_vocab = [self.idx_to_phoneme[i] for i in range(len(self.idx_to_phoneme))]
        
        print(f"✓ Vocabulary loaded from {filepath} (Hugging Face format)")
        print(f"  Vocabulary size: {self.get_vocab_size()}")
    
    def save_config(self, filepath=None):
        """
        Save tokenizer configuration.
        
        Args:
            filepath: Path to save config. If None, uses default location.
        """
        if filepath is None:
            script_dir = os.path.dirname(__file__)
            filepath = os.path.join(script_dir, "tokenizer_config.json")
        
        config = {
            'version': self.VERSION,
            'cache_size': self.syllabifier.cache_size,
            'vocab_size': self.get_vocab_size(),
            'abbreviations': self.abbreviations,
            'stress_exceptions': self.syllabifier.stress_exceptions,
            'pronunciation_variants': self.pronunciation_variants,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Config saved to {filepath}")
    
    def load_config(self, filepath=None):
        """
        Load tokenizer configuration.
        
        Args:
            filepath: Path to config file. If None, uses default location.
        """
        if filepath is None:
            script_dir = os.path.dirname(__file__)
            filepath = os.path.join(script_dir, "tokenizer_config.json")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.syllabifier.cache_size = config.get('cache_size', 10000)
        self.abbreviations = config.get('abbreviations', {})
        self.syllabifier.stress_exceptions = config.get('stress_exceptions', {})
        self.pronunciation_variants = config.get('pronunciation_variants', {})
        
        print(f"✓ Config loaded from {filepath}")
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load tokenizer from pretrained model directory (Hugging Face style).
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Loaded tokenizer instance
        """
        tokenizer = cls()
        
        # Load vocab
        vocab_path = os.path.join(model_path, "vocab.json")
        if os.path.exists(vocab_path):
            tokenizer.load_vocab(vocab_path)
        
        # Load config
        config_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            tokenizer.load_config(config_path)
        
        # Load abbreviations
        abbr_path = os.path.join(model_path, "abbreviations.txt")
        if os.path.exists(abbr_path):
            tokenizer.load_abbreviations_from_file(abbr_path)
        
        print(f"✓ Tokenizer loaded from {model_path}")
        return tokenizer
    
    def save_pretrained(self, save_directory):
        """
        Save tokenizer to directory (Hugging Face style).
        
        Args:
            save_directory: Directory to save tokenizer files
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save vocab
        vocab_path = os.path.join(save_directory, "vocab.json")
        self.save_vocab(vocab_path)
        
        # Save config
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        self.save_config(config_path)
        
        print(f"✓ Tokenizer saved to {save_directory}")
    
    def phoneme_to_sequence(self, phoneme_string):
        """
        Convert phoneme string to sequence of indices.
        
        Args:
            phoneme_string: String of phonemes (e.g., "mer-ha-'ba")
            
        Returns:
            List of integer indices
        """
        if not phoneme_string:
            return []
        
        indices = []
        i = 0
        
        while i < len(phoneme_string):
            matched = False
            
            # Check for special tokens (like <PAUSE_SHORT>)
            if phoneme_string[i] == '<':
                end_idx = phoneme_string.find('>', i)
                if end_idx != -1:
                    token = phoneme_string[i:end_idx+1]
                    if token in self.phoneme_to_idx:
                        indices.append(self.phoneme_to_idx[token])
                        i = end_idx + 1
                        matched = True
            
            # Try 2-character phonemes
            if not matched and i + 1 < len(phoneme_string):
                two_char = phoneme_string[i:i+2]
                if two_char in self.phoneme_to_idx:
                    indices.append(self.phoneme_to_idx[two_char])
                    i += 2
                    matched = True
            
            # Try 1-character phonemes
            if not matched:
                one_char = phoneme_string[i]
                if one_char in self.phoneme_to_idx:
                    indices.append(self.phoneme_to_idx[one_char])
                else:
                    # Unknown phoneme
                    indices.append(self.phoneme_to_idx['<UNK>'])
                i += 1
        
        return indices
    
    def sequence_to_phoneme(self, indices):
        """
        Convert sequence of indices back to phoneme string.
        
        Args:
            indices: List of integer indices
            
        Returns:
            Phoneme string
        """
        if not indices:
            return ""
        
        phonemes = []
        for idx in indices:
            if idx in self.idx_to_phoneme:
                phonemes.append(self.idx_to_phoneme[idx])
            else:
                phonemes.append('<UNK>')
        
        return ''.join(phonemes)
    
    def sequence_to_text(self, indices, approximate=True):
        """
        Convert phoneme indices back to readable text (approximate).
        
        Args:
            indices: List of integer indices
            approximate: If True, attempts to reconstruct readable text
            
        Returns:
            Reconstructed text or phoneme string
        """
        phoneme_string = self.sequence_to_phoneme(indices)
        
        if not approximate:
            return phoneme_string
        
        # Remove stress markers and syllable boundaries for readability
        text = phoneme_string.replace("'", "").replace("-", "")
        
        # Replace special tokens with readable equivalents
        replacements = {
            '<PAUSE_SHORT>': ',',
            '<PAUSE_MED>': ';',
            '<PAUSE_LONG>': '.',
            '<EMPHASIS>': '!',
            '<RISING>': '?',
            '<SOS>': '',
            '<EOS>': '',
            '<PAD>': '',
        }
        
        for token, replacement in replacements.items():
            text = text.replace(token, replacement)
        
        return text.strip()
    
    def text_to_sequence(self, text, add_sos_eos=False, emotion='neutral'):
        """
        Convert text directly to numerical sequence (end-to-end).
        
        Args:
            text: Input text
            add_sos_eos: Add start/end of sequence tokens
            emotion: Emotion for the text
            
        Returns:
            List of integer indices
        """
        if not text:
            return []
        
        # Convert text to phonemes
        result = self.text_to_phonemes(text, include_prosody=True, emotion=emotion)
        
        # Get full phoneme string
        phoneme_string = ' '.join(result['full_phonemes'])
        
        # Convert to indices
        indices = self.phoneme_to_sequence(phoneme_string)
        
        # Add SOS/EOS if requested
        if add_sos_eos:
            sos_idx = self.phoneme_to_idx['<SOS>']
            eos_idx = self.phoneme_to_idx['<EOS>']
            indices = [sos_idx] + indices + [eos_idx]
        
        return indices
    
    def pad_sequence(self, sequences, max_length=None, padding_value=None, truncate=True):
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of sequences (each is a list of indices)
            max_length: Maximum length. If None, uses longest sequence
            padding_value: Value to use for padding. If None, uses <PAD> token
            truncate: Whether to truncate sequences longer than max_length
            
        Returns:
            Padded sequences as list of lists
        """
        if not sequences:
            return []
        
        if padding_value is None:
            padding_value = self.get_pad_token_id()
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) > max_length:
                if truncate:
                    padded.append(seq[:max_length])
                else:
                    padded.append(seq)
            else:
                # Pad sequence
                padding_needed = max_length - len(seq)
                padded.append(seq + [padding_value] * padding_needed)
        
        return padded
    
    def get_attention_mask(self, sequences, padding_value=None):
        """
        Generate attention masks (1 for real tokens, 0 for padding).
        
        Args:
            sequences: List of sequences (padded)
            padding_value: Padding token ID. If None, uses <PAD> token
            
        Returns:
            List of attention masks (same shape as sequences)
        """
        if padding_value is None:
            padding_value = self.get_pad_token_id()
        
        masks = []
        for seq in sequences:
            mask = [1 if token != padding_value else 0 for token in seq]
            masks.append(mask)
        
        return masks
    
    def batch_text_to_sequence(self, texts, add_sos_eos=False, padding=True, 
                               max_length=None, emotion='neutral', return_attention_mask=True):
        """
        Process multiple texts at once with padding.
        
        Args:
            texts: List of input texts
            add_sos_eos: Add start/end tokens
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            emotion: Emotion for all texts (or list of emotions)
            return_attention_mask: Return attention masks
            
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        # Handle emotion as list or single value
        if isinstance(emotion, str):
            emotions = [emotion] * len(texts)
        else:
            emotions = emotion
        
        # Convert all texts to sequences
        sequences = []
        for text, emo in zip(texts, emotions):
            seq = self.text_to_sequence(text, add_sos_eos=add_sos_eos, emotion=emo)
            sequences.append(seq)
        
        # Pad sequences if requested
        if padding:
            sequences = self.pad_sequence(sequences, max_length=max_length)
        
        result = {'input_ids': sequences}
        
        # Add attention masks if requested
        if return_attention_mask:
            result['attention_mask'] = self.get_attention_mask(sequences)
        
        return result
    
    def validate_phoneme_sequence(self, sequence):
        """
        Check if phoneme sequence is valid.
        
        Args:
            sequence: List of phoneme indices
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sequence:
            return (True, "Empty sequence")
        
        for idx in sequence:
            if idx not in self.idx_to_phoneme:
                return (False, f"Invalid index {idx} not in vocabulary")
        
        return (True, "Valid sequence")
    
    def test_roundtrip(self, text):
        """
        Test text -> phonemes -> indices -> phonemes roundtrip.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with test results
        """
        # Forward pass
        phoneme_result = self.text_to_phonemes(text)
        phoneme_string = ' '.join(phoneme_result['full_phonemes'])
        indices = self.phoneme_to_sequence(phoneme_string)
        
        # Backward pass
        reconstructed_phonemes = self.sequence_to_phoneme(indices)
        reconstructed_text = self.sequence_to_text(indices, approximate=True)
        
        # Validation
        is_valid, msg = self.validate_phoneme_sequence(indices)
        
        return {
            'original_text': text,
            'normalized_text': phoneme_result['normalized'],
            'phoneme_string': phoneme_string,
            'indices': indices,
            'reconstructed_phonemes': reconstructed_phonemes,
            'reconstructed_text': reconstructed_text,
            'is_valid': is_valid,
            'validation_message': msg,
            'sequence_length': len(indices)
        }
    
    def get_phoneme_frequency(self, text):
        """
        Analyze phoneme distribution in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with phoneme counts
        """
        result = self.text_to_phonemes(text)
        phoneme_string = ' '.join(result['full_phonemes'])
        indices = self.phoneme_to_sequence(phoneme_string)
        
        # Count phonemes
        phoneme_counts = Counter()
        for idx in indices:
            if idx in self.idx_to_phoneme:
                phoneme = self.idx_to_phoneme[idx]
                phoneme_counts[phoneme] += 1
        
        return {
            'total_phonemes': len(indices),
            'unique_phonemes': len(phoneme_counts),
            'frequency': dict(phoneme_counts.most_common()),
            'coverage': len(phoneme_counts) / self.get_vocab_size()
        }
    
    def validate_phoneme_coverage(self, corpus):
        """
        Check if vocabulary covers all phonemes in corpus.
        
        Args:
            corpus: List of texts or single text
            
        Returns:
            Dictionary with coverage statistics
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        
        all_phoneme_indices = set()
        unknown_count = 0
        total_phonemes = 0
        
        for text in corpus:
            indices = self.text_to_sequence(text)
            total_phonemes += len(indices)
            
            for idx in indices:
                all_phoneme_indices.add(idx)
                if idx == self.phoneme_to_idx['<UNK>']:
                    unknown_count += 1
        
        return {
            'total_texts': len(corpus),
            'total_phonemes': total_phonemes,
            'unique_phonemes_used': len(all_phoneme_indices),
            'vocab_size': self.get_vocab_size(),
            'coverage_ratio': len(all_phoneme_indices) / self.get_vocab_size(),
            'unknown_count': unknown_count,
            'unknown_ratio': unknown_count / total_phonemes if total_phonemes > 0 else 0
        }
    
    def get_phoneme_duration_estimate(self, text):
        """
        Estimate phoneme durations (simplified model).
        
        Args:
            text: Input text
            
        Returns:
            List of (phoneme, estimated_duration_ms)
        """
        result = self.text_to_phonemes(text)
        
        # Simple duration model (in milliseconds)
        base_durations = {
            'vowels': 80,      # Vowels are longer
            'consonants': 60,  # Consonants are shorter
            'pause_short': 150,
            'pause_med': 250,
            'pause_long': 400,
        }
        
        durations = []
        
        for token in result['tokens']:
            if token.get('type') == 'marker':
                marker = token['marker']
                if 'PAUSE_SHORT' in marker:
                    durations.append((marker, base_durations['pause_short']))
                elif 'PAUSE_MED' in marker:
                    durations.append((marker, base_durations['pause_med']))
                elif 'PAUSE_LONG' in marker:
                    durations.append((marker, base_durations['pause_long']))
                else:
                    durations.append((marker, 0))
            else:
                # Regular word
                for phoneme in token.get('stressed_phonemes', []):
                    # Check if phoneme contains vowel
                    contains_vowel = any(v in phoneme.lower() for v in 'aeıioöuü')
                    
                    if contains_vowel:
                        duration = base_durations['vowels']
                        # Stressed vowels are slightly longer
                        if phoneme.startswith("'"):
                            duration *= 1.2
                    else:
                        duration = base_durations['consonants']
                    
                    durations.append((phoneme, int(duration)))
        
        return durations
    
    def get_statistics(self):
        """
        Get tokenizer statistics.
        
        Returns:
            Dictionary with various statistics
        """
        return {
            'version': self.VERSION,
            'vocab_size': self.get_vocab_size(),
            'num_abbreviations': len(self.abbreviations),
            'num_stress_exceptions': len(self.syllabifier.stress_exceptions),
            'cache_size': self.syllabifier.cache_size,
            'cache_info': self.syllabifier.get_cache_info()._asdict(),
            'special_tokens': {
                'pad_id': self.get_pad_token_id(),
                'sos_id': self.get_sos_token_id(),
                'eos_id': self.get_eos_token_id(),
            }
        }
    
    def clear_cache(self):
        """Clear the syllabification cache"""
        self.syllabifier.clear_cache()
        print("✓ Syllabification cache cleared")
    
    def add_stress_exception(self, word, stress_position):
        """
        Add a word with exceptional stress pattern.
        
        Args:
            word: Word with exceptional stress
            stress_position: Index of stressed syllable (0-based)
        """
        self.syllabifier.add_stress_exception(word, stress_position)
    
    def __repr__(self):
        """String representation of tokenizer"""
        return (f"TurkishTTSTokenizer(vocab_size={self.get_vocab_size()}, "
                f"abbreviations={len(self.abbreviations)}, "
                f"version={self.VERSION})")
    
    def __len__(self):
        """Return vocabulary size"""
        return self.get_vocab_size()