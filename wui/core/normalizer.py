import re
import unicodedata
from typing import List
import abc

class BaseWordifier(abc.ABC):
    """
    Abstract base class enforcing the standard text normalization 
    pipeline for any new languages added to the TTS system.
    """
    def __init__(self, text_block: str, abbreviations: bool = False):
        self.abbreviations = abbreviations
        self.normalized_text = self._process_pipeline(text_block)

    @abc.abstractmethod
    def _process_pipeline(self, text: str) -> str:
        """All language classes must implement their specific normalization rules here."""
        pass
        
    def get_words(self) -> List[str]:
        """
        Natively returns the fully processed text as a list of individual words.
        """
        if not self.normalized_text:
            return []
        return self.normalized_text.split()

class MultilingualWordifier:
    """
    A dynamic, multilingual text normalization router.
    """
    def __init__(self, text_block: str, language_code: str = 'en', abbreviations: bool = False):
        
        processor_class = WORDIFIER_REGISTRY.get(language_code.lower(), UniversalWordifier)
        
        if not processor_class:
            raise NotImplementedError(f"TTS Front-End Error: No wordifier registered for '{language_code}'.")
        
        self.processor = processor_class(text_block, abbreviations=abbreviations)
        
    def get_words(self) -> List[str]:
        """
        Safely extracts the word list regardless of whether the underlying
        processor is a legacy class or a new abstract base class.
        """
        if hasattr(self.processor, 'get_words'):
            return self.processor.get_words()
        elif hasattr(self.processor, 'normalized_text'):
            return self.processor.normalized_text.split()
        return []

class BaseNormalizer(abc.ABC):
    """
    Abstract base class enforcing the standard text normalization 
    pipeline for any new languages added to the TTS system.
    """
    def __init__(self, lang: str, extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        self.lang = lang
        self.extract_mode = extract
        self.upper_mode = upper
        self.wordify_mode = wordify
        self.abbreviations_mode = abbreviations
        
    @abc.abstractmethod
    def normalize(self, text: str) -> str:
        """All language classes must implement their specific cleaning and splitting rules here."""
        pass

class MultilingualNormalizer:
    """
    A dynamic, multilingual text normalization router.
    Instantiates and delegates processing to the correct language-specific class.
    """
    def __init__(self, lang: str = 'en', extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        
        processor_class = NORMALIZER_REGISTRY.get(lang.lower(), UniversalNormalizer)
        
        if not processor_class:
            raise NotImplementedError(f"TTS Front-End Error: No normalizer registered for '{lang}'.")
        
        self.processor = processor_class(
            lang=lang,
            extract=extract,
            upper=upper,
            wordify=wordify,
            abbreviations=abbreviations
        )
        
    def normalize(self, text: str) -> str:
        """
        Executes the normalize method of the dynamically loaded class.
        This provides a seamless, unified method call for your training pipeline.
        """
        return self.processor.normalize(text)
        
        
# =========
# UNIVERSAL
# =========

class UniversalWordifier(BaseWordifier):
    """
    A generic, fallback TTS text normalizer for unsupported languages.
    Performs basic unicode normalization and whitespace cleanup.
    """
    def __init__(self, text_block: str, abbreviations: bool = False):
        super().__init__(text_block, abbreviations)

    def _process_pipeline(self, text: str) -> str:
        if not text:
            return ""
            
        # 1. Unicode Normalization (Standardizes composed/decomposed characters)
        text = unicodedata.normalize("NFC", text)
        
        # 2. Basic Whitespace Cleanup
        text = text.strip()
        
        # 3. Clean Multiple Spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text

class UniversalNormalizer(BaseNormalizer):
    """
    A generic, fallback TTS text normalizer for unsupported languages.
    Performs basic unicode normalization, whitespace cleanup, and casing.
    """
    def __init__(self, lang: str = "und", extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        super().__init__(lang, extract, upper, wordify, abbreviations)
        self.whitespace_re = re.compile(r'\s+')

    def extract_graphemes(self, text: str) -> str:
        return " ".join(list(text.replace(" ", "")))

    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. Route through the MultilingualWordifier if wordification is enabled
        if self.wordify_mode:
            wordifier_router = MultilingualWordifier(text, language_code=self.lang, abbreviations=self.abbreviations_mode)
            if hasattr(wordifier_router.processor, 'normalized_text'):
                text = wordifier_router.processor.normalized_text
            elif hasattr(wordifier_router.processor, 'get_text'):
                text = wordifier_router.processor.get_text()
                
        # 2. Basic Unicode Normalization
        text = unicodedata.normalize("NFC", text)
        
        # 3. Clean Double Spaces
        text = self.whitespace_re.sub(' ', text).strip()
        
        # 4. Apply final modes
        if self.extract_mode:
            text = self.extract_graphemes(text)
            
        # 5. Basic Casing
        if self.upper_mode:
            text = text.upper()
        else:
            text = text.lower()
            
        return text.strip()        
        
# =======
# TURKISH
# =======

class TurkishWordifier(BaseWordifier):
    """
    Isolated, native TTS text normalizer for Turkish (tr).
    """
    MONTHS_TR = {
        "01":"ocak", "02":"şubat", "03":"mart", "04":"nisan",
        "05":"mayıs", "06":"haziran", "07":"temmuz", "08":"ağustos",
        "09":"eylül", "10":"ekim", "11":"kasım", "12":"aralık"
    }
    
    ORDINAL_EXCEPTIONS = {
        1: "birinci", 2: "ikinci", 3: "üçüncü", 4: "dördüncü",
        5: "beşinci", 6: "altıncı", 7: "yedinci",
        8: "sekizinci", 9: "dokuzuncu", 10: "onuncu"
    }

    ALL_ABBREVIATIONS_TR = {
        "vs.": "vesaire", "vb.": "ve benzeri", "vd.": "ve diğerleri",
        "bkz.": "bakınız", "yy.": "yüzyıl", "m.ö.": "milattan önce",
        "m.s.": "milattan sonra", "m.ö": "milattan önce", "m.s": "milattan sonra",
        "dr.": "doktor", "prof.": "profesör", "doç.": "doçent",
        "arş. gör.": "araştırma görevlisi", "öğr. gör.": "öğretim görevlisi",
        "uzm.": "uzman", "yrd. doç.": "yardımcı doçent", "alb.": "albay",
        "org.": "orgeneral", "av.": "avukat", "müh.": "mühendis",
        "t.c.": "türkiye cumhuriyeti", "a.ş.": "anonim şirketi",
        "ltd.": "limited", "şti.": "şirketi", "san.": "sanayi", "tic.": "ticaret",
        "mah.": "mahallesi", "cad.": "caddesi", "sok.": "sokağı", "no.": "numara",
        "tel.": "telefon", "nbr.": "ne haber", "slm.": "selam", "tmm.": "tamam",
        "hz.": "hazreti", "r.a.": "radıyallahu anh",
        "a.s.": "aleyhisselam", "s.a.v.": "sallallahu aleyhi ve sellem",
        "kg.": "kilogram", "gr.": "gram", "lt.": "litre", "ml.": "mililitre",
        "cm.": "santimetre", "mm.": "milimetre", "km.": "kilometre"
    }

    def __init__(self, text_block: str, abbreviations: bool = False):
        self._ABBREV_PATTERNS = [
            (re.compile(r'\b' + re.escape(k) + r'(?=\s|[.,!?]|$)', re.IGNORECASE), v)
            for k, v in sorted(self.ALL_ABBREVIATIONS_TR.items(), key=lambda x: len(x[0]), reverse=True)
        ]
        
        self.turkic_chars = r"a-zA-ZÇçĞğIıİiÖöŞşÜüƏəQqXxÑñÄäŽžŇňÝýŪūÂâÊêÎîÔôÛû"
        self.turkic_pattern = rf"([{self.turkic_chars}]+)"
        self._word_pattern = re.compile(rf"[{self.turkic_chars}]+")
        
        super().__init__(text_block, abbreviations)

    def number_to_turkish_words(self, n: int) -> str:
        units = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
        tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
        
        scales = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon", "kentilyon"]
        
        if n == 0: return "sıfır"

        if len(str(n)) > 21:
            return " ".join([units[int(d)] if d != '0' else 'sıfır' for d in str(n)])

        def process_three_digits(num: int) -> str:
            s = []
            h = num // 100
            if h > 0:
                if h > 1: s.append(units[h])
                s.append("yüz")
            rem = num % 100
            if rem >= 10:
                s.append(tens[rem // 10])
                if rem % 10 > 0: s.append(units[rem % 10])
            elif rem > 0:
                s.append(units[rem])
            return "".join(s)

        chunks = []
        temp_n = n
        while temp_n > 0:
            chunks.append(temp_n % 1000)
            temp_n //= 1000

        final_words = []
        for i in range(len(chunks) - 1, -1, -1):
            chunk = chunks[i]
            if chunk == 0: continue
            if i == 1 and chunk == 1:
                final_words.append("bin")
            else:
                chunk_text = process_three_digits(chunk)
                if chunk_text:
                    final_words.append(chunk_text)
                    if i > 0: final_words.append(scales[i])
        return "".join(final_words).strip()

    def number_to_ordinal_tr(self, n: int) -> str:
        if n in self.ORDINAL_EXCEPTIONS: return self.ORDINAL_EXCEPTIONS[n]
        base = self.number_to_turkish_words(n)
        
        vowels_in_base = [c for c in base if c in "aıoueiüö"]
        last_vowel = vowels_in_base[-1] if vowels_in_base else "i"
        
        if base.endswith(("a","ı","u","o","e","i","ü","ö")):
            suffix = "n" + ("cı" if last_vowel in "aı" else "ci" if last_vowel in "ei" else "cu" if last_vowel in "ou" else "cü")
        else:
            suffix = ("ıncı" if last_vowel in "aı" else "inci" if last_vowel in "ei" else "uncu" if last_vowel in "ou" else "üncü")
        return base + suffix

    def _process_pipeline(self, text: str) -> str:
        # 1. Unicode Normalization
        text = unicodedata.normalize("NFC", text)
        
        # 2. Expand Abbreviations
        if self.abbreviations:
            for pattern, replacement in self._ABBREV_PATTERNS:
                text = pattern.sub(replacement, text)
                
        # 3. Process Dates and Times
        text = re.sub(
            r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b",
            lambda m: f"{self.number_to_turkish_words(int(m.group(1)))} {self.MONTHS_TR.get(m.group(2).zfill(2),'')} {self.number_to_turkish_words(int(m.group(3)))}",
            text
        )
        text = re.sub(
            r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b",
            lambda m: f"{self.number_to_turkish_words(int(m.group(1)))}{self.number_to_turkish_words(int(m.group(2)))}",
            text
        )
        
        # 4. Clean Thousands Separator (e.g., 2.500 -> 2500)
        text = re.sub(r"(\d)\.(?=\d{3}(\b|\D))", r"\1", text)

        # 5. Expand Ordinals and Standard Numbers
        text = re.sub(
            r"\b(\d+)(?:\.(?!\d)|['’]?(?:inci|nci|uncu|üncü))",
            lambda m: self.number_to_ordinal_tr(int(m.group(1))),
            text, flags=re.IGNORECASE
        )
        text = re.sub(r"\b\d+\b", lambda m: self.number_to_turkish_words(int(m.group(0))), text)

        # 6. Separate and clean attached numbers
        text = re.sub(r'\b(\d+)([^\W\d_]+)', r'\1 \2', text)
        text = re.sub(r'\b([^\W\d_]+)(\d+)\b', r'\1 \2', text)
        text = re.sub(r"\b\d+\b", lambda m: self.number_to_turkish_words(int(m.group(0))), text)        
        text = re.sub(r"\d+", "", text)
        
        # 7. Whitespace cleanup
        return re.sub(r"\s+", " ", text).strip()


class TurkishNormalizer(BaseNormalizer):
    """
    Isolated, native TTS text normalizer for Turkish (tr).
    """
    def __init__(self, lang: str = "tr", extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        super().__init__(lang, extract, upper, wordify, abbreviations)
        
        self.whitelist = [
            'a', 'â', 'b', 'c', 'ç', 'd', 'e', 'ê', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'î', 
            'j', 'k', 'l', 'm', 'n', 'o', 'ô', 'ö', 'p', 'q', 'r', 's', 'ş', 't', 'u', 
            'û', 'ū', 'ü', 'v', 'w', 'x', 'y', 'z', '.', ',', '?', '!', ':', ';', "'", 
            '(', ')', '[', ']', ' ', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'ə', 'ñ', 'ŋ', 'ň', 'ý', 'ä', 'ë', 'ê', 'î', 'ô', 'û', 'ū', 'ă', 'ĕ', 'ś', 'ÿ', 'ž', 'ʻ', '’'
        ]
        self.allowed = set(self.whitelist)
       
        self.char_map = {
            "’": "'", "‘": "'", "`": "'", "´": "'",
            "“": '"', "”": '"', "«": '"', "»": '"',
            "–": "-", "—": "-", "−": "-", "~": "-",
            "…": "...", "\n": " ", "\t": " ",
        }

        self.lower_map = {
            ord('I'): 'ı', ord('İ'): 'i', ord('Ə'): 'ə',
            ord('Ñ'): 'ñ', ord('Ŋ'): 'ŋ', ord('Ý'): 'ý',
        }
        
        self.turkic_pattern = r"([a-zA-ZÇçĞğÎîıİiÖöÔôŞşÜüÛûÂâÊêƏəQqXxÑñŊŋÄäËëŽžŇňÝýŪūĂăĔĕŚśŸÿ]+)"
        
        self.whitespace_re = re.compile(r'\s+')
        self.ellipsis_find_re = re.compile(r'(?:\.\s*){2,}')
        self.punct_prefix_re = re.compile(r'([(),!?;:]|\.(?!\.))(?!(?<=[.,:])\d)([^\s])')
        self.punct_suffix_re = re.compile(r'([^\s])(?!(?<=\d)[.,:](?=\d))([!?;:]|[(),]|\.(?!\.))')

    def extract_graphemes(self, text: str) -> str:
        return " ".join(list(text.replace(" ", "")))

    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. Route through the MultilingualWordifier
        if self.wordify_mode:
            wordifier_router = MultilingualWordifier(text, language_code=self.lang, abbreviations=self.abbreviations_mode)
            if hasattr(wordifier_router.processor, 'normalized_text'):
                text = wordifier_router.processor.normalized_text
            elif hasattr(wordifier_router.processor, 'get_text'):
                text = wordifier_router.processor.get_text()

        # 2. Apply char map
        for char, replacement in self.char_map.items():
            text = text.replace(char, replacement)
        
        # 3. Turkish-safe lowercase
        text = text.translate(self.lower_map).lower()       
        
        # 4. Cleanup & Masking
        text = re.sub(r'([^\w\s\.])\1+', r'\1', text)
        text = self.whitespace_re.sub(' ', text)
        text = self.ellipsis_find_re.sub(' ^ ', text)
        
        # 5. Punctuation Splitting
        text = self.punct_prefix_re.sub(r'\1 \2', text)
        text = self.punct_suffix_re.sub(r'\1 \2', text)

        # 6. Unmasking & Quote Formatting
        text = text.replace('^', '...').replace('"', "''")
        text = self.whitespace_re.sub(' ', text)

        # 7. Whitelist Filter
        text = "".join(char for char in text if char in self.allowed).strip()
        
        # 8. Final Extraction & Casing
        if self.extract_mode:
            text = self.extract_graphemes(text)
        if self.upper_mode:
            text = text.replace('i', 'İ').replace('ı', 'I').upper()
            
        return text.strip()
        

# =======
# ENGLISH
# =======

class EnglishWordifier(BaseWordifier):
    """
    Isolated TTS text normalizer for English (en).
    """
    def _process_pipeline(self, text: str) -> str:
        # 1. Basic Whitespace Cleanup
        text = text.strip()
        
        # 2. English-Specific Abbreviations
        if self.abbreviations:
            text = text.replace("Dr.", "Doctor").replace("Mr.", "Mister").replace("St.", "Street")
            
        # 3. Currency & Numbers (Placeholder for regex expansion like $5 -> five dollars)
        text = text.replace("$", " dollars ")
        
        # 4. Final Whitespace Sanitization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
class EnglishNormalizer(BaseNormalizer):
    """
    Isolated TTS text normalizer for English (en).
    """
    def __init__(self, lang: str = "en", extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        super().__init__(lang, extract, upper, wordify, abbreviations)
        
        # 1. English-Specific Char Map
        self.whitelist = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:;'-()[] 0123456789")
        self.allowed = set(self.whitelist)
        
        self.whitespace_re = re.compile(r'\s+')
        self.punct_prefix_re = re.compile(r'([(),!?;:]|\.(?!\.))(?!(?<=[.,:])\d)([^\s])')
        self.punct_suffix_re = re.compile(r'([^\s])(?!(?<=\d)[.,:](?=\d))([!?;:]|[(),]|\.(?!\.))')

    def extract_graphemes(self, text: str) -> str:
        return " ".join(list(text.replace(" ", "")))

    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. Route through the MultilingualWordifier if wordification is enabled
        if self.wordify_mode:
            wordifier_router = MultilingualWordifier(text, language_code=self.lang, abbreviations=self.abbreviations_mode)
            # Safely extract text depending on how the underlying class stores it
            if hasattr(wordifier_router.processor, 'normalized_text'):
                text = wordifier_router.processor.normalized_text
            elif hasattr(wordifier_router.processor, 'get_text'):
                text = wordifier_router.processor.get_text()
                
        # 2. Standardize to lowercase
        text = text.lower()
        
        # 3. Punctuation Splitting (protecting numbers)
        text = self.punct_prefix_re.sub(r'\1 \2', text)
        text = self.punct_suffix_re.sub(r'\1 \2', text)
        
        # 4. Clean Double Spaces
        text = self.whitespace_re.sub(' ', text)
        
        # 5. Strip illegal characters not in the English whitelist
        text = "".join(char for char in text if char in self.allowed).strip()
        
        # 6. Apply final modes
        if self.extract_mode:
            text = self.extract_graphemes(text)
        if self.upper_mode:
            text = text.upper()
            
        return text.strip()
        
# =======
# SPANISH
# =======

class SpanishWordifier(BaseWordifier):
    """
    Isolated TTS text normalizer for Spanish (es).
    """
    def _process_pipeline(self, text: str) -> str:
        # 1. Basic Whitespace Cleanup
        text = text.strip()
        
        # 2. Spanish-Specific Abbreviations
        if self.abbreviations:
            text = (text.replace("Sr.", "Señor")
                        .replace("Sra.", "Señora")
                        .replace("Dr.", "Doctor")
                        .replace("Dra.", "Doctora")
                        .replace("Av.", "Avenida"))
            
        # 3. Currency Expansion
        text = text.replace("$", " dólares ").replace("€", " euros ")
        
        # 4. Final Whitespace Sanitization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class SpanishNormalizer(BaseNormalizer):
    """
    Isolated TTS text normalizer for Spanish (es).
    """
    def __init__(self, lang: str = "es", extract: bool = False, upper: bool = False, wordify: bool = False, abbreviations: bool = False):
        super().__init__(lang, extract, upper, wordify, abbreviations)
        
        # 1. Spanish-Specific Char Map (Includes á, é, í, ó, ú, ü, ñ, ¿, ¡)
        self.whitelist = list("abcdefghijklmnñopqrstuvwxyzáéíóúüABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚÜ.,!?:;'-()[] ¿¡0123456789")
        self.allowed = set(self.whitelist)
        
        self.whitespace_re = re.compile(r'\s+')
        self.punct_prefix_re = re.compile(r'([(),!?;:¿¡]|\.(?!\.))(?!(?<=[.,:])\d)([^\s])')
        self.punct_suffix_re = re.compile(r'([^\s])(?!(?<=\d)[.,:](?=\d))([!?;:¿¡]|[(),]|\.(?!\.))')

    def extract_graphemes(self, text: str) -> str:
        return " ".join(list(text.replace(" ", "")))

    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # 1. Route through the MultilingualWordifier if wordification is enabled
        if self.wordify_mode:
            wordifier_router = MultilingualWordifier(text, language_code=self.lang, abbreviations=self.abbreviations_mode)
            if hasattr(wordifier_router.processor, 'normalized_text'):
                text = wordifier_router.processor.normalized_text
            elif hasattr(wordifier_router.processor, 'get_text'):
                text = wordifier_router.processor.get_text()
                
        # 2. Standardize to lowercase
        text = text.lower()
        
        # 3. Punctuation Splitting (protecting numbers)
        text = self.punct_prefix_re.sub(r'\1 \2', text)
        text = self.punct_suffix_re.sub(r'\1 \2', text)
        
        # 4. Clean Double Spaces
        text = self.whitespace_re.sub(' ', text)
        
        # 5. Strip illegal characters not in the Spanish whitelist
        text = "".join(char for char in text if char in self.allowed).strip()
        
        # 6. Apply final modes
        if self.extract_mode:
            text = self.extract_graphemes(text)
        if self.upper_mode:
            text = text.upper()
            
        return text.strip()

# ------------------------------------------
# WORDIFIER AND NORMALIZER ROUTER & REGISTRY
# ------------------------------------------ 
      
WORDIFIER_REGISTRY = {
    'tr': TurkishWordifier,
    'en': EnglishWordifier,
    'es': SpanishWordifier
}

NORMALIZER_REGISTRY = {
    'tr': TurkishNormalizer,
    'en': EnglishNormalizer,
    'es': SpanishNormalizer
}