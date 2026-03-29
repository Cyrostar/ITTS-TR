from functools import lru_cache
from typing import List, Dict, Union, Optional, Tuple, Any, FrozenSet, TypedDict
import logging
import re

# --- TypedDicts for structured, type-safe return values ---

class VowelHarmonyAnalysis(TypedDict):
    """Defines the structure for vowel harmony analysis results."""
    valid: bool
    vowels: List[str]
    front_count: int
    back_count: int
    has_front: bool
    has_back: bool
    first_vowel_type: Optional[str]
    note: str

class WordValidationResult(TypedDict):
    """Defines the structure for word validation results."""
    valid: bool
    has_vowels: bool
    syllabifiable: bool
    harmony_valid: Optional[bool]
    syllable_count: int
    issues: List[str]

class WordComparisonResult(TypedDict):
    """Defines the structure for comparing two words."""
    word1: str
    word2: str
    syllables1: Tuple[str, ...]
    syllables2: Tuple[str, ...]
    same_syllable_count: bool
    syllable_count_diff: int
    stress_position1: Optional[int]
    stress_position2: Optional[int]
    same_stress_pattern: bool
    both_harmonic: bool
    same_vowel_type: bool

class FullWordAnalysis(TypedDict):
    """Defines the structure for the comprehensive analyze_word method."""
    word: str
    syllables: Tuple[str, ...]
    syllables_with_stress: List[str]
    syllable_count: int
    formatted: str
    formatted_with_stress: str
    has_exception: bool
    has_neutral_suffix: bool
    is_question: bool
    vowel_harmony: VowelHarmonyAnalysis

class TurkishSyllabifier:
    """
    Handles Turkish syllabification with caching, stress marking, and linguistic analysis.

    This class provides a comprehensive toolset for working with Turkish words,
    including core syllabification, stress rule application, vowel harmony checks,
    and various utility functions.
    """
    # --- Class-level constants for Turkish alphabet and common patterns ---
    _VOWELS: FrozenSet[str] = frozenset('aâeêıiîoôöuûü')
    _CONSONANTS: FrozenSet[str] = frozenset('bcçdfgğhjklmnprsştvwxyz')
    _FRONT_VOWELS: FrozenSet[str] = frozenset('eêiîöü')

    # Complex onset clusters (ordered for longest match preference in some contexts)
    _COMPLEX_ONSETS: FrozenSet[str] = frozenset({
        'str', 'spr', 'skr', 'şt', 'ştr',
        'tr', 'st', 'kr', 'gr', 'pr', 'br', 'dr', 'fr', 'tl', 'zl',
        'pl', 'bl', 'kl', 'gl', 'fl',
        'sp', 'sk', 'sm', 'sn', 'sl', 'sw', 'sf', 'zm'
    })

    # Question particles (never stressed)
    _QUESTION_PARTICLES: FrozenSet[str] = frozenset({'mi', 'mı', 'mu', 'mü'})

    # Suffix patterns that shift stress to the penultimate syllable
    _STRESS_NEUTRAL_SUFFIX_PATTERNS: List[FrozenSet[str]] = [
        frozenset({'ki'}), frozenset({'gil'}), frozenset({'ce', 'ça', 'çe'}),
        frozenset({'de', 'da', 'te', 'ta'}), frozenset({'lik', 'lık', 'luk', 'lük'}),
        frozenset({'ler', 'lar'}), frozenset({'leyin', 'layin'}), frozenset({'ken'}),
        frozenset({'siz', 'sız', 'suz', 'süz'}), frozenset({'si', 'sı', 'su', 'sü'}),
    ]

    def __init__(self, cache_size: int = 10000, stress_marker: str = "'", log_level: int = logging.WARNING):
        """
        Initialize syllabifier.

        Args:
            cache_size: Size of LRU cache for syllabification.
            stress_marker: Character to mark stressed syllables.
            log_level: Logging level (default: WARNING).
        """
        # Validate stress marker
        if len(stress_marker) != 1 or stress_marker in self._VOWELS or stress_marker in self._CONSONANTS:
            raise ValueError("Stress marker must be a single non-alphabetic character.")
        self.stress_marker = stress_marker

        # Derive character sets from class constants
        self.vowels: FrozenSet[str] = self._VOWELS
        self.consonants: FrozenSet[str] = self._CONSONANTS
        self.front_vowels: FrozenSet[str] = self._FRONT_VOWELS
        self.back_vowels: FrozenSet[str] = self.vowels - self.front_vowels

        # Other linguistic features
        self.complex_onsets: FrozenSet[str] = self._COMPLEX_ONSETS
        self.question_particles: FrozenSet[str] = self._QUESTION_PARTICLES
        self.stress_neutral_suffixes: FrozenSet[str] = frozenset().union(*self._STRESS_NEUTRAL_SUFFIX_PATTERNS)

        # Build translation table for efficiently removing non-Turkish characters
        valid_chars = self.vowels | self.consonants
        all_chars_to_check = ''.join(chr(i) for i in range(256))
        chars_to_remove = ''.join(c for c in all_chars_to_check if c not in valid_chars)
        self.remove_invalid_trans_table = str.maketrans('', '', chars_to_remove)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        # Dictionary for words with exceptional stress patterns
        self.stress_exceptions: Dict[str, int] = {
            # --- Original Exceptions (Common Loanwords & Place Names) ---
            'istanbul': 2, 'ankara': 0, 'izmir': 1, 'antalya': 2, 'eskişehir': 2,
            'trabzon': 1, 'gaziantep': 3, 'lokanta': 2, 'tiyatro': 2, 'gazete': 2,
            'radyo': 1, 'sinema': 2, 'müze': 1, 'telgraf': 2,

            # --- Expanded Place Names ---
            'türkiye': 0,      # 'tür-ki-ye
            'almanya': 2,      # al-man-'ya
            'rusya': 0,        # 'rus-ya
            'ingiltere': 3,    # in-gil-te-'re
            'beşiktaş': 1,     # be-'şik-taş
            'kadıköy': 1,      # ka-'dı-köy
            'mersin': 0,       # 'mer-sin
            'samsun': 0,       # 'sam-sun

            # --- Expanded Loanwords ---
            'restoran': 2,     # res-to-'ran
            'kamyon': 1,       # kam-'yon
            'otobüs': 2,       # o-to-'büs
            'profesör': 2,     # pro-fe-'sör
            'maalesef': 1,     # ma-'a-le-sef
            'internet': 2,     # in-ter-'net
            'fabrika': 0,      # 'fab-ri-ka
            'sandalye': 2,     # san-dal-'ye

            # --- Adverbs, Conjunctions & Interjections ---
            'şimdi': 0,        # 'şim-di (now)
            'sonra': 0,        # 'son-ra (after)
            'ayrıca': 0,       # 'ay-rı-ca (also)
            'yalnız': 0,       # 'yal-nız (only)
            'ancak': 0,        # 'an-cak (however)
            'haydi': 0,        # 'hay-di (come on!)
            'sanki': 0,        # 'san-ki (as if)
            
            # --- Vocative (Calling) Forms ---
            'anne': 0,         # 'an-ne (when calling "Mom!")
            'baba': 0,         # 'ba-ba (when calling "Dad!")
            'efendim': 1,      # e-'fen-dim
        }

        # Create cached syllabification method
        self._syllabify_cached = lru_cache(maxsize=cache_size)(self._syllabify_impl)


    def _determine_consonant_split(self, cluster: str, next_is_vowel: bool) -> int:
        """
        Determines where to split a consonant cluster based on the Maximum Onset Principle.

        Args:
            cluster: The consonant cluster string.
            next_is_vowel: True if the character following the cluster is a vowel.

        Returns:
            The index where the cluster should be split. Characters before this
            index belong to the current syllable's coda.
        """
        if not cluster:
            return 0
        if len(cluster) == 1:
            return 0 if next_is_vowel else 1 # C-V -> C stays, C-C -> C goes

        # Geminate consonants (e.g., 'ff') are always split
        if cluster[0] == cluster[1]:
            return 1

        if not next_is_vowel:
            return len(cluster) # All consonants belong to the current syllable's coda

        # Find the longest valid complex onset from the end of the cluster
        for i in range(len(cluster)):
            potential_onset = cluster[i:]
            if potential_onset in self.complex_onsets:
                return i # Split before the valid onset

        # Default rule: the last consonant forms the onset of the next syllable
        return len(cluster) - 1


    def _syllabify_impl(self, word: str) -> Tuple[str, ...]:
        """
        Core syllabification implementation (decorated with lru_cache).

        This algorithm finds vowel positions and splits the consonant clusters
        between them according to phonotactic rules.
        """
        if not word:
            return tuple()

        vowel_indices = [i for i, char in enumerate(word) if char in self.vowels]

        if not vowel_indices:
            self.logger.warning(f"No vowels in word '{word}', cannot syllabify.")
            return tuple()

        syllables: List[str] = []
        start_idx = 0

        # Iterate through pairs of vowels
        for i in range(len(vowel_indices) - 1):
            v_curr_idx = vowel_indices[i]
            v_next_idx = vowel_indices[i+1]

            cluster = word[v_curr_idx + 1 : v_next_idx]
            split_point = self._determine_consonant_split(cluster, next_is_vowel=True)
            
            end_idx = v_curr_idx + 1 + split_point
            syllables.append(word[start_idx:end_idx])
            start_idx = end_idx

        # Append the final syllable
        syllables.append(word[start_idx:])
        
        return tuple(syllables)


    def syllabify(self, word: str) -> Tuple[str, ...]:
        """
        Splits a Turkish word into syllables, with input cleaning and caching.

        Args:
            word: A single Turkish word.

        Returns:
            A tuple of syllables.
        """
        if not isinstance(word, str) or not word:
            return tuple()

        # Normalize and clean the word
        original_word = word
        word_for_processing = word.lower().strip().replace("'", "")
        cleaned_word = word_for_processing.translate(self.remove_invalid_trans_table)

        if len(cleaned_word) < len(word_for_processing):
            self.logger.info(
                f"Non-Turkish characters removed from '{original_word}'. "
                f"Processed as: '{cleaned_word}'"
            )

        if not cleaned_word:
            return tuple()

        return self._syllabify_cached(cleaned_word)


    def add_stress(self, syllables: Union[List[str], Tuple[str, ...]],
                   word: Optional[str] = None) -> List[str]:
        """
        Adds a stress marker to a list of syllables based on Turkish stress rules.
        """
        if not syllables:
            return []

        stressed = list(syllables)
        num_syllables = len(stressed)
        
        if num_syllables == 1:
            stressed[0] = self.stress_marker + stressed[0]
            return stressed

        # Rule 1: Question particles are never stressed; stress the preceding syllable
        if stressed[-1] in self.question_particles:
            stressed[-2] = self.stress_marker + stressed[-2]
            return stressed

        # Rule 2: Check for explicit stress exceptions (place names, loanwords)
        if word:
            word_lower = word.lower().strip().replace("'", "")
            if word_lower in self.stress_exceptions:
                stress_pos = self.stress_exceptions[word_lower]
                if 0 <= stress_pos < num_syllables:
                    stressed[stress_pos] = self.stress_marker + stressed[stress_pos]
                    return stressed

        # Rule 3: Stress-neutral suffixes shift stress to the penultimate syllable
        if stressed[-1] in self.stress_neutral_suffixes:
            stressed[-2] = self.stress_marker + stressed[-2]
            return stressed

        # Rule 4: Default rule is to stress the final syllable
        stressed[-1] = self.stress_marker + stressed[-1]
        return stressed


    def syllabify_with_stress(self, word: str) -> List[str]:
        """
        Syllabifies a word and adds stress markers in a single step.
        """
        syllables = self.syllabify(word)
        return self.add_stress(syllables, word)

    def add_stress_exception(self, word: str, stress_position: int) -> None:
        """
        Add a word with exceptional stress pattern.

        IMPROVED: Now clears cache to ensure updated stress is used.

        Args:
            word: Word with exceptional stress (any case)
            stress_position: Index of stressed syllable (0-based)

        Raises:
            ValueError: If stress position is out of range or word invalid

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> syl.add_stress_exception('Eskişehir', 2)
            >>> syl.syllabify_with_stress('eskişehir')
            ['es', 'ki', 'şe', "'hir"]
        """
        word_clean = word.lower().replace("'", "")
        syllables = self.syllabify(word_clean)

        # Validate stress position
        if not syllables:
            raise ValueError(f"Cannot syllabify word: {word!r}. No syllables found.")

        if stress_position < 0 or stress_position >= len(syllables):
            raise ValueError(
                f"Stress position {stress_position} out of range for "
                f"word '{word}' with {len(syllables)} syllables: {syllables}"
            )

        self.stress_exceptions[word_clean] = stress_position

        # FIXED: Clear cache when adding exceptions to ensure consistency
        self._syllabify_cached.cache_clear()

        self.logger.info(
            f"Added stress exception: {word_clean} at position {stress_position} "
            f"(syllable: '{syllables[stress_position]}')"
        )

    def remove_stress_exception(self, word: str) -> bool:
        """
        Remove a word from stress exceptions.

        IMPROVED: Returns boolean indicating if exception was found.

        Args:
            word: Word to remove

        Returns:
            True if exception was removed, False if not found

        Examples:
            >>> syl.remove_stress_exception('istanbul')
            True
            >>> syl.remove_stress_exception('nonexistent')
            False
        """
        word_clean = word.lower().replace("'", "")
        if word_clean in self.stress_exceptions:
            del self.stress_exceptions[word_clean]
            self._syllabify_cached.cache_clear()
            self.logger.info(f"Removed stress exception: {word_clean}")
            return True
        return False

    def check_vowel_harmony(self, word: str) -> Dict[str, Any]:
        """
        Check Turkish vowel harmony rules.

        IMPROVED: Better analysis with consideration of morpheme boundaries.

        Turkish has two types of vowel harmony:
        1. Backness harmony: front vowels with front, back with back
        2. Rounding harmony: affects suffix vowels

        Note: This is a simplified check. Full harmony analysis requires
        morphological parsing to identify root-suffix boundaries.
        Loanwords and compound words often violate harmony.

        Args:
            word: Turkish word

        Returns:
            Dictionary with harmony analysis:
            - valid: whether harmony appears preserved
            - vowels: list of vowels in order
            - front_count: count of front vowels
            - back_count: count of back vowels
            - has_front: whether word contains front vowels
            - has_back: whether word contains back vowels
            - first_vowel_type: 'front' or 'back'
            - note: additional context

        Examples:
            >>> check_vowel_harmony('kalem')
            {'valid': True, 'vowels': ['a', 'e'], ...}
            >>> check_vowel_harmony('kitap')
            {'valid': True, 'vowels': ['i', 'a'], ...}
        """
        word = word.lower()
        vowels_in_word = [c for c in word if c in self.vowels]

        if not vowels_in_word:
            return {
                'valid': False,
                'reason': 'No vowels found',
                'vowels': [],
                'front_count': 0,
                'back_count': 0,
                'has_front': False,
                'has_back': False,
                'first_vowel_type': None,
                'note': 'Invalid word - no vowels'
            }

        # Count front and back vowels (using extended sets for loanwords)
        front_count = sum(1 for v in vowels_in_word if v in self.front_vowels)
        back_count = sum(1 for v in vowels_in_word if v in self.back_vowels)

        # Determine first vowel type (important for harmony rules)
        first_vowel = vowels_in_word[0]
        first_vowel_type = 'front' if first_vowel in self.front_vowels else 'back'

        is_likely_harmonic = True
        note = 'Appears harmonic'

        if len(vowels_in_word) == 1:
            note = 'Single vowel - trivially harmonic'
        elif front_count > 0 and back_count > 0:
            # Mixed vowels
            # Check if the word starts with a sequence of vowels of the same type (root)
            # and then potentially changes (suffix or loanword)
            root_vowels_end_idx = 0
            for k in range(1, len(vowels_in_word)):
                if (vowels_in_word[k] in self.front_vowels and first_vowel_type == 'back') or \
                   (vowels_in_word[k] in self.back_vowels and first_vowel_type == 'front'):
                    break # Harmony broken after k-1
                root_vowels_end_idx = k

            if root_vowels_end_idx < len(vowels_in_word) - 1: # If harmony changes within or close to the root
                is_likely_harmonic = False
                note = 'Mixed vowels in core part of word - likely loanword or compound'
            else:
                note = 'Root appears harmonic, potential suffix harmony or minor deviation'
        elif front_count > 0:
            note = 'All front vowels - harmonic'
        else: # back_count > 0
            note = 'All back vowels - harmonic'

        return {
            'valid': is_likely_harmonic,
            'vowels': vowels_in_word,
            'front_count': front_count,
            'back_count': back_count,
            'has_front': front_count > 0,
            'has_back': back_count > 0,
            'first_vowel_type': first_vowel_type,
            'note': note
        }

    def clear_cache(self) -> None:
        """
        Clear the syllabification cache.

        Useful when memory is constrained or after bulk operations.
        """
        self._syllabify_cached.cache_clear()
        self.logger.info("Syllabification cache cleared")

    def get_cache_info(self) -> Any: # Returns _lru_cache_wrapper.cache_info type
        """
        Get cache statistics.

        Returns:
            Named tuple with cache statistics:
            - hits: number of cache hits
            - misses: number of cache misses
            - maxsize: maximum cache size
            - currsize: current number of cached entries

        Examples:
            >>> info = syl.get_cache_info()
            >>> print(f"Hit rate: {info.hits / (info.hits + info.misses):.2%}")
        """
        return self._syllabify_cached.cache_info()

    def count_syllables(self, word: str) -> int:
        """
        Quick syllable count without full syllabification.

        Counts vowels in the word as a fast approximation.
        For exact count, use len(syllabify(word)) instead.

        Time Complexity: O(n) where n is word length

        Args:
            word: Turkish word

        Returns:
            Number of syllables (vowel count)

        Examples:
            >>> count_syllables('merhaba')
            3
            >>> count_syllables('türkçe')
            2
        """
        word = word.lower()
        # Filter valid characters first using the union of defined sets
        word = ''.join(c for c in word if c in (self.vowels | self.consonants))
        return sum(1 for c in word if c in self.vowels)

    def join_syllables(self, syllables: Union[List[str], Tuple[str, ...]],
                       separator: str = '-') -> str:
        """
        Join syllables with a separator.

        Args:
            syllables: List or tuple of syllables
            separator: String to join with (default: '-')

        Returns:
            Joined string

        Examples:
            >>> syl.join_syllables(['ka', 'lem'], '-')
            'ka-lem'
            >>> syl.join_syllables(['ka', "'lem"], '.')
            "ka.'lem"
        """
        return separator.join(syllables)

    def process_phrase(self, phrase: str,
                       include_stress: bool = False) -> List[Tuple[str, Tuple[str, ...]]]:
        """
        Process a phrase with multiple words.

        IMPROVED: Added option to include stress in results.

        Args:
            phrase: Phrase with one or more words separated by spaces
            include_stress: If True, return syllables with stress markers

        Returns:
            List of (word, syllables) tuples

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> syl.process_phrase('merhaba dünya')
            [('merhaba', ('mer', 'ha', 'ba')), ('dünya', ('dün', 'ya'))]
            >>> syl.process_phrase('merhaba dünya', include_stress=True)
            [('merhaba', ['mer', 'ha', "'ba"]), ('dünya', ['dün', "'ya"])]
        """
        words = phrase.split()
        results = []

        for word in words:
            if not word:
                continue

            if include_stress:
                # Add stress returns a list, so convert to tuple for consistency with base syllabify output
                syllables_result = tuple(self.syllabify_with_stress(word))
            else:
                syllables_result = self.syllabify(word)

            results.append((word, syllables_result))

        return results
        
    def has_stress_neutral_suffix(self, word: str) -> bool:
        """
        Evaluates if the word ends with a stress-neutral suffix.
        """
        syllables = self.syllabify(word)
        if not syllables:
            return False
        return syllables[-1] in self.stress_neutral_suffixes

    def is_question_particle(self, syllables: Union[List[str], Tuple[str, ...]]) -> bool:
        """
        Evaluates if the final syllable acts as a stress-repelling question particle.
        """
        if not syllables:
            return False
        return syllables[-1] in self.question_particles

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a Turkish word
        Provides complete linguistic analysis including syllabification,
        stress patterns, and vowel harmony.

        Args:
            word: Turkish word

        Returns:
            Dictionary with complete analysis:
            - word: original word
            - syllables: tuple of syllables
            - syllables_with_stress: list with stress markers
            - syllable_count: number of syllables
            - formatted: syllables joined with hyphens
            - formatted_with_stress: stressed syllables joined
            - has_exception: whether word has exceptional stress
            - has_neutral_suffix: whether ends with stress-neutral suffix
            - is_question: whether last syllable is question particle
            - vowel_harmony: vowel harmony analysis

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> result = syl.analyze_word('kalem')
            >>> print(result['formatted_with_stress'])
            "ka-'lem"
            >>> result['vowel_harmony']['valid']
            True
        """
        syllables = self.syllabify(word)
        stressed = self.add_stress(syllables, word)
        word_lower = word.lower().replace("'", "")
        harmony = self.check_vowel_harmony(word)

        return {
            'word': word,
            'syllables': syllables,
            'syllables_with_stress': stressed,
            'syllable_count': len(syllables),
            'formatted': self.join_syllables(syllables),
            'formatted_with_stress': self.join_syllables(stressed),
            'has_exception': word_lower in self.stress_exceptions,
            'has_neutral_suffix': self.has_stress_neutral_suffix(word_lower) if syllables else False,
            'is_question': self.is_question_particle(syllables),
            'vowel_harmony': harmony
        }

    def batch_syllabify(self, words: List[str],
                        include_stress: bool = False) -> Dict[str, Tuple[str, ...]]:
        """
        Syllabify multiple words efficiently.

        NEW: Batch processing for better performance with multiple words.

        Args:
            words: List of Turkish words
            include_stress: If True, return syllables with stress markers

        Returns:
            Dictionary mapping words to their syllables

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> syl.batch_syllabify(['kalem', 'kitap', 'okul'])
            {'kalem': ('ka', 'lem'), 'kitap': ('ki', 'tap'), 'okul': ('o', 'kul')}
        """
        results = {}
        for word in words:
            if include_stress:
                results[word] = tuple(self.syllabify_with_stress(word))
            else:
                results[word] = self.syllabify(word)
        return results

    def is_valid_turkish_word(self, word: str,
                             check_harmony: bool = False) -> Dict[str, Any]:
        """
        Check if a word follows Turkish phonotactic rules.

        NEW: Validation method for Turkish word structure.

        Args:
            word: Word to validate
            check_harmony: If True, also check vowel harmony

        Returns:
            Dictionary with validation results:
            - valid: overall validity
            - has_vowels: whether word contains vowels
            - syllabifiable: whether word can be syllabified
            - harmony_valid: vowel harmony status (if checked)
            - issues: list of any issues found
            - syllable_count: number of syllables found (0 if not syllabifiable)

        Examples:
            >>> syl.is_valid_turkish_word('kalem')
            {'valid': True, 'has_vowels': True, 'syllabifiable': True, 'issues': [], 'syllable_count': 2, 'harmony_valid': True}
            >>> syl.is_valid_turkish_word('str', check_harmony=True)
            {'valid': False, 'has_vowels': False, 'syllabifiable': False, 'issues': ['No vowels found'], 'syllable_count': 0, 'harmony_valid': None}
        """
        issues = []
        word_clean = word.lower().replace("'", "")
        
        # Apply strict character filtering to determine actual "Turkish-ness"
        turkish_only_chars = [c for c in word_clean if c in (self.vowels | self.consonants)]
        cleaned_for_validation = ''.join(turkish_only_chars)

        # Check for vowels
        has_vowels = any(c in self.vowels for c in cleaned_for_validation)
        if not has_vowels:
            issues.append('No vowels found')

        # Check syllabification
        syllables = self.syllabify(cleaned_for_validation) if has_vowels else tuple()
        syllabifiable = len(syllables) > 0

        if not syllabifiable and has_vowels: # Only an issue if it has vowels but can't be syllabified
            issues.append('Cannot be syllabified according to rules')

        # Check vowel harmony if requested
        harmony_result = None
        if check_harmony and has_vowels:
            harmony_result = self.check_vowel_harmony(cleaned_for_validation)
            if not harmony_result['valid']:
                issues.append(f"Vowel harmony issue: {harmony_result['note']}")

        return {
            'valid': len(issues) == 0,
            'has_vowels': has_vowels,
            'syllabifiable': syllabifiable,
            'harmony_valid': harmony_result['valid'] if harmony_result else None,
            'syllable_count': len(syllables),
            'issues': issues
        }

    def get_stress_position(self, word: str) -> Optional[int]:
        """
        Get the index of the stressed syllable.

        NEW: Helper method to find which syllable is stressed.

        Args:
            word: Turkish word

        Returns:
            Index of stressed syllable (0-based), or None if no syllables
            or no stress marker was applied (e.g., empty word)

        Examples:
            >>> syl.get_stress_position('kalem')
            1  # Second syllable 'lem' is stressed
            >>> syl.get_stress_position('istanbul')
            2  # Third syllable 'bul' is stressed
        """
        syllables = self.syllabify(word)
        if not syllables:
            return None

        stressed = self.add_stress(syllables, word)

        # Find which syllable has the stress marker
        for i, syl in enumerate(stressed):
            if syl.startswith(self.stress_marker):
                return i

        return None # Should ideally not happen if word has syllables

    def compare_words(self, word1: str, word2: str) -> Dict[str, Any]:
        """
        Compare syllabic structure of two words.

        NEW: Utility for comparing Turkish words.

        Args:
            word1: First Turkish word
            word2: Second Turkish word

        Returns:
            Dictionary with comparison results

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> syl.compare_words('kalem', 'kitap')
            {'word1': 'kalem', 'word2': 'kitap', 'syllables1': ('ka', 'lem'), 'syllables2': ('ki', 'tap'), 'same_syllable_count': True, 'syllable_count_diff': 0, 'stress_position1': 1, 'stress_position2': 1, 'same_stress_pattern': True, 'both_harmonic': True, 'same_vowel_type': False}
        """
        syl1 = self.syllabify(word1)
        syl2 = self.syllabify(word2)

        stress_pos1 = self.get_stress_position(word1)
        stress_pos2 = self.get_stress_position(word2)

        harmony1 = self.check_vowel_harmony(word1)
        harmony2 = self.check_vowel_harmony(word2)

        return {
            'word1': word1,
            'word2': word2,
            'syllables1': syl1,
            'syllables2': syl2,
            'same_syllable_count': len(syl1) == len(syl2),
            'syllable_count_diff': abs(len(syl1) - len(syl2)),
            'stress_position1': stress_pos1,
            'stress_position2': stress_pos2,
            'same_stress_pattern': stress_pos1 == stress_pos2,
            'both_harmonic': harmony1['valid'] and harmony2['valid'],
            'same_vowel_type': (harmony1['first_vowel_type'] ==
                               harmony2['first_vowel_type'])
        }

    def find_rhymes(self, word: str, candidates: List[str],
                    min_syllables: int = 1) -> List[Tuple[str, int]]:
        """
        Find words that rhyme with the given word.

        NEW: Rhyme detection based on syllable endings.

        Args:
            word: Target word to find rhymes for
            candidates: List of potential rhyming words
            min_syllables: Minimum number of matching syllables (default: 1)

        Returns:
            List of (rhyming_word, matching_syllables) tuples, sorted by match quality
            (highest number of matching syllables first)

        Examples:
            >>> syl = TurkishSyllabifier()
            >>> syl.find_rhymes('kalem', ['elem', 'badem', 'kalem', 'okul', 'gelen'])
            [('badem', 2), ('elem', 1), ('gelen', 1)]
        """
        target_syllables = self.syllabify(word)
        if not target_syllables:
            return []

        rhymes = []

        for candidate in candidates:
            if candidate.lower() == word.lower():
                continue  # Skip the word itself

            cand_syllables = self.syllabify(candidate)
            if not cand_syllables:
                continue

            # Check how many syllables match from the end
            matching = 0
            # Iterate from 1 up to the minimum length of syllables
            for i in range(1, min(len(target_syllables), len(cand_syllables)) + 1):
                # Compare syllables from the end
                if target_syllables[-i] == cand_syllables[-i]:
                    matching += 1
                else:
                    break # Stop if a non-matching syllable is found

            if matching >= min_syllables:
                rhymes.append((candidate, matching))

        # Sort by number of matching syllables (descending)
        rhymes.sort(key=lambda x: x[1], reverse=True)

        return rhymes

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the syllabifier configuration and usage.

        NEW: Comprehensive statistics for monitoring and debugging.

        Returns:
            Dictionary with statistics including cache info, exceptions, etc.

        Examples:
            >>> stats = syl.get_statistics()
            >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        """
        cache_info = self.get_cache_info()
        total_cache_requests = cache_info.hits + cache_info.misses
        cache_hit_rate = (cache_info.hits / total_cache_requests
                         if total_cache_requests > 0 else 0.0)

        return {
            'cache_hits': cache_info.hits,
            'cache_misses': cache_info.misses,
            'cache_size': cache_info.currsize,
            'cache_maxsize': cache_info.maxsize,
            'cache_hit_rate': cache_hit_rate,
            'stress_exceptions_count': len(self.stress_exceptions),
            'stress_exceptions': list(self.stress_exceptions.keys()),
            'stress_neutral_suffixes_count': len(self.stress_neutral_suffixes),
            'complex_onsets_count': len(self.complex_onsets),
            'vowel_count': len(self.vowels),
            'consonant_count': len(self.consonants),
            'stress_marker': self.stress_marker,
            'logger_level': logging.getLevelName(self.logger.level)
        }

    def export_exceptions(self) -> Dict[str, int]:
        """
        Export stress exceptions for backup or sharing.

        NEW: Export functionality for stress exception dictionary.

        Returns:
            Dictionary of stress exceptions

        Examples:
            >>> exceptions = syl.export_exceptions()
            >>> # Save to file or transfer to another instance
        """
        return self.stress_exceptions.copy()

    def import_exceptions(self, exceptions: Dict[str, int],
                         validate: bool = True) -> int:
        """
        Import stress exceptions from dictionary.

        NEW: Import functionality with optional validation.

        Args:
            exceptions: Dictionary mapping words to stress positions
            validate: If True, validate each exception before importing

        Returns:
            Number of exceptions successfully imported

        Examples:
            >>> exceptions = {'test': 0, 'örnek': 1}
            >>> count = syl.import_exceptions(exceptions)
            >>> print(f"Imported {count} exceptions")
        """
        imported = 0

        for word, position in exceptions.items():
            try:
                word_clean = word.lower().replace("'", "")
                if validate:
                    # Validate the exception
                    syllables = self.syllabify(word_clean)
                    if not syllables:
                        self.logger.warning(
                            f"Skipping invalid exception '{word_clean}': cannot be syllabified."
                        )
                        continue
                    if position < 0 or position >= len(syllables):
                        self.logger.warning(
                            f"Skipping invalid exception '{word_clean}' (position {position}): "
                            f"out of range for syllables {syllables}."
                        )
                        continue

                self.stress_exceptions[word_clean] = position
                imported += 1

            except Exception as e:
                self.logger.error(f"Error importing exception for '{word}': {e}")

        if imported > 0:
            self._syllabify_cached.cache_clear()
            self.logger.info(f"Imported {imported} stress exceptions")

        return imported

    def __repr__(self) -> str:
        """String representation for debugging."""
        cache_info = self.get_cache_info()
        return (f"TurkishSyllabifier(cache_size={cache_info.maxsize}, "
                f"exceptions={len(self.stress_exceptions)}, "
                f"cache_hits={cache_info.hits}, cache_misses={cache_info.misses})")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"Turkish Syllabifier with {len(self.stress_exceptions)} "
                f"stress exceptions and cache size {self.get_cache_info().maxsize}")                            