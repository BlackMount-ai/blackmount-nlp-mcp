"""Text cleaning — stopword removal, stemming, normalization pipeline."""

import re

from ._stopwords import ENGLISH_STOPWORDS
from .tokenize import word_tokenize


def remove_stopwords(text: str, language: str = "english") -> str:
    """Remove stopwords from text. Currently supports English.

    Args:
        text: Input text.
        language: Language for stopword list (default 'english').

    Returns:
        Text with stopwords removed.
    """
    if language != "english":
        # Only English supported; return as-is for other languages
        return text
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t.lower() not in ENGLISH_STOPWORDS]
    return " ".join(filtered)


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text."""
    return re.sub(r"[^\w\s]", "", text)


def remove_numbers(text: str) -> str:
    """Remove all numbers from text."""
    return re.sub(r"\d+", "", text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", "", text)


def remove_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def stem(word: str) -> str:
    """Porter stemmer implemented from scratch.

    Applies the Porter stemming algorithm's major rules to reduce
    words to their stems. Not perfect but good enough for most uses.
    """
    word = word.lower().strip()
    if len(word) <= 2:
        return word

    def _is_consonant(word: str, i: int) -> bool:
        if word[i] in "aeiou":
            return False
        if word[i] == "y":
            return i == 0 or not _is_consonant(word, i - 1)
        return True

    def _measure(stem: str) -> int:
        """Count VC sequences (the 'm' value in Porter's algorithm)."""
        if not stem:
            return 0
        count = 0
        i = 0
        # Skip initial consonants
        while i < len(stem) and _is_consonant(stem, i):
            i += 1
        while i < len(stem):
            # Skip vowels
            while i < len(stem) and not _is_consonant(stem, i):
                i += 1
            if i >= len(stem):
                break
            count += 1
            # Skip consonants
            while i < len(stem) and _is_consonant(stem, i):
                i += 1
        return count

    def _has_vowel(stem: str) -> bool:
        return any(not _is_consonant(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(word: str) -> bool:
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and _is_consonant(word, len(word) - 1)
        )

    def _cvc(word: str) -> bool:
        """Check if word ends with consonant-vowel-consonant (where last c != w,x,y)."""
        if len(word) < 3:
            return False
        return (
            _is_consonant(word, len(word) - 1)
            and not _is_consonant(word, len(word) - 2)
            and _is_consonant(word, len(word) - 3)
            and word[-1] not in "wxy"
        )

    # Step 1a: plurals and -ed/-ing
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif not word.endswith("ss") and word.endswith("s"):
        word = word[:-1]

    # Step 1b
    if word.endswith("eed"):
        stem_part = word[:-3]
        if _measure(stem_part) > 0:
            word = word[:-1]  # eed -> ee
    elif word.endswith("ed"):
        stem_part = word[:-2]
        if _has_vowel(stem_part):
            word = stem_part
            # Additional cleanup
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _cvc(word):
                word += "e"
    elif word.endswith("ing"):
        stem_part = word[:-3]
        if _has_vowel(stem_part):
            word = stem_part
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _cvc(word):
                word += "e"

    # Step 1c: y -> i
    if word.endswith("y") and _has_vowel(word[:-1]) and len(word) > 2:
        word = word[:-1] + "i"

    # Step 2: map double suffixes
    step2_map = {
        "ational": "ate", "tional": "tion", "enci": "ence",
        "anci": "ance", "izer": "ize", "abli": "able",
        "alli": "al", "entli": "ent", "eli": "e",
        "ousli": "ous", "ization": "ize", "ation": "ate",
        "ator": "ate", "alism": "al", "iveness": "ive",
        "fulness": "ful", "ousness": "ous", "aliti": "al",
        "iviti": "ive", "biliti": "ble",
    }
    for suffix, replacement in sorted(step2_map.items(), key=lambda x: -len(x[0])):
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if _measure(stem_part) > 0:
                word = stem_part + replacement
            break

    # Step 3
    step3_map = {
        "icate": "ic", "ative": "", "alize": "al",
        "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
    }
    for suffix, replacement in sorted(step3_map.items(), key=lambda x: -len(x[0])):
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if _measure(stem_part) > 0:
                word = stem_part + replacement
            break

    # Step 4: remove long suffixes
    step4_suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize",
    ]
    for suffix in sorted(step4_suffixes, key=lambda x: -len(x)):
        if word.endswith(suffix):
            stem_part = word[: -len(suffix)]
            if _measure(stem_part) > 1:
                if suffix == "ion":
                    if stem_part and stem_part[-1] in "st":
                        word = stem_part
                else:
                    word = stem_part
            break

    # Step 5a: remove trailing 'e'
    if word.endswith("e"):
        stem_part = word[:-1]
        m = _measure(stem_part)
        if m > 1 or (m == 1 and not _cvc(stem_part)):
            word = stem_part

    # Step 5b: -ll -> -l
    if word.endswith("ll") and _measure(word[:-1]) > 1:
        word = word[:-1]

    return word


def clean_text(text: str, steps: list[str] | None = None) -> str:
    """Configurable text cleaning pipeline.

    Args:
        text: Input text.
        steps: List of cleaning steps to apply, in order.
               If None, applies all steps in a sensible default order.

    Available steps:
        'html', 'urls', 'emails', 'numbers', 'punctuation',
        'stopwords', 'whitespace', 'lowercase'

    Returns:
        Cleaned text.
    """
    if steps is None:
        steps = [
            "html", "urls", "emails", "lowercase",
            "numbers", "punctuation", "stopwords", "whitespace",
        ]

    step_map = {
        "html": remove_html,
        "urls": remove_urls,
        "emails": remove_emails,
        "numbers": remove_numbers,
        "punctuation": remove_punctuation,
        "stopwords": remove_stopwords,
        "whitespace": normalize_whitespace,
        "lowercase": lowercase,
    }

    for step_name in steps:
        func = step_map.get(step_name)
        if func:
            text = func(text)

    return text
