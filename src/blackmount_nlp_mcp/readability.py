"""Readability scoring — Flesch, Gunning Fog, Coleman-Liau, ARI, SMOG."""

import math
import re

from .tokenize import sentence_tokenize, word_tokenize


def syllable_count(word: str) -> int:
    """Estimate syllable count using heuristics.

    Rules:
    - Count vowel groups
    - Subtract silent 'e' at end
    - Handle special endings (-le, -es, -ed)
    - Minimum 1 syllable per word
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Common words with tricky syllable counts
    _OVERRIDES = {
        "the": 1, "are": 1, "were": 1, "where": 1, "there": 1,
        "here": 1, "fire": 1, "hire": 1, "wire": 1, "tire": 1,
        "desire": 2, "entire": 2, "require": 2, "inspire": 2,
        "people": 2, "every": 3, "different": 3, "interest": 3,
        "business": 3, "area": 3, "idea": 3, "real": 1,
        "being": 2, "doing": 2, "going": 2, "having": 2,
        "maybe": 2, "create": 2, "creative": 3, "created": 3,
    }

    if word in _OVERRIDES:
        return _OVERRIDES[word]

    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for i, char in enumerate(word):
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    # Silent 'e' at end (but not "le")
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1

    # Endings that don't add syllables
    if word.endswith("es") and count > 1:
        # Words like "makes", "takes" — silent es
        if len(word) > 3 and word[-3] not in "sx" and word[-3:-1] != "sh" and word[-3:-1] != "ch":
            count -= 1

    if word.endswith("ed") and count > 1:
        # "ed" is silent unless preceded by t or d
        if len(word) > 3 and word[-3] not in "td":
            count -= 1

    # "-tion", "-sion" = 1 syllable for that part
    # Already handled by vowel grouping

    return max(1, count)


def _text_stats(text: str) -> tuple[int, int, int, int]:
    """Return (num_words, num_sentences, num_syllables, num_chars).

    Characters counted are only letters (for Coleman-Liau / ARI).
    """
    sentences = sentence_tokenize(text)
    words = word_tokenize(text)
    # Filter to actual words (not punctuation)
    actual_words = [w for w in words if re.match(r"[A-Za-z]", w)]

    num_sentences = max(len(sentences), 1)
    num_words = max(len(actual_words), 1)
    num_syllables = sum(syllable_count(w) for w in actual_words)
    num_chars = sum(len(w) for w in actual_words)

    return num_words, num_sentences, num_syllables, num_chars


def _count_complex_words(text: str) -> int:
    """Count words with 3+ syllables (for Gunning Fog)."""
    words = word_tokenize(text)
    actual_words = [w for w in words if re.match(r"[A-Za-z]", w)]
    return sum(1 for w in actual_words if syllable_count(w) >= 3)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score.

    206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

    Score interpretation:
    - 90-100: Very easy (5th grade)
    - 80-89: Easy (6th grade)
    - 70-79: Fairly easy (7th grade)
    - 60-69: Standard (8th-9th grade)
    - 50-59: Fairly difficult (10th-12th grade)
    - 30-49: Difficult (college)
    - 0-29: Very confusing (graduate)
    """
    num_words, num_sentences, num_syllables, _ = _text_stats(text)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return round(score, 2)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level.

    0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    """
    num_words, num_sentences, num_syllables, _ = _text_stats(text)
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return round(max(0, grade), 2)


def gunning_fog(text: str) -> float:
    """Gunning Fog Index.

    0.4 * ((words/sentences) + 100 * (complex_words/words))
    """
    num_words, num_sentences, _, _ = _text_stats(text)
    complex_count = _count_complex_words(text)
    fog = 0.4 * ((num_words / num_sentences) + 100 * (complex_count / num_words))
    return round(max(0, fog), 2)


def coleman_liau(text: str) -> float:
    """Coleman-Liau Index.

    0.0588 * L - 0.296 * S - 15.8
    where L = avg letters per 100 words, S = avg sentences per 100 words
    """
    num_words, num_sentences, _, num_chars = _text_stats(text)
    l_val = (num_chars / num_words) * 100
    s_val = (num_sentences / num_words) * 100
    cli = 0.0588 * l_val - 0.296 * s_val - 15.8
    return round(max(0, cli), 2)


def automated_readability(text: str) -> float:
    """Automated Readability Index (ARI).

    4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    """
    num_words, num_sentences, _, num_chars = _text_stats(text)
    ari = 4.71 * (num_chars / num_words) + 0.5 * (num_words / num_sentences) - 21.43
    return round(max(0, ari), 2)


def smog_grade(text: str) -> float:
    """SMOG Grade — best for healthcare/medical texts.

    1.0430 * sqrt(polysyllable_count * (30/sentences)) + 3.1291

    Requires at least 30 sentences for accuracy; works with fewer
    but results are approximate.
    """
    num_words, num_sentences, _, _ = _text_stats(text)
    complex_count = _count_complex_words(text)
    smog = 1.0430 * math.sqrt(complex_count * (30 / num_sentences)) + 3.1291
    return round(max(0, smog), 2)


def reading_level(text: str) -> dict[str, str | float]:
    """Comprehensive reading level summary.

    Returns grade level and human-readable label.
    """
    fk_grade = flesch_kincaid_grade(text)
    fre = flesch_reading_ease(text)

    if fk_grade <= 5:
        label = "elementary"
    elif fk_grade <= 8:
        label = "middle school"
    elif fk_grade <= 12:
        label = "high school"
    elif fk_grade <= 16:
        label = "college"
    else:
        label = "graduate"

    return {
        "grade_level": fk_grade,
        "label": label,
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog": gunning_fog(text),
        "coleman_liau": coleman_liau(text),
        "automated_readability": automated_readability(text),
        "smog_grade": smog_grade(text),
    }
