"""Extractive summarization and comprehensive text statistics."""

import math
import re
from collections import Counter

from ._stopwords import ENGLISH_STOPWORDS
from .detect import (
    avg_sentence_length,
    avg_word_length,
    detect_language,
    paragraph_count,
    sentence_count,
    word_count,
)
from .readability import reading_level, syllable_count
from .tokenize import sentence_tokenize, word_tokenize


def extractive_summary(
    text: str, n_sentences: int = 3, title: str | None = None
) -> str:
    """Pick the best N sentences from text using scoring heuristics.

    Scoring factors:
    - Position: first and last sentences get a bonus
    - Keyword frequency: sentences with common (non-stopword) words score higher
    - Length: sentences that are too short or too long are penalized
    - Title overlap: sentences sharing words with the title score higher

    Args:
        text: Input text to summarize.
        n_sentences: Number of sentences to extract (default 3).
        title: Optional title for title-word overlap bonus.

    Returns:
        Extracted summary as a single string.
    """
    sentences = sentence_tokenize(text)
    if not sentences:
        return ""
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    # Build word frequency map (non-stopwords)
    all_words = [
        w.lower()
        for w in word_tokenize(text)
        if re.match(r"[A-Za-z]", w) and w.lower() not in ENGLISH_STOPWORDS and len(w) > 2
    ]
    word_freq = Counter(all_words)

    # Title words for overlap scoring
    title_words: set[str] = set()
    if title:
        title_words = {
            w.lower()
            for w in word_tokenize(title)
            if re.match(r"[A-Za-z]", w) and w.lower() not in ENGLISH_STOPWORDS
        }

    # Score each sentence
    scores: list[tuple[int, float]] = []

    for idx, sent in enumerate(sentences):
        score = 0.0
        sent_words = [
            w.lower()
            for w in word_tokenize(sent)
            if re.match(r"[A-Za-z]", w)
        ]

        if not sent_words:
            scores.append((idx, 0.0))
            continue

        # 1. Keyword frequency score
        content_words = [w for w in sent_words if w not in ENGLISH_STOPWORDS and len(w) > 2]
        if content_words:
            freq_score = sum(word_freq.get(w, 0) for w in content_words) / len(content_words)
            score += freq_score

        # 2. Position score
        total = len(sentences)
        if idx == 0:
            score += 2.0  # First sentence bonus
        elif idx == total - 1:
            score += 1.0  # Last sentence bonus
        elif idx <= total * 0.2:
            score += 1.0  # Early paragraph bonus
        elif idx >= total * 0.8:
            score += 0.5  # Late paragraph slight bonus

        # 3. Length penalty
        word_cnt = len(sent_words)
        if word_cnt < 5:
            score *= 0.3  # Too short
        elif word_cnt > 50:
            score *= 0.7  # Too long
        elif 10 <= word_cnt <= 30:
            score *= 1.2  # Ideal length bonus

        # 4. Title overlap
        if title_words:
            overlap = len(set(sent_words) & title_words)
            score += overlap * 1.5

        scores.append((idx, score))

    # Sort by score, pick top N, then re-sort by original position
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in scores[:n_sentences]])

    return " ".join(sentences[i] for i in top_indices)


def text_statistics(text: str) -> dict:
    """Comprehensive text statistics.

    Returns dict with: words, sentences, paragraphs, characters,
    avg_word_length, avg_sentence_length, reading_time_minutes,
    readability scores, detected language.
    """
    if not text or not text.strip():
        return {
            "words": 0,
            "sentences": 0,
            "paragraphs": 0,
            "characters": 0,
            "characters_no_spaces": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "reading_time_minutes": 0.0,
            "readability": {},
            "language": "unknown",
        }

    wc = word_count(text)
    sc = sentence_count(text)
    pc = paragraph_count(text)
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))

    # Reading time: ~238 words per minute average
    reading_time = round(wc / 238, 2) if wc > 0 else 0.0

    readability_info = reading_level(text) if wc >= 10 else {}

    lang_results = detect_language(text)
    top_lang = lang_results[0]["language"] if lang_results else "unknown"

    return {
        "words": wc,
        "sentences": sc,
        "paragraphs": pc,
        "characters": chars,
        "characters_no_spaces": chars_no_spaces,
        "avg_word_length": avg_word_length(text),
        "avg_sentence_length": avg_sentence_length(text),
        "reading_time_minutes": reading_time,
        "readability": readability_info,
        "language": top_lang,
    }
