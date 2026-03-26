"""Keyword extraction — TF-IDF, RAKE, word frequency, phrase frequency."""

import math
import re
from collections import Counter

from ._stopwords import ENGLISH_STOPWORDS
from .tokenize import ngrams, word_tokenize


def _tokenize_lower(text: str) -> list[str]:
    """Tokenize and lowercase, keeping only alphabetic words."""
    return [w.lower() for w in word_tokenize(text) if re.match(r"[A-Za-z]", w)]


def tfidf_keywords(
    documents: list[str], top_n: int = 10
) -> list[dict[str, str | float]]:
    """Extract keywords using TF-IDF computed from scratch.

    Args:
        documents: List of document strings. TF-IDF is computed across
                   all documents. If a single document, it's still useful
                   as IDF penalizes common English words.
        top_n: Number of top keywords to return.

    Returns:
        List of dicts with 'term' and 'score' keys, sorted by score descending.
        Scores are averaged across documents where the term appears.
    """
    if not documents:
        return []

    # Tokenize all docs
    doc_tokens = [_tokenize_lower(doc) for doc in documents]
    num_docs = len(documents)

    # Document frequency: how many docs contain each term
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1

    # Compute TF-IDF for each term across all docs
    tfidf_scores: dict[str, float] = {}

    for doc_idx, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        tf: Counter[str] = Counter(tokens)
        max_tf = max(tf.values())

        for term, count in tf.items():
            if term in ENGLISH_STOPWORDS or len(term) <= 2:
                continue
            # Augmented TF to prevent bias toward longer docs
            tf_val = 0.5 + 0.5 * (count / max_tf)
            idf_val = math.log((num_docs + 1) / (df[term] + 1)) + 1
            score = tf_val * idf_val

            if term not in tfidf_scores:
                tfidf_scores[term] = 0.0
            tfidf_scores[term] += score

    # Average across docs where term appears
    for term in tfidf_scores:
        tfidf_scores[term] /= df[term]

    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"term": t, "score": round(s, 4)} for t, s in sorted_terms[:top_n]]


def rake_keywords(text: str, top_n: int = 10) -> list[dict[str, str | float]]:
    """Extract keywords using RAKE (Rapid Automatic Keyword Extraction).

    RAKE algorithm:
    1. Split text into candidate phrases using stopwords and punctuation as delimiters
    2. Calculate word scores as degree(word) / frequency(word)
    3. Score each phrase as sum of its word scores
    4. Return top-scoring phrases

    Args:
        text: Input text.
        top_n: Number of top keywords to return.

    Returns:
        List of dicts with 'phrase' and 'score' keys.
    """
    if not text or not text.strip():
        return []

    # Build stopword + punctuation pattern for splitting
    text_lower = text.lower()

    # Split on stopwords and punctuation to get candidate phrases
    # First split on sentence boundaries and punctuation
    chunks = re.split(r"[.!?,;:\-\(\)\[\]{}\"\'\n\t]", text_lower)

    candidates: list[list[str]] = []
    for chunk in chunks:
        words = chunk.split()
        phrase: list[str] = []
        for word in words:
            word = re.sub(r"[^a-z]", "", word)
            if word and word not in ENGLISH_STOPWORDS and len(word) > 1:
                phrase.append(word)
            else:
                if phrase:
                    candidates.append(phrase)
                phrase = []
        if phrase:
            candidates.append(phrase)

    if not candidates:
        return []

    # Build co-occurrence: degree and frequency for each word
    word_freq: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()

    for phrase in candidates:
        degree = len(phrase) - 1
        for word in phrase:
            word_freq[word] += 1
            word_degree[word] += degree

    # Word score = (degree + frequency) / frequency
    word_score: dict[str, float] = {}
    for word in word_freq:
        word_score[word] = (word_degree[word] + word_freq[word]) / word_freq[word]

    # Phrase score = sum of word scores
    phrase_scores: dict[str, float] = {}
    for phrase in candidates:
        phrase_str = " ".join(phrase)
        if phrase_str not in phrase_scores:
            phrase_scores[phrase_str] = sum(word_score.get(w, 0) for w in phrase)

    sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"phrase": p, "score": round(s, 4)} for p, s in sorted_phrases[:top_n]]


def word_frequency(text: str, top_n: int = 20) -> list[dict[str, str | int]]:
    """Count most frequent words, excluding stopwords.

    Args:
        text: Input text.
        top_n: Number of top words to return.

    Returns:
        List of dicts with 'word' and 'count' keys.
    """
    tokens = _tokenize_lower(text)
    filtered = [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]
    counts = Counter(filtered)
    return [{"word": w, "count": c} for w, c in counts.most_common(top_n)]


def phrase_frequency(
    text: str, n: int = 2, top_n: int = 10
) -> list[dict[str, str | int]]:
    """Count most frequent n-grams (phrases), excluding stopword-only phrases.

    Args:
        text: Input text.
        n: N-gram size (default 2 for bigrams).
        top_n: Number of top phrases to return.

    Returns:
        List of dicts with 'phrase' and 'count' keys.
    """
    tokens = _tokenize_lower(text)
    grams = ngrams(tokens, n)

    # Filter out n-grams where ALL words are stopwords
    filtered = [
        g for g in grams
        if not all(w in ENGLISH_STOPWORDS for w in g)
    ]

    counts = Counter(filtered)
    results = []
    for gram, count in counts.most_common(top_n):
        results.append({"phrase": " ".join(gram), "count": count})
    return results
