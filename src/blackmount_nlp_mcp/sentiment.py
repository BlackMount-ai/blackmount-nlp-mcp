"""Rule-based sentiment analysis (VADER-style, no NLTK dependency).

Uses a built-in lexicon of 2000+ scored words with intensity modifiers,
negation handling, and booster words.
"""

import math
import re

from ._lexicon import BOOSTERS, NEGATIONS, SENTIMENT_LEXICON
from .tokenize import sentence_tokenize, word_tokenize


def _normalize_score(score: float, alpha: float = 15.0) -> float:
    """Normalize raw score to [-1, 1] using VADER-style normalization."""
    return score / math.sqrt(score * score + alpha)


def sentiment_score(text: str) -> float:
    """Compute compound sentiment score from -1 (negative) to 1 (positive).

    Handles:
    - Word-level sentiment from lexicon
    - Negation (flips sentiment of next word)
    - Boosters/dampeners (intensify or reduce sentiment)
    - Capitalization (ALL CAPS = +10% intensity)
    - Exclamation marks (add slight positive emphasis)
    - Question marks at end (reduce confidence slightly)
    """
    if not text or not text.strip():
        return 0.0

    tokens = word_tokenize(text)
    if not tokens:
        return 0.0

    sentiments: list[float] = []
    i = 0
    words_lower = [t.lower() for t in tokens]

    while i < len(tokens):
        token = tokens[i]
        token_lower = words_lower[i]

        # Skip punctuation
        if not re.match(r"[A-Za-z]", token):
            i += 1
            continue

        # Check if word is in lexicon
        score = SENTIMENT_LEXICON.get(token_lower, 0.0)

        if score != 0.0:
            # Check for ALL CAPS (emphasis)
            if token.isupper() and len(token) > 1:
                if score > 0:
                    score += 0.733
                else:
                    score -= 0.733

            # Check preceding words for negation (up to 3 words back)
            negated = False
            for j in range(max(0, i - 3), i):
                if words_lower[j] in NEGATIONS:
                    negated = True
                    break
                # "not" before "only" is not negation
                if words_lower[j] == "not" and j + 1 < i and words_lower[j + 1] == "only":
                    negated = False

            if negated:
                score *= -0.74  # VADER's negation scalar

            # Check preceding word for booster/dampener
            if i > 0:
                prev_lower = words_lower[i - 1]
                booster = BOOSTERS.get(prev_lower, 0.0)
                if booster != 0.0:
                    if score > 0:
                        score += booster
                    else:
                        score -= booster

            sentiments.append(score)
        i += 1

    if not sentiments:
        return 0.0

    raw_sum = sum(sentiments)

    # Punctuation adjustments
    excl_count = text.count("!")
    if excl_count > 0:
        raw_sum += min(excl_count, 4) * 0.292

    if text.strip().endswith("?") and raw_sum > 0:
        raw_sum *= 0.94  # Slight reduction for questions

    return round(_normalize_score(raw_sum), 4)


def sentiment_label(text: str) -> str:
    """Classify text sentiment as 'positive', 'negative', or 'neutral'.

    Thresholds:
    - score >= 0.05: positive
    - score <= -0.05: negative
    - otherwise: neutral
    """
    score = sentiment_score(text)
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def sentence_sentiments(text: str) -> list[dict[str, str | float]]:
    """Analyze sentiment of each sentence individually.

    Returns list of dicts with 'sentence', 'score', and 'label' keys.
    """
    sentences = sentence_tokenize(text)
    results = []
    for sent in sentences:
        score = sentiment_score(sent)
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        results.append({
            "sentence": sent,
            "score": score,
            "label": label,
        })
    return results


def aspect_sentiment(text: str, aspects: list[str]) -> dict[str, dict[str, float | str]]:
    """Analyze sentiment around specific aspects/topics.

    For each aspect, finds sentences mentioning it and computes
    the average sentiment of those sentences.

    Args:
        text: Input text to analyze.
        aspects: List of aspect terms to look for.

    Returns:
        Dict mapping each aspect to its sentiment score, label,
        and number of mentions.
    """
    sentences = sentence_tokenize(text)
    results = {}

    for aspect in aspects:
        aspect_lower = aspect.lower()
        scores = []

        for sent in sentences:
            if aspect_lower in sent.lower():
                scores.append(sentiment_score(sent))

        if scores:
            avg_score = round(sum(scores) / len(scores), 4)
            if avg_score >= 0.05:
                label = "positive"
            elif avg_score <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            results[aspect] = {
                "score": avg_score,
                "label": label,
                "mentions": len(scores),
            }
        else:
            results[aspect] = {
                "score": 0.0,
                "label": "not found",
                "mentions": 0,
            }

    return results
