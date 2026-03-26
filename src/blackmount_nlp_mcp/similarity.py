"""Text similarity — Jaccard, cosine (bag of words), edit distance, LCS."""

import math
import re
from collections import Counter

from .tokenize import word_tokenize


def _word_set(text: str) -> set[str]:
    """Tokenize text into a lowercase word set."""
    return {w.lower() for w in word_tokenize(text) if re.match(r"[A-Za-z]", w)}


def _word_counter(text: str) -> Counter[str]:
    """Tokenize text into a lowercase word frequency counter."""
    return Counter(
        w.lower() for w in word_tokenize(text) if re.match(r"[A-Za-z]", w)
    )


def jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity — word-level set overlap.

    J(A,B) = |A intersect B| / |A union B|

    Returns value between 0 (no overlap) and 1 (identical word sets).
    """
    set1 = _word_set(text1)
    set2 = _word_set(text2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return round(len(intersection) / len(union), 4)


def cosine_similarity_bow(text1: str, text2: str) -> float:
    """Cosine similarity using bag-of-words vectors.

    cos(A,B) = (A . B) / (|A| * |B|)

    Returns value between 0 and 1.
    """
    counter1 = _word_counter(text1)
    counter2 = _word_counter(text2)

    if not counter1 or not counter2:
        return 0.0

    # All unique words
    all_words = set(counter1.keys()) | set(counter2.keys())

    dot_product = sum(counter1.get(w, 0) * counter2.get(w, 0) for w in all_words)
    magnitude1 = math.sqrt(sum(v * v for v in counter1.values()))
    magnitude2 = math.sqrt(sum(v * v for v in counter2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return round(dot_product / (magnitude1 * magnitude2), 4)


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings.

    Minimum number of single-character edits (insertions, deletions,
    substitutions) to transform s1 into s2.

    Uses O(min(m,n)) space via two-row dynamic programming.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost: 0 if chars match, 1 if substitution needed
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insertion
                prev_row[j + 1] + 1,   # deletion
                prev_row[j] + cost,     # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized edit distance on a 0-1 scale.

    0 = identical strings, 1 = completely different.
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return round(edit_distance(s1, s2) / max_len, 4)


def longest_common_subsequence(s1: str, s2: str) -> int:
    """Length of the longest common subsequence (LCS).

    Uses O(min(m,n)) space optimization.
    """
    if not s1 or not s2:
        return 0

    # Make s2 the shorter string for space optimization
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev_row = [0] * (len(s2) + 1)

    for c1 in s1:
        curr_row = [0] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            if c1 == c2:
                curr_row[j + 1] = prev_row[j] + 1
            else:
                curr_row[j + 1] = max(curr_row[j], prev_row[j + 1])
        prev_row = curr_row

    return prev_row[-1]
