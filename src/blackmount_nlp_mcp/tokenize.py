"""Tokenization utilities — word, sentence, and n-gram generation."""

import re


# Common abbreviations that shouldn't trigger sentence breaks
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "fig", "gen", "gov", "hon", "inc", "ltd", "no",
    "oct", "nov", "dec", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "vs", "etc", "approx", "appt", "apt", "dept",
    "dpt", "est", "min", "max", "misc", "tech", "temp", "vet",
    "vol", "rev", "sgt", "cpl", "pvt", "capt", "cmdr", "lt",
    "col", "maj", "brig", "adm", "gen", "cdr", "pfc", "spc",
    "cpt", "ph", "ed", "al", "op", "cit", "ibid",
    "i.e", "e.g", "viz", "cf", "al",
}

# Contraction patterns
_CONTRACTION_RE = re.compile(
    r"(?i)\b("
    r"can't|won't|shan't|shouldn't|wouldn't|couldn't|mustn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"doesn't|don't|didn't|mightn't|needn't|oughtn't|"
    r"i'm|you're|he's|she's|it's|we're|they're|"
    r"i've|you've|we've|they've|"
    r"i'd|you'd|he'd|she'd|we'd|they'd|"
    r"i'll|you'll|he'll|she'll|we'll|they'll|"
    r"that's|who's|what's|where's|when's|why's|how's|"
    r"let's|there's|here's|"
    r"ain't|ma'am|o'clock|'twas|'tis"
    r")\b"
)

# Word tokenizer pattern: words, contractions, numbers, punctuation
_WORD_RE = re.compile(
    r"(?:[A-Za-z]+'[A-Za-z]+)"  # contractions (don't, I'm)
    r"|(?:\d+(?:\.\d+)?(?:%|st|nd|rd|th)?)"  # numbers with optional suffix
    r"|(?:[A-Za-z]+(?:-[A-Za-z]+)*)"  # words with optional hyphens
    r"|(?:[.!?;:,\"\'\(\)\[\]{}\-/])"  # punctuation as separate tokens
)


def word_tokenize(text: str) -> list[str]:
    """Split text into word tokens.

    Handles contractions (don't -> don't as single token),
    hyphenated words, numbers, and punctuation.
    """
    if not text or not text.strip():
        return []
    return _WORD_RE.findall(text)


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences.

    Handles abbreviations (Mr., Dr., etc.), ellipsis,
    and other tricky sentence boundaries.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    sentences: list[str] = []
    current = []
    i = 0

    while i < len(text):
        current.append(text[i])

        if text[i] in ".!?":
            # Check for ellipsis
            if text[i] == "." and i + 2 < len(text) and text[i + 1 : i + 3] == "..":
                current.append(".")
                current.append(".")
                i += 3
                continue

            # Check for abbreviation
            is_abbrev = False
            if text[i] == ".":
                # Get the word before the period
                current_text = "".join(current).strip()
                words = current_text.split()
                if words:
                    last_word = words[-1].rstrip(".")
                    if last_word.lower() in _ABBREVIATIONS:
                        is_abbrev = True
                    # Single letter abbreviations (initials)
                    elif len(last_word) == 1 and last_word.isalpha():
                        is_abbrev = True

            if not is_abbrev:
                # Look ahead for closing quote or space + uppercase
                j = i + 1
                while j < len(text) and text[j] in "\"')]}":
                    current.append(text[j])
                    j += 1

                # Check if next char (after spaces) starts a new sentence
                k = j
                while k < len(text) and text[k] == " ":
                    k += 1

                if k >= len(text) or text[k].isupper() or text[k] in "\"'([{":
                    sentence = "".join(current).strip()
                    if sentence:
                        sentences.append(sentence)
                    current = []
                    i = j
                    # Skip spaces
                    while i < len(text) and text[i] == " ":
                        i += 1
                    continue

        i += 1

    # Don't forget the last sentence
    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)

    return sentences


def ngrams(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    """Generate n-grams from a list of tokens.

    Args:
        tokens: List of string tokens.
        n: Size of n-grams (default 2 for bigrams).

    Returns:
        List of n-gram tuples.
    """
    if not tokens or n < 1 or n > len(tokens):
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character-level n-grams from text.

    Args:
        text: Input string.
        n: Size of character n-grams (default 3).

    Returns:
        List of character n-gram strings.
    """
    if not text or n < 1 or n > len(text):
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]
