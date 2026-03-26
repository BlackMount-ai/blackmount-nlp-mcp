"""Blackmount NLP MCP Server — 40+ text analysis tools, zero heavy dependencies."""

from mcp.server.fastmcp import FastMCP

from . import clean, detect, keywords, readability, sentiment, similarity, summarize, tokenize

mcp = FastMCP(
    "Blackmount NLP",
    description=(
        "Text analysis and NLP without the bloat. "
        "No NLTK, no spaCy, no transformers — pure Python + regex. "
        "Fast, lightweight, surprisingly powerful."
    ),
)

# ============================================================
# Tokenization Tools
# ============================================================

@mcp.tool()
def word_tokenize(text: str) -> list[str]:
    """Split text into word tokens. Handles contractions, hyphenated words, numbers, and punctuation."""
    return tokenize.word_tokenize(text)


@mcp.tool()
def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences. Handles abbreviations (Mr., Dr., etc.) and tricky boundaries."""
    return tokenize.sentence_tokenize(text)


@mcp.tool()
def generate_ngrams(tokens: list[str], n: int = 2) -> list[list[str]]:
    """Generate n-grams from a list of tokens. Returns list of n-gram lists."""
    return [list(g) for g in tokenize.ngrams(tokens, n)]


@mcp.tool()
def generate_char_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character-level n-grams from text."""
    return tokenize.char_ngrams(text, n)


# ============================================================
# Readability Tools
# ============================================================

@mcp.tool()
def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score. 90-100=very easy, 60-69=standard, 0-29=very confusing."""
    return readability.flesch_reading_ease(text)


@mcp.tool()
def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level. Returns US school grade level needed to understand text."""
    return readability.flesch_kincaid_grade(text)


@mcp.tool()
def gunning_fog_index(text: str) -> float:
    """Gunning Fog Index. Estimates years of formal education needed to understand text."""
    return readability.gunning_fog(text)


@mcp.tool()
def coleman_liau_index(text: str) -> float:
    """Coleman-Liau Index. Grade level based on characters per word and sentences per word."""
    return readability.coleman_liau(text)


@mcp.tool()
def automated_readability_index(text: str) -> float:
    """Automated Readability Index (ARI). Grade level from character and word counts."""
    return readability.automated_readability(text)


@mcp.tool()
def smog_grade_index(text: str) -> float:
    """SMOG Grade. Best for healthcare/medical texts. Counts polysyllabic words."""
    return readability.smog_grade(text)


@mcp.tool()
def count_syllables(word: str) -> int:
    """Estimate syllable count for a single word using heuristics."""
    return readability.syllable_count(word)


@mcp.tool()
def get_reading_level(text: str) -> dict:
    """Comprehensive reading level: grade level, label (elementary/middle/high school/college/graduate), and all readability scores."""
    return readability.reading_level(text)


# ============================================================
# Sentiment Tools
# ============================================================

@mcp.tool()
def get_sentiment_score(text: str) -> float:
    """Compound sentiment score from -1 (negative) to 1 (positive). VADER-style with built-in 2000+ word lexicon."""
    return sentiment.sentiment_score(text)


@mcp.tool()
def get_sentiment_label(text: str) -> str:
    """Classify text as 'positive', 'negative', or 'neutral'."""
    return sentiment.sentiment_label(text)


@mcp.tool()
def get_sentence_sentiments(text: str) -> list[dict]:
    """Per-sentence sentiment breakdown. Returns list of {sentence, score, label}."""
    return sentiment.sentence_sentiments(text)


@mcp.tool()
def get_aspect_sentiment(text: str, aspects: list[str]) -> dict:
    """Sentiment around specific topics/aspects. Finds sentences mentioning each aspect and averages their sentiment."""
    return sentiment.aspect_sentiment(text, aspects)


# ============================================================
# Keyword Extraction Tools
# ============================================================

@mcp.tool()
def extract_tfidf_keywords(documents: list[str], top_n: int = 10) -> list[dict]:
    """Extract keywords using TF-IDF computed from scratch. Pass multiple docs for best results."""
    return keywords.tfidf_keywords(documents, top_n)


@mcp.tool()
def extract_rake_keywords(text: str, top_n: int = 10) -> list[dict]:
    """RAKE keyword extraction (Rapid Automatic Keyword Extraction). Finds multi-word key phrases."""
    return keywords.rake_keywords(text, top_n)


@mcp.tool()
def get_word_frequency(text: str, top_n: int = 20) -> list[dict]:
    """Most frequent words excluding stopwords. Returns [{word, count}]."""
    return keywords.word_frequency(text, top_n)


@mcp.tool()
def get_phrase_frequency(text: str, n: int = 2, top_n: int = 10) -> list[dict]:
    """Most frequent n-grams (phrases). Default bigrams. Returns [{phrase, count}]."""
    return keywords.phrase_frequency(text, n, top_n)


# ============================================================
# Similarity Tools
# ============================================================

@mcp.tool()
def get_jaccard_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity (word-level set overlap). 0=no overlap, 1=identical word sets."""
    return similarity.jaccard_similarity(text1, text2)


@mcp.tool()
def get_cosine_similarity(text1: str, text2: str) -> float:
    """Cosine similarity using bag-of-words vectors. 0=orthogonal, 1=identical."""
    return similarity.cosine_similarity_bow(text1, text2)


@mcp.tool()
def get_edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance. Minimum single-character edits to transform s1 into s2."""
    return similarity.edit_distance(s1, s2)


@mcp.tool()
def get_normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized edit distance on 0-1 scale. 0=identical, 1=completely different."""
    return similarity.normalized_edit_distance(s1, s2)


@mcp.tool()
def get_longest_common_subsequence(s1: str, s2: str) -> int:
    """Length of longest common subsequence (LCS) between two strings."""
    return similarity.longest_common_subsequence(s1, s2)


# ============================================================
# Text Cleaning Tools
# ============================================================

@mcp.tool()
def clean_remove_stopwords(text: str) -> str:
    """Remove English stopwords (500+ built-in) from text."""
    return clean.remove_stopwords(text)


@mcp.tool()
def clean_remove_punctuation(text: str) -> str:
    """Remove all punctuation from text."""
    return clean.remove_punctuation(text)


@mcp.tool()
def clean_remove_numbers(text: str) -> str:
    """Remove all numbers from text."""
    return clean.remove_numbers(text)


@mcp.tool()
def clean_remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return clean.remove_urls(text)


@mcp.tool()
def clean_remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return clean.remove_emails(text)


@mcp.tool()
def clean_remove_html(text: str) -> str:
    """Remove HTML tags from text."""
    return clean.remove_html(text)


@mcp.tool()
def clean_normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    return clean.normalize_whitespace(text)


@mcp.tool()
def clean_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return clean.lowercase(text)


@mcp.tool()
def porter_stem(word: str) -> str:
    """Porter stemmer from scratch. Reduce word to its stem (e.g., 'running' -> 'run')."""
    return clean.stem(word)


@mcp.tool()
def clean_text_pipeline(text: str, steps: list[str] | None = None) -> str:
    """Configurable cleaning pipeline. Steps: html, urls, emails, numbers, punctuation, stopwords, whitespace, lowercase."""
    return clean.clean_text(text, steps)


# ============================================================
# Detection Tools
# ============================================================

@mcp.tool()
def detect_text_language(text: str) -> list[dict]:
    """Detect language from text. Returns top 5 matches with confidence scores. Supports 20 languages."""
    return detect.detect_language(text)


@mcp.tool()
def detect_text_encoding_type(text: str) -> str:
    """Detect character encoding type: ASCII, Latin, Cyrillic, CJK, Arabic, etc."""
    return detect.detect_encoding_type(text)


@mcp.tool()
def check_is_english(text: str) -> float:
    """Confidence that text is English (0-1 scale)."""
    return detect.is_english(text)


@mcp.tool()
def count_words(text: str) -> int:
    """Count words in text."""
    return detect.word_count(text)


@mcp.tool()
def count_sentences(text: str) -> int:
    """Count sentences in text."""
    return detect.sentence_count(text)


@mcp.tool()
def count_paragraphs(text: str) -> int:
    """Count paragraphs in text (separated by blank lines)."""
    return detect.paragraph_count(text)


@mcp.tool()
def get_avg_word_length(text: str) -> float:
    """Average word length in characters."""
    return detect.avg_word_length(text)


@mcp.tool()
def get_avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    return detect.avg_sentence_length(text)


# ============================================================
# Summarization Tools
# ============================================================

@mcp.tool()
def get_extractive_summary(text: str, n_sentences: int = 3, title: str | None = None) -> str:
    """Extract the best N sentences as a summary. Scores by position, keyword frequency, length, and title overlap."""
    return summarize.extractive_summary(text, n_sentences, title)


@mcp.tool()
def get_text_statistics(text: str) -> dict:
    """Comprehensive text stats: words, sentences, paragraphs, reading time, readability scores, language."""
    return summarize.text_statistics(text)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
