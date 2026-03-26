# blackmount-nlp-mcp

NLP without the bloat — no NLTK, no spaCy, no transformers, just results.

A Model Context Protocol (MCP) server providing 40+ text analysis tools built with pure Python and regex. Fast, lightweight, surprisingly powerful.

## Zero Heavy Dependencies

This package has **one dependency**: `mcp[cli]` for the MCP server protocol. Everything else is pure Python:

- No NLTK (60MB+ download)
- No spaCy (200MB+ models)
- No transformers (2GB+ models)
- No numpy, no scikit-learn
- Built-in 2000+ word sentiment lexicon
- Built-in 500+ stopword list
- Porter stemmer from scratch
- TF-IDF from scratch

## Install

```bash
pip install blackmount-nlp-mcp
```

## MCP Configuration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

## 40+ Tools

### Tokenization
- `word_tokenize` — split text into words (handles contractions, punctuation)
- `sentence_tokenize` — split into sentences (handles abbreviations)
- `generate_ngrams` — n-grams from token lists
- `generate_char_ngrams` — character-level n-grams

### Readability Scores
- `flesch_reading_ease` — 0-100 ease score
- `flesch_kincaid_grade` — US grade level
- `gunning_fog_index` — Fog index
- `coleman_liau_index` — Coleman-Liau index
- `automated_readability_index` — ARI
- `smog_grade_index` — SMOG (best for healthcare texts)
- `count_syllables` — syllable estimation
- `get_reading_level` — comprehensive summary with all scores

### Sentiment Analysis (VADER-style)
- `get_sentiment_score` — compound score -1 to +1
- `get_sentiment_label` — positive / negative / neutral
- `get_sentence_sentiments` — per-sentence breakdown
- `get_aspect_sentiment` — sentiment around specific topics

### Keyword Extraction
- `extract_tfidf_keywords` — TF-IDF from scratch
- `extract_rake_keywords` — RAKE algorithm
- `get_word_frequency` — top words (excluding stopwords)
- `get_phrase_frequency` — top n-gram phrases

### Text Similarity
- `get_jaccard_similarity` — word set overlap
- `get_cosine_similarity` — bag-of-words cosine
- `get_edit_distance` — Levenshtein distance
- `get_normalized_edit_distance` — 0-1 scale
- `get_longest_common_subsequence` — LCS length

### Text Cleaning
- `clean_remove_stopwords` — remove 500+ English stopwords
- `clean_remove_punctuation` / `clean_remove_numbers`
- `clean_remove_urls` / `clean_remove_emails` / `clean_remove_html`
- `clean_normalize_whitespace` / `clean_lowercase`
- `porter_stem` — Porter stemmer from scratch
- `clean_text_pipeline` — configurable multi-step cleaning

### Language & Content Detection
- `detect_text_language` — 20 languages by word frequency heuristic
- `detect_text_encoding_type` — ASCII, Latin, Cyrillic, CJK, Arabic, etc.
- `check_is_english` — confidence score 0-1
- `count_words` / `count_sentences` / `count_paragraphs`
- `get_avg_word_length` / `get_avg_sentence_length`

### Summarization & Statistics
- `get_extractive_summary` — pick best N sentences by scoring
- `get_text_statistics` — comprehensive stats (words, readability, language, reading time)

## Use as a Library

```python
from blackmount_nlp_mcp.sentiment import sentiment_score, sentiment_label
from blackmount_nlp_mcp.readability import reading_level
from blackmount_nlp_mcp.keywords import rake_keywords

text = "This product is absolutely amazing! The quality is excellent."

print(sentiment_score(text))   # 0.8745
print(sentiment_label(text))   # "positive"
print(reading_level(text))     # {"grade_level": 5.2, "label": "elementary", ...}
print(rake_keywords(text))     # [{"phrase": "product", "score": 1.0}, ...]
```

## Development

```bash
git clone https://github.com/blackmount/blackmount-nlp-mcp
cd blackmount-nlp-mcp
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
