# blackmount-nlp-mcp

[![PyPI version](https://img.shields.io/pypi/v/blackmount-nlp-mcp)](https://pypi.org/project/blackmount-nlp-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**NLP for MCP — zero heavy dependencies.**

45 text analysis tools delivered as a FastMCP server. No NLTK. No spaCy. No transformers. One dependency (`mcp[cli]`), under 50 KB of NLP code, ready in seconds.

---

## Why this exists

| | blackmount-nlp-mcp | NLTK | spaCy | transformers |
|---|---|---|---|---|
| **Package size** | 42 KB | 60 MB+ | 200 MB+ | 2 GB+ |
| **Dependencies** | 1 | many | many | many |
| **Tokenization** | ✅ | ✅ | ✅ | ✅ |
| **Sentiment analysis** | ✅ | ✅ | ❌ | ✅ |
| **Readability scores** | ✅ | ❌ | ❌ | ❌ |
| **Keyword extraction** | ✅ | ✅ | ❌ | ❌ |
| **Text similarity** | ✅ | ✅ | ✅ | ✅ |
| **Language detection** | ✅ (20 langs) | ❌ | ❌ | ❌ |

Everything is implemented from scratch in pure Python — Porter stemmer, TF-IDF, RAKE, Levenshtein, VADER-style sentiment, Flesch / Gunning Fog / Coleman-Liau / ARI / SMOG readability, extractive summarization, language detection — plus a built-in 2000+ word sentiment lexicon and 500+ stopword list, all baked into the package.

---

## Quick start

```bash
pip install blackmount-nlp-mcp
```

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

Restart Claude Desktop. All 45 NLP tools are now available.

---

## Tool catalog

### Tokenization (4 tools)

| Tool | Description |
|------|-------------|
| `word_tokenize` | Split text into words, handling contractions and punctuation |
| `sentence_tokenize` | Split into sentences, handling common abbreviations |
| `generate_ngrams` | Generate word-level n-grams from a token list |
| `generate_char_ngrams` | Generate character-level n-grams |

### Readability (8 tools)

| Tool | Description |
|------|-------------|
| `flesch_reading_ease` | 0–100 ease score (higher = easier) |
| `flesch_kincaid_grade` | US grade level estimate |
| `gunning_fog_index` | Fog index based on complex word ratio |
| `coleman_liau_index` | Coleman-Liau grade-level index |
| `automated_readability_index` | ARI grade-level index |
| `smog_grade_index` | SMOG grade (recommended for healthcare text) |
| `count_syllables` | Syllable count estimation for any word |
| `get_reading_level` | All readability scores in one call with a plain-English label |

### Sentiment Analysis (4 tools)

| Tool | Description |
|------|-------------|
| `get_sentiment_score` | Compound sentiment score from −1.0 (negative) to +1.0 (positive) |
| `get_sentiment_label` | Returns `positive`, `negative`, or `neutral` |
| `get_sentence_sentiments` | Per-sentence sentiment breakdown |
| `get_aspect_sentiment` | Sentiment score scoped to specific topics or keywords |

### Keyword Extraction (4 tools)

| Tool | Description |
|------|-------------|
| `extract_tfidf_keywords` | TF-IDF keyword ranking across a corpus |
| `extract_rake_keywords` | RAKE algorithm — phrase-level keyword extraction |
| `get_word_frequency` | Top words by frequency, stopwords excluded |
| `get_phrase_frequency` | Top n-gram phrases by frequency |

### Text Similarity (5 tools)

| Tool | Description |
|------|-------------|
| `get_jaccard_similarity` | Word-set overlap, 0–1 |
| `get_cosine_similarity` | Bag-of-words cosine similarity, 0–1 |
| `get_edit_distance` | Levenshtein edit distance |
| `get_normalized_edit_distance` | Edit distance normalized to 0–1 |
| `get_longest_common_subsequence` | LCS length between two strings |

### Text Cleaning (10 tools)

| Tool | Description |
|------|-------------|
| `clean_remove_stopwords` | Strip 500+ English stopwords |
| `clean_remove_punctuation` | Remove all punctuation |
| `clean_remove_numbers` | Remove numeric tokens |
| `clean_remove_urls` | Strip URLs |
| `clean_remove_emails` | Strip email addresses |
| `clean_remove_html` | Strip HTML tags |
| `clean_normalize_whitespace` | Collapse and trim whitespace |
| `clean_lowercase` | Lowercase the text |
| `porter_stem` | Porter stemmer (pure Python, no NLTK) |
| `clean_text_pipeline` | Configurable multi-step cleaning in one call |

### Detection (8 tools)

| Tool | Description |
|------|-------------|
| `detect_text_language` | Identify language from 20 supported languages |
| `detect_text_encoding_type` | Detect script type: ASCII, Latin, Cyrillic, CJK, Arabic, etc. |
| `check_is_english` | English confidence score, 0–1 |
| `count_words` | Word count |
| `count_sentences` | Sentence count |
| `count_paragraphs` | Paragraph count |
| `get_avg_word_length` | Mean word length in characters |
| `get_avg_sentence_length` | Mean sentence length in words |

### Summarization (2 tools)

| Tool | Description |
|------|-------------|
| `get_extractive_summary` | Select the N highest-scoring sentences from a document |
| `get_text_statistics` | Full document stats: word count, readability, language, reading time |

---

## Use as a library

The submodules are importable directly — no MCP server required:

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

---

## Development

```bash
git clone https://github.com/BlackMount-ai/blackmount-nlp-mcp
cd blackmount-nlp-mcp
pip install -e .
pytest tests/ -v
```

---

## Blackmount ecosystem

blackmount-nlp-mcp is part of the [Blackmount](https://blackmount.ai) ecosystem.

Also check out **[blackmount-mcp](https://github.com/BlackMount-ai/blackmount-mcp)** — browser memory, AI chat search, and session analytics as an MCP server. Different audience, same zero-bloat philosophy.

---

## License

MIT
