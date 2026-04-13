# blackmount-nlp-mcp

[![PyPI version](https://img.shields.io/pypi/v/blackmount-nlp-mcp)](https://pypi.org/project/blackmount-nlp-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**NLP for MCP — zero heavy dependencies.** Built by [Blackmount](https://blackmount.ai).

45 text analysis tools as a FastMCP server. No NLTK. No spaCy. No transformers. One dependency (`mcp[cli]`), under 50 KB of NLP code, ready in seconds. Requires Python 3.10+.

---

## Why this exists

| | blackmount-nlp-mcp | NLTK | spaCy | transformers |
|---|---|---|---|---|
| **Wheel size** | 42 KB | 1.5 MB+ | 8 MB+ (+ models) | 400 KB+ (+ models) |
| **Direct dependencies** | 1 | many | many | many |
| **Tokenization** | ✅ | ✅ | ✅ | ✅ |
| **Sentiment analysis** | ✅ | ✅ | ❌ | ✅ |
| **Readability scores** | ✅ | ❌ | ❌ | ❌ |
| **Keyword extraction** | ✅ | ✅ | ❌ | ❌ |
| **Text similarity** | ✅ | ✅ | ✅ | ✅ |
| **Language detection** | ✅ (18 langs) | ❌ | ❌ | ❌ |

Everything is implemented from scratch in pure Python — Porter stemmer, TF-IDF, RAKE, Levenshtein, VADER-style sentiment, Flesch / Gunning Fog / Coleman-Liau / ARI / SMOG readability, extractive summarization, language detection — plus a built-in 2000+ word sentiment lexicon and 500+ stopword list, all baked into the package.

---

## Quick start

```bash
pip install blackmount-nlp-mcp
```

### Claude Desktop

Add to your config file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "nlp": {
      "command": "blackmount-nlp-mcp"
    }
  }
}
```

### Any MCP client

The server runs over stdio. Point your client at the `blackmount-nlp-mcp` command:

```bash
blackmount-nlp-mcp
```

Restart your editor. All 45 NLP tools are now available — just ask in natural language.

---

## Tool catalog

### Tokenization (4 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `word_tokenize` | Split text into words, handling contractions and punctuation | "Tokenize this paragraph into words" |
| `sentence_tokenize` | Split into sentences, handling common abbreviations | "Break this text into individual sentences" |
| `generate_ngrams` | Generate word-level n-grams from a token list | "Generate bigrams from these tokens" |
| `generate_char_ngrams` | Generate character-level n-grams | "Get character trigrams for this word" |

### Readability (8 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `flesch_reading_ease` | 0–100 ease score (higher = easier) | "Calculate the Flesch Reading Ease score" |
| `flesch_kincaid_grade` | US grade level estimate | "What grade level is this written at?" |
| `gunning_fog_index` | Fog index based on complex word ratio | "Calculate the Fog index for this text" |
| `coleman_liau_index` | Coleman-Liau grade-level index | "Get the Coleman-Liau score" |
| `automated_readability_index` | ARI grade-level index | "What's the ARI for this document?" |
| `smog_grade_index` | SMOG grade (recommended for healthcare text) | "Calculate the SMOG grade for this document" |
| `count_syllables` | Syllable count estimation for any word | "How many syllables in 'extraordinary'?" |
| `get_reading_level` | All readability scores in one call with a plain-English label | "Give me a full readability report for this text" |

### Sentiment Analysis (4 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `get_sentiment_score` | Compound sentiment score from −1.0 to +1.0 | "What's the sentiment of this customer review?" |
| `get_sentiment_label` | Returns `positive`, `negative`, or `neutral` | "Is this feedback positive or negative?" |
| `get_sentence_sentiments` | Per-sentence sentiment breakdown | "Show me the sentiment of each sentence" |
| `get_aspect_sentiment` | Sentiment scoped to specific topics | "What's the sentiment around 'pricing' in these reviews?" |

### Keyword Extraction (4 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `extract_tfidf_keywords` | TF-IDF keyword ranking across a corpus | "What are the key terms across these docs?" |
| `extract_rake_keywords` | RAKE algorithm — phrase-level keyword extraction | "Extract the key phrases from this article" |
| `get_word_frequency` | Top words by frequency, stopwords excluded | "What are the most common words in this text?" |
| `get_phrase_frequency` | Top n-gram phrases by frequency | "What two-word phrases appear most often?" |

### Text Similarity (5 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `get_jaccard_similarity` | Word-set overlap, 0–1 | "How similar are these two paragraphs?" |
| `get_cosine_similarity` | Bag-of-words cosine similarity, 0–1 | "Calculate cosine similarity between these texts" |
| `get_edit_distance` | Levenshtein edit distance | "How many edits to turn 'kitten' into 'sitting'?" |
| `get_normalized_edit_distance` | Edit distance normalized to 0–1 | "How different are these two strings?" |
| `get_longest_common_subsequence` | LCS length between two strings | "What's the LCS length of these two strings?" |

### Text Cleaning (10 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `clean_remove_stopwords` | Strip 500+ English stopwords | "Remove stopwords from this text" |
| `clean_remove_punctuation` | Remove all punctuation | "Strip the punctuation" |
| `clean_remove_numbers` | Remove numeric tokens | "Remove all numbers from this" |
| `clean_remove_urls` | Strip URLs | "Clean out the URLs" |
| `clean_remove_emails` | Strip email addresses | "Remove email addresses from this text" |
| `clean_remove_html` | Strip HTML tags | "Strip the HTML from this content" |
| `clean_normalize_whitespace` | Collapse and trim whitespace | "Normalize the whitespace" |
| `clean_lowercase` | Lowercase the text | "Convert this to lowercase" |
| `porter_stem` | Porter stemmer (pure Python, no NLTK) | "Stem the word 'running'" |
| `clean_text_pipeline` | Configurable multi-step cleaning in one call | "Clean this text: remove HTML, URLs, and stopwords" |

### Detection (8 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `detect_text_language` | Identify language from 18 supported languages | "What language is this text written in?" |
| `detect_text_encoding_type` | Detect script: ASCII, Latin, Cyrillic, CJK, Arabic | "What script does this text use?" |
| `check_is_english` | English confidence score, 0–1 | "Is this text in English?" |
| `count_words` | Word count | "How many words are in this?" |
| `count_sentences` | Sentence count | "Count the sentences" |
| `count_paragraphs` | Paragraph count | "How many paragraphs?" |
| `get_avg_word_length` | Mean word length in characters | "What's the average word length?" |
| `get_avg_sentence_length` | Mean sentence length in words | "How long are the sentences on average?" |

### Summarization (2 tools)

| Tool | Description | Try asking |
|------|-------------|------------|
| `get_extractive_summary` | Select the N highest-scoring sentences from a document | "Summarize this article in 3 sentences" |
| `get_text_statistics` | Full document stats: words, readability, language, reading time | "Give me a statistical profile of this text" |

---

## Use as a library

The submodules are importable directly — no MCP server required:

```python
from blackmount_nlp_mcp.sentiment import sentiment_score, sentiment_label
from blackmount_nlp_mcp.readability import reading_level
from blackmount_nlp_mcp.keywords import rake_keywords

text = "This product is absolutely amazing! The quality is excellent."

print(sentiment_score(text))
# 0.9285

print(sentiment_label(text))
# 'positive'

print(reading_level(text))
# {'grade_level': 12.39, 'label': 'college',
#  'flesch_reading_ease': 14.27, 'flesch_kincaid_grade': 12.39,
#  'gunning_fog': 19.58, 'coleman_liau': 10.94,
#  'automated_readability': 7.51, 'smog_grade': 11.21}

print(rake_keywords(text))
# [{'phrase': 'absolutely amazing', 'score': 4.0},
#  {'phrase': 'product', 'score': 1.0},
#  {'phrase': 'quality', 'score': 1.0},
#  {'phrase': 'excellent', 'score': 1.0}]
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

blackmount-nlp-mcp is built by [Blackmount](https://blackmount.ai) — tools for people who work with AI.

**[blackmount-mcp](https://github.com/BlackMount-ai/blackmount-mcp)** — Browser memory, AI chat search, and session analytics as an MCP server. Pair it with blackmount-nlp-mcp to analyze your saved conversations: extract keywords from chat history, score readability of AI responses, detect sentiment trends across sessions.

**[app.blackmount.ai](https://app.blackmount.ai)** — The full Blackmount platform. Search, organize, and analyze everything your AI tools produce.

---

## License

MIT
