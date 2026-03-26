# blackmount-nlp-mcp

Text analysis and NLP MCP server. Pure Python + regex. No NLTK, no spaCy, no transformers, no numpy.

## Repo Structure
- `src/blackmount_nlp_mcp/` — all source code
  - `server.py` — FastMCP server with 40+ tools
  - `tokenize.py` — word/sentence tokenization, n-grams
  - `readability.py` — Flesch, Gunning Fog, Coleman-Liau, ARI, SMOG
  - `sentiment.py` — VADER-style rule-based sentiment analysis
  - `keywords.py` — TF-IDF, RAKE, word/phrase frequency
  - `similarity.py` — Jaccard, cosine, edit distance, LCS
  - `clean.py` — stopwords, stemming, text cleaning pipeline
  - `detect.py` — language detection, encoding, text stats
  - `summarize.py` — extractive summarization, comprehensive stats
  - `_lexicon.py` — 2000+ word sentiment lexicon (built-in)
  - `_stopwords.py` — 500+ English stopwords (built-in)
- `tests/` — pytest tests with real text samples
- `examples/` — usage examples

## Key Rules
- ZERO external dependencies beyond mcp[cli] — no NLTK, no spaCy, no numpy
- All NLP algorithms implemented from scratch in pure Python
- Sentiment lexicon and stopwords are built into the code (not downloaded)
- Python 3.10+ required

## Development
```bash
pip install -e .
pytest tests/ -v
```
