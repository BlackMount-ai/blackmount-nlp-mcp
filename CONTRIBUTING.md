# Contributing to blackmount-nlp-mcp

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/BlackMount-ai/blackmount-nlp-mcp
cd blackmount-nlp-mcp
pip install -e .
pytest tests/ -v
```

## Adding a New Tool

1. **Implement the function** in the appropriate domain module (`sentiment.py`, `readability.py`, `keywords.py`, etc.). Keep it pure Python — stdlib + regex only.
2. **Register it in `server.py`** as a one-line `@mcp.tool()` wrapper that delegates to your function.
3. **Add tests** in `tests/test_nlp.py` covering: normal input, empty input, unicode, edge cases.
4. **Update the README** tool catalog with the new tool name and description.

## The One Rule

**Zero heavy dependencies.** No NLTK, no spaCy, no transformers, no numpy, no scikit-learn. The only runtime dependency is `mcp[cli]`. This constraint is the product — every algorithm is implemented from scratch in pure Python.

If a tool genuinely needs something beyond stdlib + regex, open an issue to discuss before implementing.

## Extending the Lexicon

The sentiment lexicon (`_lexicon.py`) and stopwords (`_stopwords.py`) are vendored as Python literals. To add words:

1. Add entries to the appropriate dict in `_lexicon.py`
2. Verify no duplicate keys exist
3. Run the sentiment tests to ensure scores still make sense

## Running Tests

```bash
pytest tests/ -v                    # full suite
pytest tests/test_nlp.py::TestSentiment -v   # one category
```

## Pull Requests

- One feature per PR
- Tests must pass on Python 3.10-3.13
- Commit messages should be clear and descriptive
