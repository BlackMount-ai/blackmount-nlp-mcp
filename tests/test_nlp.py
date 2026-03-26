"""Tests for blackmount-nlp-mcp using real text samples."""

import pytest

from blackmount_nlp_mcp.tokenize import (
    char_ngrams,
    ngrams,
    sentence_tokenize,
    word_tokenize,
)
from blackmount_nlp_mcp.readability import (
    automated_readability,
    coleman_liau,
    flesch_kincaid_grade,
    flesch_reading_ease,
    gunning_fog,
    reading_level,
    smog_grade,
    syllable_count,
)
from blackmount_nlp_mcp.sentiment import (
    aspect_sentiment,
    sentence_sentiments,
    sentiment_label,
    sentiment_score,
)
from blackmount_nlp_mcp.keywords import (
    phrase_frequency,
    rake_keywords,
    tfidf_keywords,
    word_frequency,
)
from blackmount_nlp_mcp.similarity import (
    cosine_similarity_bow,
    edit_distance,
    jaccard_similarity,
    longest_common_subsequence,
    normalized_edit_distance,
)
from blackmount_nlp_mcp.clean import (
    clean_text,
    lowercase,
    normalize_whitespace,
    remove_emails,
    remove_html,
    remove_numbers,
    remove_punctuation,
    remove_stopwords,
    remove_urls,
    stem,
)
from blackmount_nlp_mcp.detect import (
    avg_sentence_length,
    avg_word_length,
    detect_encoding_type,
    detect_language,
    is_english,
    paragraph_count,
    sentence_count,
    word_count,
)
from blackmount_nlp_mcp.summarize import extractive_summary, text_statistics


# Real text samples
GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that "
    "all men are created equal. Now we are engaged in a great civil war, testing "
    "whether that nation, or any nation so conceived and so dedicated, can long "
    "endure. We are met on a great battle-field of that war."
)

REVIEW_POSITIVE = (
    "This product is absolutely amazing! I love how easy it is to use. "
    "The quality is excellent and the customer service was incredibly helpful. "
    "I would highly recommend this to anyone looking for a great experience."
)

REVIEW_NEGATIVE = (
    "Terrible product, complete waste of money. It broke after two days and "
    "the customer service was rude and unhelpful. I'm extremely disappointed "
    "and would never buy from this company again. Absolutely horrible experience."
)

TECH_ARTICLE = (
    "Machine learning algorithms process large datasets to identify patterns "
    "and make predictions. Neural networks, inspired by the human brain, "
    "consist of interconnected nodes that learn from training data. "
    "Deep learning, a subset of machine learning, uses multiple layers "
    "of neural networks to analyze complex features. These technologies "
    "have revolutionized natural language processing, computer vision, "
    "and autonomous driving systems."
)


# ============================================================
# Tokenization Tests
# ============================================================

class TestTokenize:
    def test_word_tokenize_basic(self):
        tokens = word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_word_tokenize_contractions(self):
        tokens = word_tokenize("I don't think it's right")
        assert "don't" in tokens
        assert "it's" in tokens

    def test_word_tokenize_empty(self):
        assert word_tokenize("") == []
        assert word_tokenize("   ") == []

    def test_sentence_tokenize_basic(self):
        sents = sentence_tokenize("Hello world. How are you? I'm fine!")
        assert len(sents) == 3

    def test_sentence_tokenize_abbreviations(self):
        sents = sentence_tokenize("Dr. Smith went to Washington. He arrived at noon.")
        assert len(sents) == 2
        assert "Dr. Smith" in sents[0]

    def test_sentence_tokenize_gettysburg(self):
        sents = sentence_tokenize(GETTYSBURG)
        assert len(sents) >= 2  # At least 2 sentences

    def test_ngrams(self):
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        bigrams = ngrams(tokens, 2)
        assert ("the", "cat") in bigrams
        assert ("cat", "sat") in bigrams
        assert len(bigrams) == 5

    def test_ngrams_trigrams(self):
        tokens = ["a", "b", "c", "d"]
        trigrams = ngrams(tokens, 3)
        assert len(trigrams) == 2
        assert ("a", "b", "c") in trigrams

    def test_char_ngrams(self):
        result = char_ngrams("hello", 3)
        assert "hel" in result
        assert "ell" in result
        assert "llo" in result
        assert len(result) == 3


# ============================================================
# Readability Tests
# ============================================================

class TestReadability:
    def test_syllable_count(self):
        assert syllable_count("the") == 1
        assert syllable_count("hello") == 2
        assert syllable_count("beautiful") == 3
        assert syllable_count("extraordinary") >= 4

    def test_flesch_reading_ease(self):
        score = flesch_reading_ease(GETTYSBURG)
        assert isinstance(score, float)
        # Gettysburg address should be moderately readable
        assert 0 <= score <= 100

    def test_flesch_kincaid_grade(self):
        grade = flesch_kincaid_grade(GETTYSBURG)
        assert isinstance(grade, float)
        assert grade >= 0

    def test_gunning_fog(self):
        fog = gunning_fog(TECH_ARTICLE)
        assert isinstance(fog, float)
        # Technical article should have higher fog index
        assert fog > 5

    def test_coleman_liau(self):
        cli = coleman_liau(GETTYSBURG)
        assert isinstance(cli, float)
        assert cli >= 0

    def test_automated_readability(self):
        ari = automated_readability(GETTYSBURG)
        assert isinstance(ari, float)
        assert ari >= 0

    def test_smog_grade(self):
        smog = smog_grade(TECH_ARTICLE)
        assert isinstance(smog, float)
        assert smog > 0

    def test_reading_level(self):
        result = reading_level(GETTYSBURG)
        assert "grade_level" in result
        assert "label" in result
        assert result["label"] in ("elementary", "middle school", "high school", "college", "graduate")


# ============================================================
# Sentiment Tests
# ============================================================

class TestSentiment:
    def test_positive_review(self):
        score = sentiment_score(REVIEW_POSITIVE)
        assert score > 0.1
        assert sentiment_label(REVIEW_POSITIVE) == "positive"

    def test_negative_review(self):
        score = sentiment_score(REVIEW_NEGATIVE)
        assert score < -0.1
        assert sentiment_label(REVIEW_NEGATIVE) == "negative"

    def test_neutral_text(self):
        text = "The meeting is scheduled for Tuesday at 3pm in room 204."
        label = sentiment_label(text)
        assert label == "neutral"

    def test_negation_handling(self):
        # "not good" should be negative or at least less positive
        pos_score = sentiment_score("The food was good")
        neg_score = sentiment_score("The food was not good")
        assert neg_score < pos_score

    def test_capitalization_emphasis(self):
        normal = sentiment_score("This is amazing")
        caps = sentiment_score("This is AMAZING")
        assert caps >= normal  # CAPS should add emphasis

    def test_sentence_sentiments(self):
        text = "I love this product! But the packaging was terrible."
        results = sentence_sentiments(text)
        assert len(results) >= 2
        assert all("score" in r and "label" in r for r in results)

    def test_aspect_sentiment(self):
        text = (
            "The food was absolutely delicious and well-presented. "
            "However, the service was slow and the waiter was rude. "
            "The ambiance was pleasant and relaxing."
        )
        result = aspect_sentiment(text, ["food", "service", "ambiance"])
        assert "food" in result
        assert "service" in result
        assert result["food"]["label"] == "positive"
        assert result["service"]["label"] == "negative"

    def test_empty_text(self):
        assert sentiment_score("") == 0.0
        assert sentiment_label("") == "neutral"


# ============================================================
# Keyword Tests
# ============================================================

class TestKeywords:
    def test_tfidf_single_doc(self):
        results = tfidf_keywords([TECH_ARTICLE], top_n=5)
        assert len(results) > 0
        assert all("term" in r and "score" in r for r in results)

    def test_tfidf_multiple_docs(self):
        results = tfidf_keywords([GETTYSBURG, TECH_ARTICLE, REVIEW_POSITIVE], top_n=10)
        assert len(results) > 0

    def test_rake_keywords(self):
        results = rake_keywords(TECH_ARTICLE, top_n=5)
        assert len(results) > 0
        assert all("phrase" in r and "score" in r for r in results)

    def test_word_frequency(self):
        results = word_frequency(GETTYSBURG, top_n=5)
        assert len(results) > 0
        assert all("word" in r and "count" in r for r in results)

    def test_phrase_frequency(self):
        text = "the big dog chased the big cat. the big dog ran away."
        results = phrase_frequency(text, n=2, top_n=5)
        assert len(results) > 0
        # "the big" or "big dog" should be frequent
        phrases = [r["phrase"] for r in results]
        assert any("big" in p for p in phrases)


# ============================================================
# Similarity Tests
# ============================================================

class TestSimilarity:
    def test_jaccard_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_different(self):
        score = jaccard_similarity("hello world", "goodbye moon")
        assert score == 0.0

    def test_jaccard_partial(self):
        score = jaccard_similarity("the cat sat on the mat", "the dog sat on the rug")
        assert 0 < score < 1

    def test_cosine_identical(self):
        assert cosine_similarity_bow("hello world", "hello world") == 1.0

    def test_cosine_different(self):
        score = cosine_similarity_bow("hello world", "goodbye moon")
        assert score == 0.0

    def test_edit_distance(self):
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "abc") == 0

    def test_normalized_edit_distance(self):
        dist = normalized_edit_distance("cat", "hat")
        assert 0 < dist < 1
        assert normalized_edit_distance("abc", "abc") == 0.0

    def test_lcs(self):
        assert longest_common_subsequence("ABCBDAB", "BDCAB") == 4
        assert longest_common_subsequence("", "abc") == 0
        assert longest_common_subsequence("abc", "abc") == 3


# ============================================================
# Cleaning Tests
# ============================================================

class TestClean:
    def test_remove_stopwords(self):
        result = remove_stopwords("the cat sat on the mat")
        assert "the" not in result.lower().split()
        assert "cat" in result.lower()

    def test_remove_punctuation(self):
        assert remove_punctuation("Hello, world!") == "Hello world"

    def test_remove_numbers(self):
        assert "42" not in remove_numbers("The answer is 42.")

    def test_remove_urls(self):
        result = remove_urls("Visit https://example.com for info")
        assert "https://example.com" not in result

    def test_remove_emails(self):
        result = remove_emails("Contact test@example.com for info")
        assert "test@example.com" not in result

    def test_remove_html(self):
        result = remove_html("<p>Hello <b>world</b></p>")
        assert "<p>" not in result
        assert "Hello" in result

    def test_normalize_whitespace(self):
        assert normalize_whitespace("hello   world\n\ntest") == "hello world test"

    def test_lowercase(self):
        assert lowercase("Hello World") == "hello world"

    def test_stem(self):
        assert stem("running") == "run"
        assert stem("cats") == "cat"
        assert stem("happily") not in ("happily",)  # Should be stemmed
        assert stem("connected") == "connect"

    def test_clean_pipeline(self):
        text = "<p>Visit https://example.com! Contact: test@example.com</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "https://" not in result
        assert "test@" not in result

    def test_clean_pipeline_custom_steps(self):
        result = clean_text("Hello World! 123", steps=["lowercase", "numbers"])
        assert result == "hello world! "


# ============================================================
# Detection Tests
# ============================================================

class TestDetect:
    def test_detect_english(self):
        results = detect_language(GETTYSBURG)
        assert results[0]["language"] == "english"

    def test_detect_spanish(self):
        text = "Hola, como estas? Estoy muy bien, gracias por preguntar."
        results = detect_language(text)
        assert results[0]["language"] == "spanish"

    def test_detect_french(self):
        text = "Bonjour, comment allez-vous? Je suis tres bien, merci. Elle est dans la maison avec les enfants."
        results = detect_language(text)
        top_languages = [r["language"] for r in results[:3]]
        assert "french" in top_languages

    def test_detect_encoding_ascii(self):
        assert detect_encoding_type("Hello world") == "ASCII"

    def test_detect_encoding_cyrillic(self):
        assert detect_encoding_type("Привет мир") == "Cyrillic"

    def test_is_english(self):
        assert is_english(GETTYSBURG) > 0.5
        assert is_english("Bonjour comment allez vous je suis bien") < 0.5

    def test_word_count(self):
        assert word_count("Hello beautiful world") == 3

    def test_sentence_count(self):
        assert sentence_count("Hello. World. Test.") == 3

    def test_paragraph_count(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert paragraph_count(text) == 3

    def test_avg_word_length(self):
        avg = avg_word_length("Hi there everyone")
        assert avg > 0

    def test_avg_sentence_length(self):
        avg = avg_sentence_length("Hello world. This is a test.")
        assert avg > 0


# ============================================================
# Summarization Tests
# ============================================================

class TestSummarize:
    def test_extractive_summary(self):
        summary = extractive_summary(TECH_ARTICLE, n_sentences=2)
        assert len(summary) > 0
        # Summary should be shorter than original
        assert len(summary) < len(TECH_ARTICLE)

    def test_extractive_summary_with_title(self):
        summary = extractive_summary(
            TECH_ARTICLE, n_sentences=2, title="Machine Learning Overview"
        )
        assert len(summary) > 0

    def test_extractive_summary_short_text(self):
        # If text has fewer sentences than requested, return all
        short = "Just one sentence."
        assert extractive_summary(short, n_sentences=3) == short

    def test_text_statistics(self):
        stats = text_statistics(TECH_ARTICLE)
        assert stats["words"] > 0
        assert stats["sentences"] > 0
        assert stats["reading_time_minutes"] > 0
        assert "readability" in stats
        assert stats["language"] == "english"

    def test_text_statistics_empty(self):
        stats = text_statistics("")
        assert stats["words"] == 0
        assert stats["sentences"] == 0
