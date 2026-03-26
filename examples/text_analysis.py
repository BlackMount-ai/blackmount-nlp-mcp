"""Example: Comprehensive text analysis with blackmount-nlp-mcp.

Demonstrates readability scoring, sentiment analysis, keyword extraction,
and text statistics — all with zero heavy dependencies.
"""

from blackmount_nlp_mcp.readability import reading_level
from blackmount_nlp_mcp.sentiment import sentiment_label, sentiment_score, sentence_sentiments
from blackmount_nlp_mcp.keywords import rake_keywords, word_frequency
from blackmount_nlp_mcp.summarize import extractive_summary, text_statistics
from blackmount_nlp_mcp.clean import clean_text

# Sample: product review
review = """
I've been using this laptop for about three months now and I have to say,
it's been a mixed experience. The display is absolutely gorgeous — sharp,
vibrant colors, and the 120Hz refresh rate makes everything feel smooth.
Battery life is also impressive, easily lasting 10-12 hours on a single charge.

However, the keyboard is disappointing. The keys feel mushy and lack the
satisfying tactile feedback I was hoping for. The trackpad is also too small
and occasionally registers phantom clicks. For a laptop at this price point,
I expected better build quality.

The performance is where this machine really shines though. It handles
multitasking with ease, and even light video editing runs without a hitch.
The speakers are surprisingly good for a laptop this thin.

Overall, I'd give it a 7/10. Great performance and display, but the input
devices need serious improvement. Would recommend if you primarily use an
external keyboard and mouse.
"""

def main():
    print("=" * 60)
    print("TEXT ANALYSIS EXAMPLE")
    print("=" * 60)

    # 1. Text Statistics
    print("\n--- Text Statistics ---")
    stats = text_statistics(review)
    print(f"Words: {stats['words']}")
    print(f"Sentences: {stats['sentences']}")
    print(f"Paragraphs: {stats['paragraphs']}")
    print(f"Reading time: {stats['reading_time_minutes']} minutes")
    print(f"Language: {stats['language']}")

    # 2. Readability
    print("\n--- Readability ---")
    level = reading_level(review)
    print(f"Grade level: {level['grade_level']}")
    print(f"Reading level: {level['label']}")
    print(f"Flesch Reading Ease: {level['flesch_reading_ease']}")
    print(f"Gunning Fog: {level['gunning_fog']}")

    # 3. Sentiment
    print("\n--- Overall Sentiment ---")
    score = sentiment_score(review)
    label = sentiment_label(review)
    print(f"Score: {score}")
    print(f"Label: {label}")

    print("\n--- Per-Sentence Sentiment ---")
    for item in sentence_sentiments(review)[:5]:
        sent_preview = item["sentence"][:60] + "..." if len(item["sentence"]) > 60 else item["sentence"]
        print(f"  [{item['label']:>8}] ({item['score']:+.3f}) {sent_preview}")

    # 4. Keywords
    print("\n--- Top Keywords (RAKE) ---")
    for kw in rake_keywords(review, top_n=8):
        print(f"  {kw['phrase']} (score: {kw['score']:.2f})")

    print("\n--- Word Frequency ---")
    for wf in word_frequency(review, top_n=10):
        print(f"  {wf['word']}: {wf['count']}")

    # 5. Summary
    print("\n--- Extractive Summary (3 sentences) ---")
    summary = extractive_summary(review, n_sentences=3, title="Laptop Review")
    print(summary)

    # 6. Cleaned text
    print("\n--- Cleaned Text (first 200 chars) ---")
    cleaned = clean_text(review, steps=["lowercase", "urls", "punctuation", "whitespace"])
    print(cleaned[:200] + "...")


if __name__ == "__main__":
    main()
