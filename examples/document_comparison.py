"""Example: Comparing documents with blackmount-nlp-mcp.

Demonstrates text similarity, TF-IDF keyword extraction across multiple
documents, and sentiment comparison.
"""

from blackmount_nlp_mcp.similarity import (
    cosine_similarity_bow,
    edit_distance,
    jaccard_similarity,
    longest_common_subsequence,
    normalized_edit_distance,
)
from blackmount_nlp_mcp.keywords import tfidf_keywords
from blackmount_nlp_mcp.sentiment import sentiment_score, sentiment_label
from blackmount_nlp_mcp.summarize import text_statistics

# Three news-style paragraphs on related topics
doc_ai = """
Artificial intelligence continues to transform industries worldwide.
Companies are investing billions in AI research and development, with
applications ranging from healthcare diagnostics to autonomous vehicles.
The rapid advancement of large language models has sparked both excitement
and concern about the future of work and creativity.
"""

doc_climate = """
Climate change remains one of the most pressing challenges facing humanity.
Rising temperatures are causing more frequent extreme weather events,
threatening biodiversity, and impacting food security. International
cooperation and innovative technologies are essential for reducing
greenhouse gas emissions and building a sustainable future.
"""

doc_tech = """
The technology sector continues to evolve at a breakneck pace.
Artificial intelligence, cloud computing, and cybersecurity are driving
innovation across industries. Companies are investing heavily in digital
transformation, recognizing that technology is essential for remaining
competitive in the modern economy.
"""

documents = {
    "AI Article": doc_ai,
    "Climate Article": doc_climate,
    "Tech Article": doc_tech,
}


def main():
    print("=" * 60)
    print("DOCUMENT COMPARISON EXAMPLE")
    print("=" * 60)

    # 1. Pairwise similarity
    print("\n--- Pairwise Similarity ---")
    names = list(documents.keys())
    texts = list(documents.values())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            jaccard = jaccard_similarity(texts[i], texts[j])
            cosine = cosine_similarity_bow(texts[i], texts[j])
            print(f"\n  {names[i]} vs {names[j]}:")
            print(f"    Jaccard:  {jaccard:.4f}")
            print(f"    Cosine:   {cosine:.4f}")

    # 2. TF-IDF keywords across all documents
    print("\n--- TF-IDF Keywords (across all docs) ---")
    all_docs = list(documents.values())
    kws = tfidf_keywords(all_docs, top_n=15)
    for kw in kws:
        print(f"  {kw['term']}: {kw['score']:.4f}")

    # 3. Sentiment comparison
    print("\n--- Sentiment Comparison ---")
    for name, text in documents.items():
        score = sentiment_score(text)
        label = sentiment_label(text)
        print(f"  {name}: {score:+.4f} ({label})")

    # 4. Statistics comparison
    print("\n--- Statistics Comparison ---")
    print(f"  {'Document':<20} {'Words':>6} {'Sentences':>10} {'Avg Sent Len':>13} {'Reading Time':>13}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*13} {'-'*13}")
    for name, text in documents.items():
        stats = text_statistics(text)
        print(
            f"  {name:<20} {stats['words']:>6} {stats['sentences']:>10} "
            f"{stats['avg_sentence_length']:>13.1f} {stats['reading_time_minutes']:>10.2f} min"
        )

    # 5. String-level comparison
    print("\n--- Edit Distance (first sentences) ---")
    first_sents = {
        "AI": "Artificial intelligence continues to transform industries worldwide.",
        "Climate": "Climate change remains one of the most pressing challenges facing humanity.",
        "Tech": "The technology sector continues to evolve at a breakneck pace.",
    }
    keys = list(first_sents.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = edit_distance(first_sents[keys[i]], first_sents[keys[j]])
            norm = normalized_edit_distance(first_sents[keys[i]], first_sents[keys[j]])
            lcs = longest_common_subsequence(first_sents[keys[i]], first_sents[keys[j]])
            print(f"  {keys[i]} vs {keys[j]}: edit_dist={dist}, normalized={norm:.4f}, LCS={lcs}")


if __name__ == "__main__":
    main()
