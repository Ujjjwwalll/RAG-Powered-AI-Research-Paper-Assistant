"""
reranker.py - Cross-Encoder Reranking Module
================================
What this file does:
    Takes a question and a list of candidate chunks retrieved from ChromaDB,
    scores each (question, chunk) pair using a CrossEncoder model, converts
    the score to a 0-100 relevance percentage, and returns the top N chunks
    sorted from highest to lowest relevance.

Input:  A question string, a list of chunk dictionaries, and top_n (int)
Output: The top_n chunk dictionaries, each with "relevance_percent" added
"""

import math
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME

# Load the CrossEncoder reranker model ONCE at module level.
# Note: The model (~90 MB) will download automatically on the first run.
print(f"Loading reranker model: {RERANKER_MODEL_NAME} ...")
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
print("Reranker model loaded successfully!")


# ============================================================
# FUNCTION: rerank_chunks
# ============================================================
# What it does:
#   Scores each chunk against the question using the CrossEncoder model,
#   adds "relevance_percent" (0-100) to each chunk, sorts from highest to
#   lowest relevance, and returns the top_n chunks.
#
# Parameters:
#   question (str): The search question
#   list_of_chunks (list): List of chunk dicts from vector_store.search_store
#   top_n (int): How many top chunks to keep after reranking
# Returns:
#   top_chunks (list): Top N chunk dicts sorted by relevance_percent (high to low)
def rerank_chunks(question, list_of_chunks, top_n):
    if len(list_of_chunks) == 0:
        return []

    print(f"  Reranking {len(list_of_chunks)} chunks down to top {top_n}...")

    # Print the order BEFORE reranking
    print("  Order BEFORE reranking:")
    for i in range(len(list_of_chunks)):
        chunk = list_of_chunks[i]
        print(f"    {i + 1}. {chunk['source']} - {chunk['location']} (distance: {chunk['distance']:.4f})")

    # Step 1: Build a list of [question, chunk_text] pairs using a simple for loop
    list_of_pairs = []
    for chunk in list_of_chunks:
        pair = [question, chunk["text"]]
        list_of_pairs.append(pair)

    # Step 2: Score every pair with the CrossEncoder model
    raw_scores = reranker_model.predict(list_of_pairs)

    # Step 3 & 4: Convert each raw score to a percentage with sigmoid and store it
    for i in range(len(list_of_chunks)):
        raw_score = float(raw_scores[i])
        relevance_percent = round(100 / (1 + math.exp(-raw_score)))
        list_of_chunks[i]["relevance_percent"] = relevance_percent

    # Step 5: Sort chunks from highest to lowest relevance_percent (bubble sort)
    sorted_chunks = list(list_of_chunks)
    total = len(sorted_chunks)
    for i in range(total):
        for j in range(0, total - i - 1):
            if sorted_chunks[j]["relevance_percent"] < sorted_chunks[j + 1]["relevance_percent"]:
                temp = sorted_chunks[j]
                sorted_chunks[j] = sorted_chunks[j + 1]
                sorted_chunks[j + 1] = temp

    # Step 6: Keep only the top_n chunks
    top_chunks = sorted_chunks[:top_n]

    # Print the order AFTER reranking
    print("  Order AFTER reranking:")
    for i in range(len(top_chunks)):
        chunk = top_chunks[i]
        print(f"    {i + 1}. {chunk['source']} - {chunk['location']} (relevance: {chunk['relevance_percent']}%)")

    return top_chunks


# ============================================================
# Test block: Run this file alone to test reranking
# Usage: python reranker.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RERANKER MODULE TEST")
    print("=" * 60)

    sample_question = "Which algorithm did the authors propose for machine translation?"

    sample_chunks = [
        {
            "text": "The weather in Paris is sunny during spring and autumn.",
            "source": "travel.txt",
            "location": "Lines 1-10",
            "distance": 0.65,
        },
        {
            "text": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms for sequence transduction and machine translation.",
            "source": "paper1.pdf",
            "location": "Page 1",
            "distance": 0.42,
        },
        {
            "text": "Table 2 lists the hardware specifications of the eight GPUs used during training.",
            "source": "paper1.pdf",
            "location": "Page 7",
            "distance": 0.40,
        },
        {
            "text": "Self-attention replaces recurrence in our proposed Transformer model, allowing parallel computation.",
            "source": "paper1.pdf",
            "location": "Page 2",
            "distance": 0.45,
        },
    ]

    print(f"\nQuestion: {sample_question}\n")
    reranked = rerank_chunks(sample_question, sample_chunks, top_n=3)

    print(f"\nTop {len(reranked)} Reranked Chunks:")
    for i in range(len(reranked)):
        item = reranked[i]
        print(f"  #{i + 1}: {item['source']} ({item['location']}) -> {item['relevance_percent']}%")
        print(f"       {item['text']}")

    print("\nReranker module test complete!")
