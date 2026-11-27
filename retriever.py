"""
retriever.py - Retrieval Module
================================
What this file does:
    Takes a search question, embeds it using the same embedding model
    that was used for the chunks, fetches candidate chunks from ChromaDB,
    reranks them with the CrossEncoder reranker, and returns the top results.

Input:  A question string, optional file_name_filter
Output: A list of dictionaries with "text", "source", "location", "distance", "relevance_percent"
"""

from embedder import embed_single_query
from vector_store import search_store
from reranker import rerank_chunks
from config import TOP_K_RETRIEVE, TOP_K_FINAL


# ============================================================
# FUNCTION: retrieve_relevant_chunks
# ============================================================
# What it does:
#   Step 1: Embeds the question and fetches TOP_K_RETRIEVE candidate chunks from ChromaDB.
#   Step 2: Reranks the candidates with rerank_chunks and keeps TOP_K_FINAL chunks.
#
# Parameters:
#   question (str): The search question (can be expanded)
#   file_name_filter (str or None): If given, only search in this file
# Returns:
#   A list of dicts with "text", "source", "location", "distance", "relevance_percent"
def retrieve_relevant_chunks(question, file_name_filter=None):
    print(f"\n--- Retrieval: Two-step search + rerank ---")
    print(f"  Question: {question[:100]}...")
    print(f"  Step 1 fetch: {TOP_K_RETRIEVE}, Step 2 keep: {TOP_K_FINAL}")

    if file_name_filter:
        print(f"  Filtering by file: {file_name_filter}")

    # Step 1: Embed the question using the SAME model used for chunks
    question_embedding = embed_single_query(question)

    # Step 2: Fetch TOP_K_RETRIEVE candidate chunks from ChromaDB
    candidate_chunks = search_store(question_embedding, TOP_K_RETRIEVE, file_name_filter)
    print(f"  Fetched {len(candidate_chunks)} candidate chunks from ChromaDB.")

    if len(candidate_chunks) == 0:
        return []

    # Step 3: Rerank the candidates and keep only TOP_K_FINAL chunks
    final_chunks = rerank_chunks(question, candidate_chunks, TOP_K_FINAL)

    print(f"  Returning {len(final_chunks)} reranked chunks.")
    return final_chunks


# ============================================================
# Test block: Run this file alone to test retrieval
# Usage: python retriever.py
# NOTE: This test requires that some chunks are already stored in ChromaDB.
#       Run vector_store.py or pipeline.py first to add some data.
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RETRIEVER MODULE TEST")
    print("=" * 60)

    from vector_store import get_total_chunk_count, list_all_documents

    # Check if there are any chunks in the database
    total_chunks = get_total_chunk_count()
    print(f"\nChunks in database: {total_chunks}")

    if total_chunks == 0:
        print("No chunks found in the database.")
        print("Please run pipeline.py first to ingest some files.")
        print("Or run vector_store.py to add test chunks.")
    else:
        # Show what documents are available
        documents = list_all_documents()
        print(f"Available documents: {documents}")

        # Test with a sample question
        test_question = "What is the main topic of this document?"
        print(f"\nTest question: {test_question}")

        results = retrieve_relevant_chunks(test_question)

        print(f"\n--- Results ---")
        for i in range(len(results)):
            result = results[i]
            print(f"\nResult {i+1}:")
            print(f"  Source: {result['source']}")
            print(f"  Location: {result['location']}")
            print(f"  Relevance: {result['relevance_percent']}%")
            print(f"  Text preview: {result['text'][:200]}...")

    print("\nRetriever module test complete!")
