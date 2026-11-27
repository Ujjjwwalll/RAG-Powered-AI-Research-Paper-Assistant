"""
embedder.py - Text Embedding Module
================================
What this file does:
    Uses the sentence-transformers library to convert text into numerical
    vectors (embeddings). These embeddings capture the meaning of the text
    so that similar texts have similar vectors.

Input:  Text strings (single or list)
Output: Embedding vectors (lists of 384 floating-point numbers)
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

# Load the embedding model ONCE at module level.
# This means the model is loaded into memory when this file is first imported,
# and it stays loaded. This is much faster than loading it every time
# we need to embed something.
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Embedding model loaded successfully! (Dimension: {embedding_model.get_sentence_embedding_dimension()})")


# ============================================================
# FUNCTION: embed_list_of_texts
# ============================================================
# What it does:
#   Takes a list of text strings and returns a list of embeddings.
#   Each embedding is a plain Python list of floating-point numbers.
#
# Parameters:
#   list_of_texts (list): A list of strings to embed
# Returns:
#   list_of_embeddings (list): A list of lists, each inner list has 384 numbers
def embed_list_of_texts(list_of_texts):
    print(f"  Embedding {len(list_of_texts)} texts...")

    # The model returns numpy arrays, but ChromaDB needs plain Python lists
    # So we convert each numpy array to a plain list using a for loop
    numpy_embeddings = embedding_model.encode(list_of_texts, show_progress_bar=False)

    list_of_embeddings = []
    for numpy_array in numpy_embeddings:
        # Convert numpy array to a regular Python list of floats
        python_list = numpy_array.tolist()
        list_of_embeddings.append(python_list)

    print(f"  Embedding complete: {len(list_of_embeddings)} embeddings created.")
    return list_of_embeddings


# ============================================================
# FUNCTION: embed_single_query
# ============================================================
# What it does:
#   Takes a single text string (like a user's question) and returns
#   its embedding as a plain Python list.
#
# Parameters:
#   query_text (str): A single string to embed
# Returns:
#   embedding (list): A list of 384 floating-point numbers
def embed_single_query(query_text):
    print(f"  Embedding query: '{query_text[:80]}...'")

    # Encode returns a numpy array for a single string
    numpy_embedding = embedding_model.encode(query_text)

    # Convert to a plain Python list
    embedding = numpy_embedding.tolist()

    print(f"  Query embedding created (dimension: {len(embedding)}).")
    return embedding


# ============================================================
# Test block: Run this file alone to test embedding
# Usage: python embedder.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDER MODULE TEST")
    print("=" * 60)

    # Test 1: Embed a single query
    test_query = "What algorithm was used in the research paper?"
    query_embedding = embed_single_query(test_query)
    print(f"\nTest 1 - Single query embedding:")
    print(f"  Query: {test_query}")
    print(f"  Embedding length: {len(query_embedding)} (should be 384)")
    print(f"  First 5 values: {query_embedding[:5]}")

    # Test 2: Embed a list of texts
    test_texts = [
        "Machine learning uses data to learn patterns.",
        "The study was conducted on 1000 patients.",
        "Python is a popular programming language.",
    ]
    embeddings = embed_list_of_texts(test_texts)
    print(f"\nTest 2 - Batch embedding:")
    print(f"  Number of texts: {len(test_texts)}")
    print(f"  Number of embeddings: {len(embeddings)}")
    for i in range(len(embeddings)):
        print(f"  Embedding {i+1} length: {len(embeddings[i])}")

    print("\nEmbedder module test complete!")
