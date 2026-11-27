"""
vector_store.py - ChromaDB Vector Store Module
================================
What this file does:
    Manages the ChromaDB vector database where chunks and their embeddings
    are stored. Provides functions to add, search, list, and delete chunks.
    The database is saved on disk (persistent) so data survives restarts.

Input:  Chunks with embeddings (for adding), query embeddings (for searching)
Output: Search results (list of matching chunks), file lists, counts
"""

import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME


# Create a persistent ChromaDB client that saves data to disk
# This runs ONCE when the module is imported
print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create the collection (like a table in a database)
# We use COSINE distance for similarity search, which works well for text embeddings
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # Use cosine similarity for distance
)
print(f"ChromaDB collection '{COLLECTION_NAME}' ready. Current chunks: {collection.count()}")


# ============================================================
# FUNCTION: is_document_already_added
# ============================================================
# What it does:
#   Checks if chunks from a specific file are already in the database.
#   This prevents adding the same file twice (duplicate prevention).
#
# Parameters:
#   file_name (str): The name of the file to check (e.g., "paper1.pdf")
# Returns:
#   True if the file's chunks already exist, False otherwise
def is_document_already_added(file_name):
    try:
        # Query ChromaDB for any chunks with this source file name
        results = collection.get(
            where={"source": file_name},
            limit=1,  # We only need to know if at least 1 exists
        )
        # If the ids list is not empty, the document has chunks stored
        if len(results["ids"]) > 0:
            print(f"  Document '{file_name}' is already in the database.")
            return True
        else:
            print(f"  Document '{file_name}' is NOT in the database.")
            return False
    except Exception as error:
        print(f"  Error checking document: {error}")
        return False


# ============================================================
# FUNCTION: add_chunks_to_store
# ============================================================
# What it does:
#   Adds chunks (text, embeddings, and metadata) to the ChromaDB collection.
#   Adds in batches of 100 to avoid memory issues with large files.
#
# Parameters:
#   list_of_chunks (list): List of dicts with "text", "location", "source", "chunk_id"
#   list_of_embeddings (list): List of embedding vectors (matching order of chunks)
# Returns:
#   Nothing (prints progress messages)
def add_chunks_to_store(list_of_chunks, list_of_embeddings):
    total_chunks = len(list_of_chunks)
    print(f"  Adding {total_chunks} chunks to ChromaDB...")

    batch_size = 100
    # Process chunks in batches
    for start_index in range(0, total_chunks, batch_size):
        end_index = start_index + batch_size
        if end_index > total_chunks:
            end_index = total_chunks

        # Prepare the batch data
        batch_ids = []
        batch_documents = []
        batch_embeddings = []
        batch_metadatas = []

        for i in range(start_index, end_index):
            chunk = list_of_chunks[i]
            batch_ids.append(chunk["chunk_id"])
            batch_documents.append(chunk["text"])
            batch_embeddings.append(list_of_embeddings[i])
            batch_metadatas.append({
                "source": chunk["source"],
                "location": chunk["location"],
            })

        # Add this batch to ChromaDB
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            print(f"    Added batch: chunks {start_index + 1} to {end_index}")
        except Exception as error:
            print(f"    Error adding batch {start_index + 1}-{end_index}: {error}")

    print(f"  Stored {total_chunks} chunks in the vector DB.")
    print(f"  Total chunks in collection now: {collection.count()}")


# ============================================================
# FUNCTION: search_store
# ============================================================
# What it does:
#   Searches the ChromaDB collection for chunks similar to the query.
#   Returns the top-k most similar chunks with their text, source,
#   location, and distance score.
#
# Parameters:
#   query_embedding (list): The embedding vector of the user's question
#   top_k (int): How many similar chunks to return
#   file_name_filter (str or None): If given, only search chunks from this file
# Returns:
#   A list of dictionaries with "text", "source", "location", "distance"
def search_store(query_embedding, top_k, file_name_filter=None):
    print(f"  Searching ChromaDB for top {top_k} similar chunks...")

    # Build the query parameters
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    # If a file filter is given, add a WHERE clause to only search that file
    if file_name_filter is not None:
        query_params["where"] = {"source": file_name_filter}
        print(f"  Filtering by source: {file_name_filter}")

    try:
        results = collection.query(**query_params)
    except Exception as error:
        print(f"  Error searching ChromaDB: {error}")
        return []

    # ChromaDB returns results in nested lists, so we need to extract them
    # results["documents"][0] = list of document texts
    # results["metadatas"][0] = list of metadata dicts
    # results["distances"][0] = list of distance scores
    list_of_results = []

    if results["documents"] and len(results["documents"][0]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):
            result = {
                "text": documents[i],
                "source": metadatas[i]["source"],
                "location": metadatas[i]["location"],
                "distance": distances[i],
            }
            list_of_results.append(result)

    print(f"  Found {len(list_of_results)} matching chunks.")
    return list_of_results


# ============================================================
# FUNCTION: list_all_documents
# ============================================================
# What it does:
#   Returns a sorted list of unique file names (sources) stored in the DB.
#   This is used to show the user which files have been indexed.
#
# Parameters: None
# Returns:
#   A sorted list of unique source file names (strings)
def list_all_documents():
    try:
        # Get all metadatas from the collection
        all_data = collection.get()
        all_metadatas = all_data["metadatas"]

        # Collect unique source names
        unique_sources = []
        for metadata in all_metadatas:
            source_name = metadata["source"]
            if source_name not in unique_sources:
                unique_sources.append(source_name)

        # Sort alphabetically
        unique_sources.sort()
        return unique_sources
    except Exception as error:
        print(f"  Error listing documents: {error}")
        return []


# ============================================================
# FUNCTION: delete_document
# ============================================================
# What it does:
#   Deletes ALL chunks belonging to a specific file from the database.
#
# Parameters:
#   file_name (str): The name of the file whose chunks should be deleted
# Returns:
#   Nothing (prints a confirmation message)
def delete_document(file_name):
    print(f"  Deleting all chunks for: {file_name}")

    try:
        # Get the IDs of all chunks with this source
        results = collection.get(
            where={"source": file_name},
        )
        chunk_ids = results["ids"]

        if len(chunk_ids) == 0:
            print(f"  No chunks found for '{file_name}'.")
            return

        # Delete all those chunks
        collection.delete(ids=chunk_ids)
        print(f"  Deleted {len(chunk_ids)} chunks for '{file_name}'.")
        print(f"  Remaining chunks in collection: {collection.count()}")
    except Exception as error:
        print(f"  Error deleting document: {error}")


# ============================================================
# FUNCTION: get_total_chunk_count
# ============================================================
# What it does:
#   Returns the total number of chunks stored in the database.
#
# Parameters: None
# Returns:
#   An integer - the total chunk count
def get_total_chunk_count():
    return collection.count()


# ============================================================
# Test block: Run this file alone to test vector store operations
# Usage: python vector_store.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VECTOR STORE MODULE TEST")
    print("=" * 60)

    # Create some fake chunks for testing
    fake_chunks = [
        {
            "text": "Machine learning is a method of data analysis.",
            "location": "Page 1",
            "source": "test_paper.pdf",
            "chunk_id": "test_paper.pdf_Page 1_c0",
        },
        {
            "text": "Neural networks are inspired by the human brain.",
            "location": "Page 2",
            "source": "test_paper.pdf",
            "chunk_id": "test_paper.pdf_Page 2_c0",
        },
        {
            "text": "Sales revenue was 1 million dollars in Q1.",
            "location": "Rows 1-20",
            "source": "test_data.csv",
            "chunk_id": "test_data.csv_Rows 1-20_c0",
        },
    ]

    # Create fake embeddings (384 dimensions each, filled with 0.1 for testing)
    fake_embeddings = []
    for i in range(len(fake_chunks)):
        # Each embedding is a list of 384 numbers
        fake_embedding = [0.1 + (i * 0.01)] * 384
        fake_embeddings.append(fake_embedding)

    # Test 1: Check if document exists
    print("\nTest 1: Check if document exists")
    exists = is_document_already_added("test_paper.pdf")
    print(f"  Result: {exists}")

    # Test 2: Add chunks (only if not already added)
    if not exists:
        print("\nTest 2: Adding fake chunks")
        add_chunks_to_store(fake_chunks, fake_embeddings)

    # Test 3: Get total count
    print(f"\nTest 3: Total chunks in DB: {get_total_chunk_count()}")

    # Test 4: List all documents
    print("\nTest 4: All documents in DB:")
    documents = list_all_documents()
    for doc in documents:
        print(f"  - {doc}")

    # Test 5: Search
    print("\nTest 5: Searching for similar chunks")
    # Use a fake query embedding
    fake_query_embedding = [0.1] * 384
    results = search_store(fake_query_embedding, top_k=2)
    for result in results:
        print(f"  Source: {result['source']}, Location: {result['location']}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text: {result['text'][:100]}")

    # Test 6: Delete a document
    print("\nTest 6: Deleting test_paper.pdf")
    delete_document("test_paper.pdf")

    # Test 7: Delete the other test document
    print("\nTest 7: Deleting test_data.csv")
    delete_document("test_data.csv")

    print(f"\nFinal chunk count: {get_total_chunk_count()}")
    print("\nVector store module test complete!")
