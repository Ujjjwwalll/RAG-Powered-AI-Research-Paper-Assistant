"""
pipeline.py - Main Pipeline Module
================================
What this file does:
    Connects ALL the backend modules together into two main workflows:
    1. ingest_file(): Load -> Chunk -> Embed -> Store (for uploading files)
    2. ask_question(): Retrieve -> Augment -> Generate (for answering questions)

    This is the ONLY module that app.py needs to call for the main operations.

Input:  File paths (for ingestion), question strings (for Q&A)
Output: Status dictionaries (for ingestion), answer dictionaries (for Q&A)
"""

import os

from loader import load_file
from chunker import split_pages_into_chunks
from embedder import embed_list_of_texts
from vector_store import is_document_already_added, add_chunks_to_store
from retriever import retrieve_relevant_chunks
from prompt_builder import build_final_prompt
from llm import generate_answer


# ============================================================
# FUNCTION: ingest_file
# ============================================================
# What it does:
#   Takes a file path and runs the full ingestion pipeline:
#   Load the file -> Split into chunks -> Embed chunks -> Store in ChromaDB.
#   Returns a status dictionary with details about what happened.
#
# Parameters:
#   file_path (str): The full path to the file to ingest
# Returns:
#   A dictionary with keys:
#     "file_name" - name of the file
#     "file_type" - extension of the file
#     "entries"   - number of entries the loader created
#     "chunks"    - number of chunks after splitting
#     "status"    - "added", "skipped", or "failed"
#     "message"   - a human-readable status message
def ingest_file(file_path):
    # Get the file name and extension
    file_name = os.path.basename(file_path)

    # Get the file extension (lowercase, without the dot)
    if "." in file_name:
        file_type = file_name.rsplit(".", 1)[1].lower()
    else:
        file_type = "unknown"

    print(f"\n{'='*60}")
    print(f"INGESTING FILE: {file_name}")
    print(f"{'='*60}")

    # Step 0: Check if this file is already in the database
    already_exists = is_document_already_added(file_name)
    if already_exists:
        message = f"'{file_name}' is already indexed. Skipping to avoid duplicates."
        print(f"  {message}")
        return {
            "file_name": file_name,
            "file_type": file_type,
            "entries": 0,
            "chunks": 0,
            "status": "skipped",
            "message": message,
        }

    # Step 1: LOAD the file using the appropriate loader
    print(f"\nStep 1: Loading file...")
    list_of_entries = load_file(file_path)

    # Check if the loader returned anything
    if len(list_of_entries) == 0:
        message = f"Could not extract any text from '{file_name}'. The file may be empty, corrupted, or unsupported."
        print(f"  {message}")
        return {
            "file_name": file_name,
            "file_type": file_type,
            "entries": 0,
            "chunks": 0,
            "status": "failed",
            "message": message,
        }

    number_of_entries = len(list_of_entries)
    print(f"  Loaded {number_of_entries} entries from {file_name}.")

    # Step 2: CHUNK the entries into smaller overlapping pieces
    print(f"\nStep 2: Chunking text...")
    list_of_chunks = split_pages_into_chunks(list_of_entries)

    if len(list_of_chunks) == 0:
        message = f"Chunking produced no chunks for '{file_name}'. The extracted text may be too short."
        print(f"  {message}")
        return {
            "file_name": file_name,
            "file_type": file_type,
            "entries": number_of_entries,
            "chunks": 0,
            "status": "failed",
            "message": message,
        }

    number_of_chunks = len(list_of_chunks)
    print(f"  Created {number_of_chunks} chunks.")

    # Step 3: EMBED the chunks using sentence-transformers
    print(f"\nStep 3: Embedding chunks...")
    # Collect just the text from each chunk for embedding
    chunk_texts = []
    for chunk in list_of_chunks:
        chunk_texts.append(chunk["text"])

    list_of_embeddings = embed_list_of_texts(chunk_texts)
    print(f"  Created {len(list_of_embeddings)} embeddings.")

    # Step 4: STORE the chunks and embeddings in ChromaDB
    print(f"\nStep 4: Storing in ChromaDB...")
    add_chunks_to_store(list_of_chunks, list_of_embeddings)

    message = f"Successfully indexed '{file_name}': {number_of_entries} entries -> {number_of_chunks} chunks."
    print(f"\n  {message}")

    return {
        "file_name": file_name,
        "file_type": file_type,
        "entries": number_of_entries,
        "chunks": number_of_chunks,
        "status": "added",
        "message": message,
    }


# ============================================================
# FUNCTION: expand_question_for_search
# ============================================================
# What it does:
#   Expands generic/vague questions by appending relevant search keywords
#   so that embedding search and reranking find the right sections.
#   The original question is still used for the final LLM prompt and UI.
#
# Parameters:
#   question (str): The user's original question
# Returns:
#   expanded_question (str): The question with helpful search terms added
def expand_question_for_search(question):
    question_lower = question.lower()
    expanded_question = question

    if "problem" in question_lower:
        expanded_question = expanded_question + " abstract introduction motivation limitations of existing approaches we propose"

    if "algorithm" in question_lower or "method" in question_lower or "architecture" in question_lower:
        expanded_question = expanded_question + " model architecture approach proposed method we use"

    if "dataset" in question_lower or "data" in question_lower:
        expanded_question = expanded_question + " dataset training data experiments evaluation benchmark"

    if "limitation" in question_lower:
        expanded_question = expanded_question + " limitations future work drawbacks conclusion"

    if "summar" in question_lower:
        expanded_question = expanded_question + " abstract introduction conclusion contributions results"

    if expanded_question != question:
        print(f"  Expanded search query: {expanded_question}")

    return expanded_question


# ============================================================
# FUNCTION: ask_question
# ============================================================
# What it does:
#   Takes a user's question, expands it for better search, retrieves and
#   reranks relevant chunks from ChromaDB, builds a prompt with the
#   ORIGINAL question, and sends it to the LLM for an answer.
#
# Parameters:
#   question (str): The user's question
#   file_name_filter (str or None): If given, only search in this file
# Returns:
#   A dictionary with keys:
#     "answer"  - the LLM's response text
#     "sources" - the list of retrieved chunk dictionaries
def ask_question(question, file_name_filter=None):
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Expand the question for retrieval and reranking ONLY
    search_question = expand_question_for_search(question)

    # Step 1: RETRIEVE relevant chunks from ChromaDB (with reranking)
    print(f"\nStep 1: Retrieving relevant chunks...")
    list_of_retrieved_chunks = retrieve_relevant_chunks(
        search_question,
        file_name_filter=file_name_filter,
    )

    # Check if any chunks were found
    if len(list_of_retrieved_chunks) == 0:
        no_docs_message = ("I could not find any relevant information in the uploaded files. "
                           "Please make sure you have uploaded and processed some files first.")
        print(f"  No relevant chunks found.")
        return {
            "answer": no_docs_message,
            "sources": [],
        }

    # Step 2: AUGMENT - Build the prompt with context and the ORIGINAL question
    print(f"\nStep 2: Building prompt...")
    final_prompt = build_final_prompt(question, list_of_retrieved_chunks)
    print(f"  Prompt built ({len(final_prompt)} characters).")

    # Step 3: GENERATE - Send the prompt to the LLM
    print(f"\nStep 3: Generating answer...")
    answer = generate_answer(final_prompt)

    print(f"\n--- Answer generated ---")
    print(f"  Answer preview: {answer[:200]}...")

    return {
        "answer": answer,
        "sources": list_of_retrieved_chunks,
    }


# ============================================================
# Test block: Run this file alone to test the full pipeline
# Usage: python pipeline.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PIPELINE MODULE TEST - Full Flow")
    print("=" * 60)

    from vector_store import list_all_documents, get_total_chunk_count

    # Check the data/uploads folder for files to test with
    test_folder = "data/uploads"

    if not os.path.isdir(test_folder):
        os.makedirs(test_folder, exist_ok=True)
        print(f"\nCreated {test_folder}/ folder.")
        print("Please place at least one PDF or CSV file there and run again.")
    else:
        test_files = os.listdir(test_folder)
        if len(test_files) == 0:
            print(f"\nNo files found in {test_folder}/.")
            print("Please place at least one PDF or CSV file there and run again.")
        else:
            print(f"\nFound {len(test_files)} files in {test_folder}/:")
            for f in test_files:
                print(f"  - {f}")

            # Ingest one PDF and one CSV if available
            pdf_file = None
            csv_file = None
            for f in test_files:
                if f.lower().endswith(".pdf") and pdf_file is None:
                    pdf_file = f
                elif f.lower().endswith(".csv") and csv_file is None:
                    csv_file = f

            # Create a small sample CSV in data/uploads if none exists
            if csv_file is None:
                sample_csv_name = "sample_experiments.csv"
                sample_csv_path = os.path.join(test_folder, sample_csv_name)
                with open(sample_csv_path, "w", encoding="utf-8") as csv_f:
                    csv_f.write("Model,Dataset,BLEU_Score,Training_Days\n")
                    csv_f.write("Transformer Base,WMT 2014 EN-DE,27.3,3.5\n")
                    csv_f.write("Transformer Big,WMT 2014 EN-FR,41.8,3.5\n")
                csv_file = sample_csv_name
                print(f"\nCreated sample CSV for testing: {sample_csv_name}")

            files_to_ingest = []
            if pdf_file is not None:
                files_to_ingest.append(pdf_file)
            if csv_file is not None:
                files_to_ingest.append(csv_file)

            for file_to_ingest in files_to_ingest:
                file_path = os.path.join(test_folder, file_to_ingest)
                print(f"\n--- Ingesting: {file_to_ingest} ---")
                result = ingest_file(file_path)
                print(f"  Status: {result['status']} | {result['message']}")

            # Show what's in the database
            print(f"\n--- Database Status ---")
            print(f"  Total chunks: {get_total_chunk_count()}")
            print(f"  Documents: {list_all_documents()}")

            # Ask 3 test questions
            test_questions = [
                "What problem does this paper solve?",
                "Which algorithm or architecture was used?",
                "What dataset and BLEU score are shown in the CSV?",
            ]

            for q in test_questions:
                print(f"\n--- Asking: {q} ---")
                qa_result = ask_question(q)
                print(f"\n=== ANSWER ===")
                print(qa_result["answer"])
                print(f"\n=== SOURCES ({len(qa_result['sources'])}) ===")
                for i in range(len(qa_result["sources"])):
                    source = qa_result["sources"][i]
                    print(f"  Source {i+1}: {source['source']} | {source['location']} | Relevance: {source.get('relevance_percent', 0)}%")

    print("\nPipeline module test complete!")
