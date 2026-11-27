"""
chunker.py - Text Chunking Module
================================
What this file does:
    Takes the list of entries from the loader (each entry has text, location,
    and source) and splits each entry's text into smaller overlapping chunks.
    Each chunk keeps the original metadata (location and source) so we always
    know where it came from.

Input:  A list of dictionaries from the loader, each with "text", "location", "source"
Output: A list of dictionaries, each with "text", "location", "source", "chunk_id"
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# FUNCTION: clean_text
# ============================================================
# What it does:
#   Cleans messy PDF text (for example, one word per line) by replacing
#   newlines with spaces and collapsing repeated spaces into single spaces.
#
# Parameters:
#   raw_text (str): The raw extracted text from a PDF page
# Returns:
#   cleaned_text (str): Clean text with single spaces
def clean_text(raw_text):
    # Replace newline characters with a single space
    text_with_spaces = raw_text.replace("\n", " ")

    # Collapse repeated spaces, tabs, etc. into single spaces
    cleaned_text = " ".join(text_with_spaces.split())

    return cleaned_text


# ============================================================
# FUNCTION: split_pages_into_chunks
# ============================================================
# What it does:
#   Takes the list of entries from the loader and splits each entry's text
#   into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
#
# WHY OVERLAP?
#   When we split text into chunks, a sentence might be cut in half right
#   at the boundary between two chunks. By overlapping (e.g., 100 characters),
#   the end of one chunk and the start of the next chunk share some text.
#   This means important sentences that fall on boundaries are still
#   captured in at least one chunk, improving search quality.
#
# Parameters:
#   list_of_entries (list): Output from the loader - list of dicts with
#                           "text", "location", "source"
# Returns:
#   list_of_chunks (list): List of dicts, each with "text", "location",
#                          "source", "chunk_id"
def split_pages_into_chunks(list_of_entries):
    print(f"\n--- Chunking: Processing {len(list_of_entries)} entries ---")

    # Create the text splitter with our settings from config
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # Use simple character count for length
        is_separator_regex=False,
    )

    list_of_chunks = []

    # Process each entry from the loader
    for entry in list_of_entries:
        entry_text = entry["text"]
        entry_location = entry["location"]
        entry_source = entry["source"]

        # Clean PDF text ONLY (entries whose location starts with "Page").
        # Do NOT clean TXT, DOCX, CSV, or Excel entries so their line and "|" structure stays intact.
        if entry_location.startswith("Page"):
            entry_text = clean_text(entry_text)

        # Split this entry's text into smaller pieces
        split_texts = text_splitter.split_text(entry_text)

        # Create a chunk dictionary for each split piece
        chunk_index = 0
        for chunk_text in split_texts:
            # Skip empty chunks (just in case)
            if chunk_text.strip() == "":
                continue

            # Check if this is a tabular entry (CSV or Excel)
            is_table_entry = entry_location.startswith("Rows") or entry_location.startswith("Sheet:")

            # Skip tiny chunks with fewer than 30 words (unless it is from CSV/Excel)
            word_count = len(chunk_text.split())
            if not is_table_entry and word_count < 30:
                print(f"  Skipping tiny chunk ({word_count} words) from {entry_source}, {entry_location}")
                continue

            # Create a unique ID for this chunk
            # Format: filename_location_cN (e.g., "paper1.pdf_Page 3_c0")
            chunk_id = f"{entry_source}_{entry_location}_c{chunk_index}"

            chunk = {
                "text": chunk_text,
                "location": entry_location,
                "source": entry_source,
                "chunk_id": chunk_id,
            }
            list_of_chunks.append(chunk)
            chunk_index = chunk_index + 1

    print(f"--- Chunking complete: Created {len(list_of_chunks)} chunks ---")
    return list_of_chunks


# ============================================================
# Test block: Run this file alone to test chunking
# Usage: python chunker.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CHUNKER MODULE TEST")
    print("=" * 60)

    # Create some sample entries of different types to test with
    sample_entries = [
        {
            "text": "Machine learning is a subset of artificial intelligence that "
                    "focuses on building systems that learn from data. These systems "
                    "can improve their performance over time without being explicitly "
                    "programmed. Deep learning is a further subset of machine learning "
                    "that uses neural networks with many layers. These deep neural "
                    "networks can learn complex patterns in large amounts of data. "
                    "Common applications include image recognition, natural language "
                    "processing, and speech recognition. The field has seen rapid "
                    "growth in recent years due to increased computing power and "
                    "the availability of large datasets. Transfer learning allows "
                    "models trained on one task to be fine-tuned for another task, "
                    "which has made deep learning more accessible and practical for "
                    "a wider range of applications. Reinforcement learning is another "
                    "important area where agents learn to make decisions by interacting "
                    "with an environment and receiving rewards or penalties.",
            "location": "Page 1",
            "source": "sample_paper.pdf",
        },
        {
            "text": "Name: Alice | Age: 30 | City: New York\n"
                    "Name: Bob | Age: 25 | City: San Francisco\n"
                    "Name: Charlie | Age: 35 | City: Chicago",
            "location": "Rows 1-3",
            "source": "people.csv",
        },
        {
            "text": "Introduction to the study of algorithms and their complexity. "
                    "This chapter covers the basic concepts needed to understand "
                    "how algorithms work and how to measure their efficiency.",
            "location": "Paragraphs 1-10",
            "source": "textbook.docx",
        },
        {
            "text": "Attention\nis\nall\nyou\nneed.\nThe\ndominant\nsequence\ntransduction\nmodels\n"
                    "are\nbased\non\ncomplex\nrecurrent\nor\nconvolutional\nneural\nnetworks\n"
                    "that\ninclude\nan\nencoder\nand\na\ndecoder.\nWe\npropose\na\nnew\nsimple\n"
                    "network\narchitecture\nbased\nsolely\non\nattention\nmechanisms.",
            "location": "Page 2",
            "source": "messy_paper.pdf",
        },
    ]

    print("\n--- Testing clean_text on messy one-word-per-line PDF text ---")
    messy_sample = sample_entries[3]["text"]
    print(f"Before clean_text:\n{messy_sample[:80]}...")
    print(f"After clean_text:\n{clean_text(messy_sample)}")

    print(f"\nInput: {len(sample_entries)} entries")
    for i in range(len(sample_entries)):
        print(f"  Entry {i+1}: source={sample_entries[i]['source']}, "
              f"location={sample_entries[i]['location']}, "
              f"text length={len(sample_entries[i]['text'])}")

    # Run the chunker
    chunks = split_pages_into_chunks(sample_entries)

    print(f"\nOutput: {len(chunks)} chunks")
    # Print details of each chunk
    for i in range(len(chunks)):
        chunk = chunks[i]
        text_preview = chunk["text"][:100]
        print(f"\n  Chunk {i+1}:")
        print(f"    ID: {chunk['chunk_id']}")
        print(f"    Source: {chunk['source']}")
        print(f"    Location: {chunk['location']}")
        print(f"    Text ({len(chunk['text'])} chars): {text_preview}...")

    print("\nChunker module test complete!")
