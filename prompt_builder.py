"""
prompt_builder.py - Prompt Building Module (Augmentation Step)
================================
What this file does:
    Takes the retrieved chunks and the user's question and combines them
    into a single prompt that tells the LLM how to answer. This is the
    "Augmentation" step in RAG (Retrieval-Augmented Generation).

Input:  A question (string) and a list of retrieved chunk dictionaries
Output: A formatted prompt string ready to send to the LLM
"""


# ============================================================
# FUNCTION: format_context_from_chunks
# ============================================================
# What it does:
#   Converts each retrieved chunk into a formatted text block with
#   source and location labels, then joins them all together.
#
# Parameters:
#   list_of_retrieved_chunks (list): List of dicts with "text", "source", "location"
# Returns:
#   A formatted context string with all chunks labeled and separated
def format_context_from_chunks(list_of_retrieved_chunks):
    formatted_blocks = []

    for chunk in list_of_retrieved_chunks:
        # Get the relevance percent (default to 0 if not present)
        relevance_percent = chunk.get("relevance_percent", 0)
        # Create a labeled block for each chunk
        block = f"[Source: {chunk['source']}, Location: {chunk['location']}, Relevance: {relevance_percent}%]\n{chunk['text']}"
        formatted_blocks.append(block)

    # Join all blocks with a separator line
    separator = "\n" + "-" * 40 + "\n"
    context = separator.join(formatted_blocks)

    return context


# ============================================================
# FUNCTION: build_final_prompt
# ============================================================
# What it does:
#   Combines the system instructions, the formatted context, and the
#   user's question into one complete prompt for the LLM.
#   The instructions tell the LLM to:
#   - Only use the given context (no outside knowledge)
#   - Cite sources with file name and location
#   - Say "I could not find this information" if the answer is not there
#
# Parameters:
#   question (str): The user's question
#   list_of_retrieved_chunks (list): List of retrieved chunk dicts
# Returns:
#   A complete prompt string ready to send to the LLM
def build_final_prompt(question, list_of_retrieved_chunks):
    # Format the retrieved chunks into a labeled context string
    context = format_context_from_chunks(list_of_retrieved_chunks)

    # Build the complete prompt using the template
    prompt = f"""You are a helpful research assistant.
Answer the question using ONLY the context given below.
If the answer is not present in the context, reply exactly: "I could not find this information in the uploaded files."
Cite every fact in EXACTLY this format: (file_name, location), for example (paper1.pdf, Page 3) or (data.csv, Rows 1-20). Do not write the word "Source" inside the brackets.
If the context contains information from MORE THAN ONE file, answer separately for each file and clearly name the file, instead of mixing them.
Do not use any knowledge from outside the context.
Keep the answer clear and well structured.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


# ============================================================
# Test block: Run this file alone to test prompt building
# Usage: python prompt_builder.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PROMPT BUILDER MODULE TEST")
    print("=" * 60)

    # Create some sample retrieved chunks
    sample_chunks = [
        {
            "text": "The study used a convolutional neural network (CNN) architecture "
                    "with 5 convolutional layers and 2 fully connected layers. The model "
                    "was trained on the ImageNet dataset.",
            "source": "paper1.pdf",
            "location": "Page 3",
            "distance": 0.15,
            "relevance_percent": 92,
        },
        {
            "text": "The dataset contained 50,000 training images and 10,000 test images. "
                    "Data augmentation techniques including rotation and flipping were applied.",
            "source": "paper1.pdf",
            "location": "Page 5",
            "distance": 0.22,
            "relevance_percent": 85,
        },
        {
            "text": "Revenue: 1000000 | Profit: 250000 | Year: 2023\n"
                    "Revenue: 1200000 | Profit: 300000 | Year: 2024",
            "source": "sales.csv",
            "location": "Rows 1-20",
            "distance": 0.45,
            "relevance_percent": 31,
        },
    ]

    # Test formatting context
    print("\n--- Test 1: Format context from chunks ---")
    context = format_context_from_chunks(sample_chunks)
    print(context)

    # Test building the full prompt
    print("\n--- Test 2: Build full prompt ---")
    test_question = "Which algorithm was used and what dataset was it trained on?"
    prompt = build_final_prompt(test_question, sample_chunks)
    print(prompt)

    print("\n" + "=" * 60)
    print(f"Total prompt length: {len(prompt)} characters")
    print("\nPrompt builder module test complete!")
