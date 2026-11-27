"""
llm.py - LLM (Large Language Model) Module
================================
What this file does:
    Sends prompts to the Groq API and returns the generated answers.
    This module is the ONLY place where the LLM provider (Groq) is used.
    If you want to switch to a different LLM provider later, you only
    need to change THIS file - nothing else.

Input:  A prompt string
Output: An answer string from the LLM
"""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS


# Create the Groq client
# If the API key is None (not set in .env), the client will fail on the first call
groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"Groq client created successfully. Model: {GROQ_MODEL_NAME}")
    except Exception as error:
        print(f"Error creating Groq client: {error}")
        groq_client = None
else:
    print("WARNING: GROQ_API_KEY is not set. Please check your .env file.")
    print("The app will not be able to generate answers until the key is set.")


# ============================================================
# FUNCTION: generate_answer
# ============================================================
# What it does:
#   Sends a prompt to the Groq chat completion API and returns the
#   generated answer text. If anything goes wrong (no API key, no
#   internet, etc.), it returns a friendly error message instead of
#   crashing.
#
# Parameters:
#   prompt (str): The complete prompt to send to the LLM
# Returns:
#   answer (str): The LLM's response text, or an error message
def generate_answer(prompt):
    print(f"\n--- Generating answer with Groq ({GROQ_MODEL_NAME}) ---")
    print(f"  Prompt length: {len(prompt)} characters")

    # Check if the client was created successfully
    if groq_client is None:
        error_message = ("Error: The Groq API client is not set up. "
                         "Please make sure your GROQ_API_KEY is set in the .env file "
                         "and that it is valid.")
        print(f"  {error_message}")
        return error_message

    try:
        # Send the prompt to Groq's chat completion API
        # The prompt goes as a "user" message
        chat_response = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        # Extract the answer text from the response
        answer = chat_response.choices[0].message.content
        print(f"  Answer generated successfully ({len(answer)} characters).")
        return answer

    except Exception as error:
        # Catch any error (network issues, invalid key, rate limits, etc.)
        error_message = f"Error generating answer: {str(error)}"
        print(f"  {error_message}")

        # Return a user-friendly message
        if "api_key" in str(error).lower() or "auth" in str(error).lower():
            return ("Error: Your Groq API key appears to be invalid. "
                    "Please check the GROQ_API_KEY in your .env file.")
        elif "rate" in str(error).lower():
            return ("Error: You have hit the Groq API rate limit. "
                    "Please wait a moment and try again.")
        else:
            return (f"Error: Could not generate an answer. "
                    f"Details: {str(error)}")


# ============================================================
# Test block: Run this file alone to test the LLM connection
# Usage: python llm.py
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)

    if GROQ_API_KEY is None:
        print("\nGROQ_API_KEY is not set in .env file.")
        print("Please copy .env.example to .env and add your key.")
    else:
        print(f"\nAPI key loaded (starts with: {GROQ_API_KEY[:8]}...)")

        # Send a simple test prompt
        test_prompt = "What is 2 + 2? Answer in one sentence."
        print(f"\nTest prompt: {test_prompt}")

        answer = generate_answer(test_prompt)
        print(f"\nAnswer: {answer}")

    print("\nLLM module test complete!")
