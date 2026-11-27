"""
app.py - Streamlit Web UI
================================
What this file does:
    Provides a clean, modern web interface for the AI Research Paper Assistant.
    Users can upload files, process them, and ask questions through this UI.
    All backend logic is handled by pipeline.py and vector_store.py.

Input:  User interactions (file uploads, button clicks, chat messages)
Output: A web page with file management sidebar, chat interface, and source display
"""

import os
import streamlit as st
from pipeline import ingest_file, ask_question
from vector_store import list_all_documents, delete_document, get_total_chunk_count
from config import UPLOAD_FOLDER_PATH, MIN_RELEVANCE_PERCENT


# ============================================================
# SECTION 1: PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# SECTION 2: CUSTOM CSS
# ============================================================
# A small block of custom CSS for nicer spacing, rounded source cards,
# a colored location badge, and a softer header style.

custom_css = """
<style>
/* Make the main header softer and centered */
.main-header {
    text-align: center;
    padding: 0.5rem 0 0.2rem 0;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38BDF8;
}
.main-header p {
    font-size: 1.05rem;
    opacity: 0.85;
    margin-top: -0.5rem;
}

/* Rounded card style for source chunks in the expander */
.source-card {
    background-color: #f8f9fb;
    border: 1px solid #e0e4ea;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

/* Colored badge for file name and location */
.location-badge {
    display: inline-block;
    background-color: #e8f0fe;
    color: #1a56db;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.4rem;
}

/* Similarity score styling */
.similarity-score {
    display: inline-block;
    background-color: #edf7ed;
    color: #2e7d32;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.4rem;
}

/* Rank number badge (#1, #2, ...) */
.rank-number {
    display: inline-block;
    background-color: #1E3A5F;
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-right: 0.4rem;
}

/* Green badge when chunk is cited in the answer */
.used-badge {
    display: inline-block;
    background-color: #dcfce7;
    color: #166534;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.4rem;
}

/* Scrollable box for full chunk text */
.source-text-box {
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #333;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
    padding-right: 0.4rem;
}

/* Footer styling */
.footer {
    text-align: center;
    color: #999;
    font-size: 0.82rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #eee;
    margin-top: 2rem;
}

/* Quick question buttons - add spacing */
.stButton > button {
    border-radius: 8px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================
# SECTION 3: SESSION STATE INITIALIZATION
# ============================================================
# Initialize session state variables if they do not exist yet.
# Session state keeps data between Streamlit reruns.

if "chat_history" not in st.session_state:
    # List of dicts: {"role": "user"/"assistant", "content": "...", "sources": [...]}
    st.session_state["chat_history"] = []

if "file_filter" not in st.session_state:
    # Which file to filter questions by (None = all files)
    st.session_state["file_filter"] = None


# ============================================================
# SECTION 4: HEADER
# ============================================================

st.markdown(
    '<div class="main-header">'
    '<h1>📚 AI Research Paper Assistant</h1>'
    '<p>Upload PDF, TXT, DOCX, Excel or CSV files and ask questions. '
    'Get answers with references.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SECTION 5: SIDEBAR - FILE MANAGEMENT
# ============================================================

with st.sidebar:
    # --- 5a: File Upload Section ---
    st.header("📄 Your Files")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, DOCX, XLSX, CSV)",
        type=["pdf", "txt", "docx", "xlsx", "csv"],
        accept_multiple_files=True,
    )
    st.caption("Supported: PDF, TXT, DOCX, XLSX, CSV")

    # --- 5b: Process Files Button ---
    process_button = st.button("⚙️ Process Files", type="primary", use_container_width=True)

    if process_button:
        if uploaded_files is None or len(uploaded_files) == 0:
            st.warning("Please upload at least one file first.")
        else:
            # Create the upload folder if it does not exist
            if not os.path.isdir(UPLOAD_FOLDER_PATH):
                os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)

            total_files = len(uploaded_files)
            progress_bar = st.progress(0, text="Starting file processing...")

            for file_index in range(total_files):
                uploaded_file = uploaded_files[file_index]
                file_name = uploaded_file.name

                # Update progress bar
                progress_value = (file_index) / total_files
                progress_bar.progress(progress_value, text=f"Processing {file_name}...")

                with st.spinner(f"Processing {file_name}..."):
                    # Save the uploaded file to the uploads folder
                    save_path = os.path.join(UPLOAD_FOLDER_PATH, file_name)
                    try:
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    except Exception as error:
                        st.error(f"❌ Could not save {file_name}: {str(error)}")
                        continue

                    # Ingest the file through the pipeline
                    try:
                        result = ingest_file(save_path)
                    except Exception as error:
                        st.error(f"❌ Error processing {file_name}: {str(error)}")
                        continue

                    # Show result for this file
                    if result["status"] == "added":
                        st.success(
                            f"✅ **{file_name}** ({result['file_type'].upper()}) — "
                            f"{result['chunks']} chunks indexed"
                        )
                    elif result["status"] == "skipped":
                        st.info(
                            f"ℹ️ **{file_name}** ({result['file_type'].upper()}) — "
                            f"Already indexed"
                        )
                    else:
                        st.error(f"❌ **{file_name}** — {result['message']}")

            # Complete the progress bar
            progress_bar.progress(1.0, text="All files processed!")

    # --- 5c: Indexed Files Section ---
    st.divider()
    st.header("📚 Indexed Files")

    indexed_files = list_all_documents()
    total_chunks = get_total_chunk_count()

    if len(indexed_files) == 0:
        st.info("No files yet. Upload files above to begin.")
    else:
        # Show total chunks as a small metric
        st.metric("Total Chunks", total_chunks)

        # Display each indexed file with an icon and delete button
        for indexed_file_name in indexed_files:
            # Choose an icon based on file type
            if "." in indexed_file_name:
                file_ext = indexed_file_name.rsplit(".", 1)[1].lower()
            else:
                file_ext = ""

            if file_ext == "pdf":
                file_icon = "📕"
            elif file_ext == "txt":
                file_icon = "📄"
            elif file_ext == "docx":
                file_icon = "📘"
            elif file_ext == "xlsx":
                file_icon = "📗"
            elif file_ext == "csv":
                file_icon = "📊"
            else:
                file_icon = "📁"

            # Create a row with file name and delete button
            col_name, col_delete = st.columns([4, 1])

            with col_name:
                st.write(f"{file_icon} {indexed_file_name}")

            with col_delete:
                # Each delete button needs a unique key
                delete_key = f"delete_{indexed_file_name}"
                if st.button("🗑️", key=delete_key, help=f"Delete {indexed_file_name}"):
                    try:
                        delete_document(indexed_file_name)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Error deleting: {str(error)}")

    # --- 5d: Search Filter ---
    st.divider()
    st.header("🔎 Search in")

    # Build the list of filter options
    filter_options = ["All files"]
    for file_name in indexed_files:
        filter_options.append(file_name)

    selected_filter = st.selectbox(
        "Choose which files to search",
        options=filter_options,
        label_visibility="collapsed",
    )

    # Set the file filter in session state
    if selected_filter == "All files":
        st.session_state["file_filter"] = None
    else:
        st.session_state["file_filter"] = selected_filter

    # --- 5e: Clear Chat Button ---
    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


# ============================================================
# SECTION 6: MAIN CHAT AREA
# ============================================================

# --- 6a: Welcome message if no files are indexed ---
if len(indexed_files) == 0:
    st.info(
        "👋 **Welcome! Get started in 3 simple steps:**\n\n"
        "1. **Upload files** in the sidebar (PDF, TXT, DOCX, XLSX, CSV)\n"
        "2. **Click Process Files** to index them\n"
        "3. **Ask your questions** in the chat below"
    )


# ============================================================
# HELPER FUNCTION: display_sources
# ============================================================
# What it does: Shows the filtered and sorted source chunks in a styled expander
# Parameters: sources (list of chunk dicts), answer_text (str)
# Returns: Nothing (renders UI elements)
def display_sources(sources, answer_text=""):
    if sources is None or len(sources) == 0:
        return

    # Sort chunks from highest to lowest relevance_percent
    sorted_sources = list(sources)
    total = len(sorted_sources)
    for i in range(total):
        for j in range(0, total - i - 1):
            score_a = sorted_sources[j].get("relevance_percent", 0)
            score_b = sorted_sources[j + 1].get("relevance_percent", 0)
            if score_a < score_b:
                temp = sorted_sources[j]
                sorted_sources[j] = sorted_sources[j + 1]
                sorted_sources[j + 1] = temp

    # Keep only chunks whose relevance_percent >= MIN_RELEVANCE_PERCENT
    filtered_sources = []
    for chunk in sorted_sources:
        if chunk.get("relevance_percent", 0) >= MIN_RELEVANCE_PERCENT:
            filtered_sources.append(chunk)

    # If none pass the threshold, still show the single best chunk
    if len(filtered_sources) == 0:
        filtered_sources = [sorted_sources[0]]

    with st.expander(f"📖 View Sources ({len(filtered_sources)} relevant chunks)"):
        for i in range(len(filtered_sources)):
            source = filtered_sources[i]
            rank_number = i + 1
            relevance = source.get("relevance_percent", 0)

            # Check if BOTH the file name and location appear in the LLM answer
            used_html = ""
            if answer_text and (source["source"] in answer_text) and (source["location"] in answer_text):
                used_html = '<span class="used-badge">✅ Used in answer</span>'

            # Build the styled source card using HTML (full text inside a scrollable box)
            source_html = f"""
            <div class="source-card">
                <span class="rank-number">#{rank_number}</span>
                <span class="location-badge">{source['source']} • {source['location']}</span>
                <span class="similarity-score">🎯 {relevance}% relevance</span>
                {used_html}
                <div class="source-text-box">{source['text']}</div>
            </div>
            """
            st.markdown(source_html, unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTION: process_question
# ============================================================
# What it does: Takes a question, calls the pipeline, and updates chat history
# Parameters: question (str)
# Returns: Nothing (updates session state and renders UI)
def process_question(question):
    # Check if any files are indexed
    if len(indexed_files) == 0:
        st.warning("⚠️ Please upload and process some files first before asking questions.")
        return

    # Add the user's question to chat history
    user_message = {
        "role": "user",
        "content": question,
        "sources": None,
    }
    st.session_state["chat_history"].append(user_message)

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(question)

    # Get the file filter from session state
    file_filter = st.session_state["file_filter"]

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching files and generating answer..."):
            try:
                result = ask_question(question, file_name_filter=file_filter)
                answer = result["answer"]
                sources = result["sources"]
            except Exception as error:
                answer = f"An error occurred: {str(error)}"
                sources = []

        # Display the answer
        st.markdown(answer)

        # Display sources
        display_sources(sources, answer)

    # Add the assistant's answer to chat history (with sources)
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
    }
    st.session_state["chat_history"].append(assistant_message)


# --- 6c: Display chat history ---
# Re-display all previous messages from session state
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this is an assistant message with sources, show them again
        if message["role"] == "assistant" and message.get("sources"):
            display_sources(message["sources"], message["content"])


# --- 6e: Chat input ---
chat_input = st.chat_input("Ask a question about your files...")

if chat_input:
    if len(indexed_files) == 0:
        st.warning("⚠️ Please upload and process some files first before asking questions.")
    else:
        process_question(chat_input)


# ============================================================
# SECTION 7: FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Built with Streamlit, ChromaDB, Sentence-Transformers and Groq'
    '</div>',
    unsafe_allow_html=True,
)
