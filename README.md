# 📚 RAG-Powered AI Research Paper Assistant

A simple **Retrieval-Augmented Generation (RAG)** application that lets you upload research papers and other documents, then ask questions and get answers **with source references**.

Upload PDFs, text files, Word documents, Excel spreadsheets, or CSV files — the app extracts the content, stores it in a vector database, and uses an LLM to answer your questions based on the actual document text.

---

## ✅ Supported File Types

| Type | Extension | Notes |
|------|-----------|-------|
| PDF | `.pdf` | Extracted with PyMuPDF (fallback: PyPDF) |
| Plain Text | `.txt` | UTF-8 or Latin-1 encoding |
| Word | `.docx` | Paragraphs + tables (not `.doc`) |
| Excel | `.xlsx` | All sheets, all rows (not `.xls`) |
| CSV | `.csv` | All rows, all columns |

---

## 📁 Project Structure

```
research_paper_assistant/
│
├── data/
│   └── uploads/               # Uploaded files are saved here
├── chroma_db/                 # Vector DB files (auto-created)
│
├── config.py                  # All settings in one place
├── loader.py                  # ALL file loaders + router
├── chunker.py                 # Split text into chunks
├── embedder.py                # Sentence-transformer embeddings
├── vector_store.py            # ChromaDB add/search/list/delete
├── reranker.py                # Cross-encoder reranking of chunks
├── retriever.py               # Question -> relevant chunks
├── prompt_builder.py          # Context + question -> prompt
├── llm.py                     # Call Groq LLM for the answer
├── pipeline.py                # Connects everything together
├── app.py                     # Streamlit web UI
├── requirements.txt           # Python dependencies
├── .env                       # API key 
├── .gitignore                 # Files to ignore in git
└── README.md                  # This file
```

---

## 🚀 Setup Instructions

### Step 1: Clone or download the project

Place the project folder anywhere on your computer.

### Step 2: Create a virtual environment

Open a terminal in the project folder and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up your Groq API key

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys) and create a free API key.
2. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
   On Windows:
   ```bash
   copy .env.example .env
   ```
3. Open `.env` and replace `your_key_here` with your actual API key:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

### Step 5: Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧪 Testing Each Module Individually

Every backend module can be run and tested on its own. This is useful for learning how each part works:

```bash
python config.py          # Check that all settings load correctly
python loader.py          # Test file loading (place files in data/uploads/ first)
python chunker.py         # Test text chunking with sample data
python embedder.py        # Test embedding (downloads model on first run)
python vector_store.py    # Test ChromaDB add/search/delete
python reranker.py        # Test cross-encoder reranking (downloads ~90 MB model on first run)
python retriever.py       # Test retrieval (needs data in ChromaDB)
python prompt_builder.py  # Test prompt building with sample data
python llm.py             # Test Groq API connection (needs .env key)
python pipeline.py        # Test the full flow (needs files in data/uploads/)
```

---

## 🔄 How Data Flows Through the Modules

Here is exactly how data moves through the system, step by step:

### Ingestion Flow (when you upload files):

```
User uploads file(s)
        ↓
    loader.py         → Reads the file, returns a list of text entries
        ↓                 with location labels (Page 1, Rows 1-20, etc.)
    chunker.py        → Splits each entry into smaller overlapping chunks
        ↓                 (each chunk keeps its source file name + location)
    embedder.py       → Converts each chunk's text into a 384-dimension
        ↓                 numerical vector (embedding)
    vector_store.py   → Stores text + embedding + metadata in ChromaDB
                          on disk (chroma_db/ folder)
```

### Question Answering Flow (when you ask a question):

```
User types a question
        ↓
    pipeline.py       → Expands vague questions with helpful search keywords
        ↓                 (original question is kept for the LLM and UI)
    embedder.py       → Converts the search question into a 384-dimension vector
        ↓                 (same model as the chunks)
    vector_store.py   → Step 1: Searches ChromaDB for the top 15 candidate chunks
        ↓                 (cosine similarity)
    reranker.py       → Step 2: Scores each (question, chunk) pair with a
        ↓                 CrossEncoder (~90 MB, auto-downloads on first run)
        ↓                 and keeps the top 5 most relevant chunks
    prompt_builder.py → Combines the top 5 chunks + original question into
        ↓                 one prompt with instructions for the LLM
    llm.py            → Sends the prompt to Groq API and gets the answer
        ↓
    app.py            → Displays the answer with source references
                          and a cleaner, numbered "Sources" section
```

---

## 🔍 How retrieval works now

1. **Step 1 — Broad Embedding Search (15 chunks):** When you ask a question, the app expands generic questions (like *"What problem does this solve?"*) with related research terms, turns it into a vector, and fetches the top **15** candidate chunks from ChromaDB.
2. **Step 2 — Cross-Encoder Reranking (down to 5 chunks):** A CrossEncoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB, downloaded automatically on first run) reads the question and each of the 15 chunks together, gives each chunk a **0–100% relevance score**, and keeps only the best **5** chunks.

> **⚠️ Important if you are updating from an earlier version:**
> Because PDF text cleaning and tiny-chunk filtering were updated in `chunker.py`, please **delete your old indexed files in the sidebar (🗑️) and re-upload/re-process them** so the new cleaner chunks are stored in ChromaDB.

---

## 📝 Notes and Known Limitations

- **PDF text extraction can be messy.** Research papers often have two-column layouts, tables, and mathematical formulas. The extracted text may not look perfectly formatted. This is a limitation of text extraction from PDFs, not a bug.

- **DOCX and TXT files do not have real page numbers.** Word documents' page layout depends on fonts, margins, and printer settings, so paragraph ranges are used as references. TXT files use line ranges.

- **Excel and CSV rows are converted to text.** Very large spreadsheets (thousands of rows) will create many chunks and take longer to process and search.

- **Only `.xlsx` Excel files are supported** (not the older `.xls` format). Only `.docx` Word files are supported (not the older `.doc` format).

- **Chunk size affects answer quality.** The default chunk size (800 characters with 100 character overlap) is a reasonable starting point. Smaller chunks may give more precise answers but lose context; larger chunks keep more context but may include irrelevant text.

- **"Summarize the whole document" gives a partial summary.** Because only the top 5 most relevant chunks are retrieved, asking for a summary will give a summary of the most relevant parts, not a complete summary of the entire document.

- **The Groq free tier has rate limits.** If you send too many questions quickly, you may hit the rate limit. Wait a moment and try again.

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---------|------|
| Language | Python 3.10+ |
| PDF Loading | PyMuPDF + PyPDF (fallback) |
| Word Loading | python-docx |
| Excel/CSV Loading | pandas + openpyxl |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (persistent, on disk) |
| LLM | Groq API (Llama 3.1 8B Instant) |
| Web UI | Streamlit |
| Secrets | python-dotenv (.env file) |
