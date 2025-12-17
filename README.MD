# PDF Question Answering with Streamlit GUI (RAG + FAISS)

This project is a **PDF Question Answering application with a graphical user interface (GUI)** built using **Streamlit**. It implements a **Retrieval-Augmented Generation (RAG)** pipeline:

* Extract text from an uploaded PDF
* Split the text into **chunks**
* Create embeddings and store them in a **FAISS vector index**
* Perform **similarity search** to retrieve the most relevant chunks for a question
* Use an OpenAI chat model to answer **strictly from retrieved PDF context**

It also supports **local caching** of the FAISS index so the same PDF does not need to be re-embedded every time.

---

## Project Structure

```
AI-Task-05/
│
├── app.py               # Streamlit GUI application (RAG + FAISS)
├── example.pdf          # Sample PDF (optional)
├── .env                 # Stores OPENAI_API_KEY (NOT committed)
├── .env.example         # Environment variable template (safe to commit)
├── .gitignore           # Git ignore rules
├── .rag_cache/          # Cached FAISS index + metadata (auto-generated)
└── README.md            # Documentation (this file)
```

---

## 1. Setup Guide

### Prerequisites

* Windows / macOS / Linux
* Python 3.10 or higher
* OpenAI API key
* pip (Python package manager)

---

### Installation

#### 1) Create (optional) virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS / Linux**

```bash
source venv/bin/activate
```

#### 2) Install dependencies

```bash
pip install streamlit openai python-dotenv pypdf numpy faiss-cpu
```

---

## 2. API Key Configuration (Important)

Create a file named `.env` in the project root:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Notes:

* Do **not** use quotes around the key
* Never commit `.env` to GitHub
* Rotate the key if it was ever exposed

Commit a safe template instead:

**.env.example**

```env
OPENAI_API_KEY=
```

---

## 3. Running the Streamlit App

### Recommended command (works even when `streamlit` is not on PATH)

```bash
python -m streamlit run app.py
```

Alternative:

```bash
py -m streamlit run app.py
```

Streamlit will print a local URL such as `http://localhost:8501`. Open it in your browser.

---

## 4. How It Works (RAG + FAISS)

1. User uploads a PDF using the GUI.
2. The app extracts text using `pypdf`.
3. Text is split into overlapping chunks.
4. Each chunk is embedded using OpenAI embeddings.
5. Embeddings are stored in a FAISS index (cosine similarity via inner product on normalized vectors).
6. When a question is asked, the question is embedded and searched against FAISS.
7. The top-K chunks are retrieved and passed to the LLM.
8. The model answers using **only the retrieved chunks**.

If the answer is not present in the retrieved context, the model responds:

> I don't know based on the provided PDF.

---

## 5. Caching

When caching is enabled, the app saves:

* FAISS index: `.rag_cache/<hash>.faiss`
* Chunk metadata: `.rag_cache/<hash>.json`

If the same PDF is uploaded again (same content hash), the index is loaded from cache and embedding is skipped.

---

## 6. Common Issues

### A) `'streamlit' is not recognized`

Use:

```bash
python -m streamlit run app.py
```

This avoids Windows PATH issues.

### B) `AuthenticationError: 401` (invalid API key)

* Check `.env` contains the correct key
* Ensure there are **no quotes**
* Restart the terminal
* Rotate the key if it was committed/exposed

### C) No text extracted

Some PDFs are scanned images (no selectable text). In that case, OCR is required (not included in this version).

---

## 7. `.gitignore` (Recommended)

Ensure these are ignored:

```gitignore
.env
.env.*
!.env.example
.rag_cache/
__pycache__/
*.pyc
venv/
.venv/
```

---

## License

Educational / experimental use.
