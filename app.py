import os
import re
import json
import hashlib
from typing import List, Tuple

import numpy as np
import streamlit as st
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

INDEX_DIR = ".rag_cache"
# ----------------------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    text = "\n".join(pages_text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = EMBED_MODEL, batch_size: int = 64) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_vecs.extend([item.embedding for item in resp.data])
    return np.array(all_vecs, dtype=np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
    index.add(vectors)
    return index


def cache_paths(doc_hash: str) -> Tuple[str, str]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, f"{doc_hash}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{doc_hash}.json")
    return index_path, meta_path


def save_cache(doc_hash: str, index: faiss.Index, chunks: List[str], chunk_size: int, overlap: int) -> None:
    index_path, meta_path = cache_paths(doc_hash)
    faiss.write_index(index, index_path)
    meta = {
        "doc_hash": doc_hash,
        "chunks": chunks,
        "embed_model": EMBED_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_cache(doc_hash: str) -> Tuple[faiss.Index, List[str]]:
    index_path, meta_path = cache_paths(doc_hash)
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"]


def retrieve(index: faiss.Index, chunks: List[str], query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float, str]]:
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, top_k)
    out = []
    for cid, score in zip(ids[0].tolist(), scores[0].tolist()):
        if cid == -1:
            continue
        out.append((cid, float(score), chunks[cid]))
    return out


def answer_question(client: OpenAI, question: str, retrieved: List[Tuple[int, float, str]]) -> str:
    context_blocks = []
    for cid, score, text in retrieved:
        context_blocks.append(f"[Chunk {cid} | score={score:.3f}]\n{text}")

    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context from the PDF. "
        "If the answer is not in the context, say exactly: 'I don't know based on the provided PDF.' "
        "Be concise and precise."
    )

    user = f"""CONTEXT (retrieved from PDF):
{context}

QUESTION:
{question}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# Streamlit needs io imported after top sometimes; keep it here to avoid confusion
import io


def init_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")
    if '"' in api_key or "'" in api_key:
        raise RuntimeError("OPENAI_API_KEY contains quotes. Remove quotes from .env.")
    return OpenAI(api_key=api_key)


def main():
    st.set_page_config(page_title="PDF RAG (FAISS) - Streamlit", layout="wide")
    st.title("PDF Question Answering (RAG + FAISS)")

    with st.sidebar:
        st.header("Settings")
        chunk_size = st.slider("Chunk size (characters)", 300, 2000, DEFAULT_CHUNK_SIZE, 50)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, DEFAULT_CHUNK_OVERLAP, 10)
        top_k = st.slider("Top-K chunks to retrieve", 1, 10, DEFAULT_TOP_K, 1)
        use_cache = st.checkbox("Use FAISS cache (.rag_cache)", value=True)
        show_chunks = st.checkbox("Show retrieved chunks", value=False)

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if "index" not in st.session_state:
        st.session_state.index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "doc_hash" not in st.session_state:
        st.session_state.doc_hash = None

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        doc_hash = sha256_bytes(pdf_bytes)

        st.info(f"Loaded: {uploaded.name}  |  Hash: {doc_hash[:12]}...")

        build_clicked = st.button("Build / Load Index", type="primary")

        if build_clicked:
            with st.spinner("Preparing index..."):
                client = init_client()

                if use_cache:
                    try:
                        index, chunks = load_cache(doc_hash)
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.doc_hash = doc_hash
                        st.success("Loaded cached FAISS index (no re-embedding needed).")
                    except FileNotFoundError:
                        text = read_pdf_text_from_bytes(pdf_bytes)
                        if not text:
                            st.error("No text could be extracted from this PDF (might be scanned).")
                            return
                        chunks = chunk_text(text, chunk_size, chunk_overlap)
                        if not chunks:
                            st.error("Chunking produced no chunks.")
                            return

                        vecs = embed_texts(client, chunks)
                        vecs = normalize_rows(vecs)
                        index = build_faiss_index(vecs)

                        save_cache(doc_hash, index, chunks, chunk_size, chunk_overlap)

                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.doc_hash = doc_hash
                        st.success("Built FAISS index and cached it.")
                else:
                    text = read_pdf_text_from_bytes(pdf_bytes)
                    if not text:
                        st.error("No text could be extracted from this PDF (might be scanned).")
                        return
                    chunks = chunk_text(text, chunk_size, chunk_overlap)
                    if not chunks:
                        st.error("Chunking produced no chunks.")
                        return
                    vecs = embed_texts(client, chunks)
                    vecs = normalize_rows(vecs)
                    index = build_faiss_index(vecs)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.doc_hash = doc_hash
                    st.success("Built FAISS index (not cached).")

    st.divider()

    if st.session_state.index is None or st.session_state.chunks is None:
        st.warning("Upload a PDF and click **Build / Load Index** to start asking questions.")
        return

    question = st.text_input("Ask a question about the PDF")
    ask = st.button("Ask", type="primary", disabled=(not question.strip()))

    if ask and question.strip():
        with st.spinner("Searching and generating answer..."):
            client = init_client()

            q_vec = embed_texts(client, [question])[0]
            q_vec = (q_vec / (np.linalg.norm(q_vec) + 1e-10)).astype(np.float32)

            retrieved = retrieve(st.session_state.index, st.session_state.chunks, q_vec, top_k)
            answer = answer_question(client, question, retrieved)

        st.subheader("Answer")
        st.write(answer)

        if show_chunks:
            st.subheader("Retrieved chunks")
            for cid, score, text in retrieved:
                with st.expander(f"Chunk {cid} (score={score:.3f})"):
                    st.write(text)


if __name__ == "__main__":
    main()

