# 🔹 Version 1: FAISS + DistilBERT QA

This version implements a simple extractive question-answering system using FAISS for semantic search and DistilBERT for span-based answers. It is lightweight, fast, and suitable for straightforward question-answering over small to medium documents.

---

## 🚀 Features

- Local-only: No API keys or cloud dependencies
- Uses `BAAI/bge-small-en` for document and query embeddings
- FAISS vector index for fast similarity search
- `distilbert-base-uncased-distilled-squad` model for span-based extractive QA
- Simple Streamlit interface

---

## 📁 How It Works

1. Upload a PDF
2. Extract text and split into 512-character chunks
3. Generate embeddings using `bge-small-en`
4. Store and search embeddings using FAISS
5. Pass top matching chunks to DistilBERT QA pipeline
6. Output most likely answer span

---

## ▶️ Run

```bash
cd v1_faiss_distilbert
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- Limited reasoning capability — answers are extracted spans, not generated
- Does not maintain conversational context across questions
- Only retrieves a fixed number of chunks (no dynamic context optimization)
- May return incomplete or irrelevant spans if context is poorly retrieved
- Embedding model (`bge-small-en`) is fast but not as accurate as larger models

---

## 📦 Folder Contents
v1_faiss_distilbert/
├── app.py # Streamlit UI and QA logic
├── faiss_index.bin # Saved FAISS index (generated after upload)
├── requirements.txt # Dependencies


---

## 🧠 Model Details

- **Embeddings**: [`BAAI/bge-small-en`](https://huggingface.co/BAAI/bge-small-en) – 384-dimension vector representations
- **QA Model**: [`distilbert-base-uncased-distilled-squad`](https://huggingface.co/distilbert-base-uncased-distilled-squad) – Trained on SQuAD for extractive question answering
- **Retriever**: [FAISS](https://github.com/facebookresearch/faiss) – fast approximate nearest neighbor search

---