# 🔹 Version 2: LangChain + Mistral 7B Instruct

This version implements a conversational document Q&A chatbot powered by a local large language model (LLM) using the Mistral-7B-Instruct model. It leverages LangChain to combine semantic search, memory, and dialogue handling for significantly better answers and multi-turn interactions.

---

## 🚀 Features

- Powered by `mistralai/Mistral-7B-Instruct-v0.1` — a strong open-source LLM
- Embedding using `BAAI/bge-large-en` for improved document understanding
- Conversational memory with LangChain for multi-turn dialogue
- FAISS for efficient vector search
- Hugging Face `pipeline()` for local text generation
- GPU-accelerated inference and embedding
- Streamlit UI with chat-like interface and history

---

## 📁 How It Works

1. Upload a PDF
2. Extract and split text using LangChain’s `RecursiveCharacterTextSplitter`
3. Generate embeddings with `bge-large-en` and build FAISS index
4. Retrieve relevant chunks via semantic search
5. Use LangChain’s `ConversationalRetrievalChain` to handle questions and chat history
6. Generate answers using a locally loaded Mistral model via HuggingFace `pipeline()`

---

## ▶️ Run

```bash
cd v2_langchain_mistral
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations

- Large memory and GPU requirements (at least 12–16 GB VRAM recommended)
- Slower inference time compared to v1 (tradeoff for better accuracy and reasoning)
- No fine-tuning or document summarization (yet)
- Longer documents may require batching or smarter chunking for performance

## 📦 Folder Contents

v2_langchain_mistral/
├── app.py              # Streamlit app with LangChain logic
├── requirements.txt    # Dependencies
