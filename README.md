# 📟 AI-Powered Document Q&A Chatbot

This project demonstrates two approaches to building a privacy-friendly, local chatbot that can answer questions based on uploaded PDF documents. It walks through the evolution from a basic question-answering system using traditional pipelines to an advanced chatbot powered by a local large language model (LLM).

Built using Python, Streamlit, FAISS, and Hugging Face models.

---

## 📁 Project Versions

| Version | Description |
| ------- | ----------- |
|         |             |

|   |
| - |

| [v1\_faiss\_distilbert](./v1_faiss_distilbert/)   | Lightweight baseline using FAISS and DistilBERT for extractive QA                        |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [v2\_langchain\_mistral](./v2_langchain_mistral/) | Conversational chatbot using LangChain, Mistral 7B Instruct, and Hugging Face embeddings |

---

## 🔍 Features

✅ PDF document ingestion\
✅ Text chunking and embedding using `bge` Sentence Transformers\
✅ FAISS-based semantic search for relevant document parts\
✅ Extractive QA pipeline (v1)\
✅ Full conversational chatbot with memory and local LLM (v2)\
✅ GPU acceleration (v2)\
✅ No reliance on OpenAI or external APIs\
✅ Custom tokenizer and model configuration with `AutoModelForCausalLM`\
✅ Streamlit interface with chat history and file uploads\
✅ Mistral-based inference through HuggingFace pipeline

---

## 📅 Demo Screenshots

| Upload & Process PDF | Ask Questions |
| -------------------- | ------------- |
|                      |               |

---

## 🏐 Technologies Used

- **Python 3.11+**
- **Streamlit** – UI framework
- **FAISS** – Vector similarity search
- **LangChain** – Conversational pipeline (v2)
- **HuggingFace Transformers** – QA and LLM models
- **Sentence Transformers** – Document embeddings
- **Mistral-7B-Instruct** – Local LLM for RAG
- **torch**, **transformers**, **langchain**, **pdfplumber**

---

## 🚀 Getting Started

### Clone the Repo

```bash
git clone https://github.com/your-username/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### Set up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Install Dependencies (per version)

```bash
cd v2_langchain_mistral
pip install -r requirements.txt
```

> Ensure you have a GPU and the correct version of PyTorch + CUDA if using v2.

---

## 🧪 Version Details

### 🔹 v1: FAISS + DistilBERT QA

- Uses `BAAI/bge-small-en` for embeddings (384-dim)
- Uses `distilbert-base-uncased-distilled-squad` for QA
- Extractive only — selects best span from context
- Fast, lightweight, but lacks reasoning and context handling

### 🔹 v2: LangChain + Mistral 7B Instruct

- Uses `BAAI/bge-large-en` embeddings
- RAG + Conversational memory (LangChain)
- Powered by `mistralai/Mistral-7B-Instruct-v0.1`
- Handles follow-up questions, clarifications, and long-form responses
- Runs on GPU with `device_map="auto"`
- HuggingFace pipeline used for text generation

---

## 📦 Folder Structure

```
.
├── README.md
├── v1_faiss_distilbert/
│   ├── app.py
│   └── requirements.txt
├── v2_langchain_mistral/
│   ├── app.py
│   └── requirements.txt
├── screenshots/
│   ├── upload.png
│   └── chat.png
```

---

## 📌 TODO & Future Work

-

---

## 🧠 Author

**Sidharth Srinivasan**\
📢 [LinkedIn](https://www.linkedin.com/in/sidharth-srinivasan)
📃 [Portfolio](https://your-portfolio.com/)

---

## ⭐️ If you like this project

Please consider giving it a star ⭐ and sharing it with others!

