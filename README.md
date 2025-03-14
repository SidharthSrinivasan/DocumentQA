# AI-Powered Document Q&A

This project is a **Streamlit-based Q&A application** that allows users to upload PDF documents and ask questions about their content. It leverages **DistilGPT-2** for generating answers and **FAISS** for efficient similarity-based retrieval of relevant document sections.

---

## 🚀 Features
- 📄 **Upload PDFs** and extract text automatically
- 🔍 **Semantic Search** using **FAISS** for efficient document retrieval
- 🧠 **AI-Powered Answer Generation** with **DistilGPT-2**
- ⚡ **Fast and Lightweight** implementation optimized for local use

---

## 📦 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SidharthSrinivasan/DocumentQA.git
cd DocumentQA
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
---

## 🏃 Usage
### Run the Streamlit App
```bash
streamlit run main.py
```
### Upload a PDF and Ask Questions!
1. Click **"Upload a PDF"** and select a file.
2. Type a question in the input box.
3. The AI retrieves relevant context and generates an answer.

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – Web UI
- **Transformers** – Model loading (DistilGPT-2)
- **FAISS** – Fast document retrieval
- **Sentence-Transformers** – Text embeddings
- **pdfplumber** – PDF text extraction

---

## 🔧 Troubleshooting
### Common Issues & Fixes
- **Memory Issues / Crashes**
  - Reduce document size before uploading.
  - Run `gc.collect()` and `torch.cuda.empty_cache()` to free memory.
- **Model Loading Errors**
  - Ensure you have installed `torch`, `transformers`, and `accelerate`.
  - Use `device_map="cpu"` to avoid GPU memory issues.

---

## 👨‍💻 Contributors
- **Sidharth Srinivasan** – [GitHub](https://github.com/SidharthSrinivasan)

