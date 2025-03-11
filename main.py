import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import faiss
import numpy as np
import streamlit as st
import pdfplumber

# Set device (CPU/GPU)
device = "cpu"

embedding_model = SentenceTransformer('BAAI/bge-small-en', device=device)
# This Sentence Transformer uses dimensions of size 384
dimension = 384
index = faiss.IndexFlatL2(dimension)
faiss_index_path = "faiss_index.bin"  # File to store FAISS index persistently

llm_model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(llm_model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if device == "mps" else -1)

# Load FAISS index if it exists
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)


def get_embedding(text):
    embedding = embedding_model.encode(text, convert_to_numpy=True)

    # Ensure the embedding is 2D (1, embedding_dimension)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    return embedding


def store_embedding(text_chunks):
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    embeddings = np.vstack(embeddings)  # Convert to 2D array

    if embeddings.shape[0] > 0 and embeddings.shape[1] > 0:
        index.add(embeddings)

        # Save FAISS index
        faiss.write_index(index, faiss_index_path)


def generate_answer(question, context):
    results = qa_pipeline(question=question, context=context, top_k=3)  # Get top 3 answers

    # Pick the highest-confidence answer above a threshold
    best_answer = max(results, key=lambda x: x['score']) if results else None

    if best_answer and best_answer['score'] > 0.5:  # Adjust the threshold as needed
        return best_answer['answer']
    else:
        return "I couldn't find a reliable answer in the document."


# Using Streamlit for webapp
st.title("AI-Powered Document Q&A")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = " ".join(filter(None, (page.extract_text() for page in pdf.pages)))

    text_chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    with st.spinner("Processing document..."):
        store_embedding(text_chunks)

    question = st.text_input("Ask a question:")
    if question:
        query_embedding = get_embedding(question).reshape(1, -1)
        _, I = index.search(query_embedding, k=5)  # Retrieve top 5 chunks

        retrieved_chunks = [text_chunks[i] for i in I[0] if 0 <= i < len(text_chunks)]
        retrieved_context = " ".join(retrieved_chunks)  # Merge retrieved text

        if retrieved_chunks:
            answer = generate_answer(question, retrieved_chunks)
            st.write("**Answer:**", answer)
        else:
            st.write("No relevant answer found.")
