import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import streamlit as st
import pdfplumber
import psutil

huggingface_token = st.secrets["huggingface"]["token"]

# Set device (CPU)
device = "cpu"  # Ensure we're using CPU

# Load Mistral 7B model and tokenizer
llm_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(llm_model_name, use_auth_token=huggingface_token)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_auth_token=huggingface_token)

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
faiss_index_path = "faiss_index.bin"

# Load FAISS index if it exists
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)


# Function to get embeddings using SentenceTransformer (BGE model)
def get_embedding(text):
    embedding_model = SentenceTransformer('BAAI/bge-small-en', device=device)
    embedding = embedding_model.encode(text, convert_to_numpy=True)

    # Ensure the embedding is 2D (1, embedding_dimension)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    return embedding


# Function to store embeddings in FAISS index
def store_embedding(text_chunks):
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    embeddings = np.vstack(embeddings)

    if embeddings.shape[0] > 0 and embeddings.shape[1] > 0:
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)


# Function to generate an answer using Mistral 7B
def generate_answer(question, context):
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token if none is defined

    # Tokenize question and context together
    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Use no_grad to save memory
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=200)

    # Decode the output to get the text response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Cleanup memory
    del input_ids
    del attention_mask
    torch.cuda.empty_cache()  # Free cached memory (even though we're using CPU, this is a good practice)

    return answer


# Function to monitor memory usage
def check_memory():
    memory_info = psutil.virtual_memory()
    print(f"Memory usage: {memory_info.percent}%")


# Streamlit Web App
st.title("AI-Powered Document Q&A")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = " ".join(filter(None, (page.extract_text() for page in pdf.pages)))

    # Chunk the text into manageable pieces
    text_chunks = [text[i:i + 512] for i in range(0, len(text), 512)]

    # Store the embeddings
    with st.spinner("Processing document..."):
        store_embedding(text_chunks)

    # Handle question input
    question = st.text_input("Ask a question:")
    if question:
        check_memory()  # Check memory before the query

        query_embedding = get_embedding(question).reshape(1, -1)
        _, I = index.search(query_embedding, k=5)  # Retrieve top 5 chunks based on similarity

        retrieved_chunks = [text_chunks[i] for i in I[0] if 0 <= i < len(text_chunks)]
        retrieved_context = " ".join(retrieved_chunks)  # Merge the top retrieved chunks into context

        if retrieved_chunks:
            answer = generate_answer(question, retrieved_context)  # Generate answer using Mistral 7B
            st.write("**Answer:**", answer)
        else:
            st.write("No relevant answer found.")

        check_memory()  # Check memory after the query
