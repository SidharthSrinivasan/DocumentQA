import os
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Disable Streamlit watchdog warnings (optional, helps on some systems)
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

# Set Streamlit page config as the first Streamlit command
st.set_page_config(page_title="PDF Chatbot", layout="wide")

# ---------------- LLM Setup ----------------
@st.cache_resource
def load_llm():
    """Load and cache the local LLM pipeline for text generation."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # Offloads smartly if needed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.15
    )
    return HuggingFacePipeline(pipeline=pipe)

# Load the LLM pipeline (cached)
llm = load_llm()

# ---------------- Streamlit UI ----------------
st.title("📄 Chat With Your Document (Local LLM)")

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and indexing your document..."):
        import tempfile
        # Save uploaded file to a temporary location for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        # Load PDF pages
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        # Try using GPU for embeddings, fallback to CPU if OOM
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={"device": "cuda"}
            )
        except RuntimeError:
            st.warning("GPU out of memory for embedding model. Falling back to CPU.")
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={"device": "cpu"}
            )
        # Build FAISS vector index
        vectordb = FAISS.from_documents(docs, embeddings)
        # Set up retriever and memory for conversational QA
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False
        )
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        # User chat input
        question = st.chat_input("Ask a question about the document...")
        if question:
            with st.spinner("Thinking..."):
                result = qa_chain({"question": question})
                raw_answer = result["answer"]
            # Clean up answer formatting if needed
            if "Answer:" in raw_answer:
                clean_answer = raw_answer.split("Answer:", 1)[1].strip()
                clean_answer = "Answer: " + clean_answer
            else:
                clean_answer = raw_answer.strip()
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("ai", clean_answer))
        # Display chat history
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)
