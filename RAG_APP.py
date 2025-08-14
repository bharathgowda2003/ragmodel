import streamlit as st
from dotenv import load_dotenv
import os
from langchain.groq import ChatGrog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import CombineDocumentsChain  # Check if this is the correct class
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalChain  # Ensure this is the correct import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader  # Corrected from PyPpFLoader
from langchain.huggingface import HuggingFaceEmbeddings  # Corrected import
import time

load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
st.image(" images.jpg")
st.title("Dynamic RAG with Groq, FAISS, and Llama3")
#Initialize session state for vector store and chat history
if "vector" not in st.session_state:
      st.session_state.vector = None
if "chat_history" not in st.session_state:
     st.session_state.chat_history = []
  with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
if st.button("Process Documents"):
  if uploaded_files:
    with st.spinner("Processing documents..."):
    docs = []
    for file in uploaded_files:
      with open(file.name, "wb") as f:
        f.write(file.getbuffer())
      loader = PyPDFLoder(file.name)
      docs.extend(loader.load())
