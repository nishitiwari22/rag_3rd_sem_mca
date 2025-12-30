import streamlit as st
import tempfile

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Updated embedding wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace LLM
from transformers import pipeline


st.title("📚 RAG Demo: Intelligent Document Question Answering")
st.write("Upload a PDF and ask questions. The system retrieves context and generates RAG-based answers.")


# File Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    st.success(f"PDF loaded successfully with {len(docs)} chunks!")

    # Embedding model (updated)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector database
    faiss_db = FAISS.from_documents(docs, embedding=embedder)

    # Load LLM
    generator = pipeline("text-generation", model="distilgpt2")

    # Ask question (always visible)
    user_query = st.text_input("Ask a question about the document")

    if uploaded_file and user_query:
        # Retrieve the most relevant chunks
        results = faiss_db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in results])

        # RAG prompt
        prompt = f"""
        Use ONLY the context to answer the question.
        If the answer is not present, reply: "The document does not contain this information."

        Context:
        {context}

        Question: {user_query}
        Answer:
        """

        response = generator(
            prompt,
            max_length=180,
            num_return_sequences=1
        )[0]["generated_text"]

        st.subheader("Answer:")
        st.write(response)




    # User question
    user_query = st.text_input("Ask a question about the document")

    if user_query:
        # Retrieve most relevant chunks
        results = faiss_db.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in results])

        # Prompt for RAG
        prompt = f"""
Answer the question using ONLY the context below. 
If answer is not in context, say 'The document does not contain this information.'

Context:
{context}

Question: {user_query}
Answer:
"""

        response = generator(prompt, max_length=180, num_return_sequences=1)[0]["generated_text"]

        st.subheader("Answer:")
        st.write(response)
