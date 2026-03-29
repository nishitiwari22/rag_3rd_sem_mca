<img width="1887" height="917" alt="image" src="https://github.com/user-attachments/assets/134e0be6-d821-4c96-aa9d-e812c27c345c" />

<img width="1911" height="943" alt="image" src="https://github.com/user-attachments/assets/e319524e-d7b8-4f74-a2f6-683ac7323dd6" />


<img width="1911" height="927" alt="image" src="https://github.com/user-attachments/assets/a3bbf1da-7c27-44f6-a878-45c503e2682a" />


Project: RAG-based Document Question Answering System
Built an end-to-end Retrieval-Augmented Generation (RAG) system to enable intelligent question answering over PDF documents.
Implemented document ingestion pipeline using PyPDFLoader to extract and process unstructured text data.
Designed an efficient text chunking strategy using RecursiveCharacterTextSplitter (chunk size: 500, overlap: 50) to improve retrieval accuracy.
Generated semantic embeddings using Hugging Face Transformers (all-MiniLM-L6-v2) for contextual understanding of text.
Developed a vector search system using FAISS for fast similarity-based retrieval of relevant document chunks.
Integrated a lightweight LLM (distilgpt2) via Hugging Face pipeline to generate context-aware responses.
Engineered prompt templates with strict context grounding, reducing hallucinations and ensuring answer reliability.
Built an interactive UI using Streamlit allowing users to upload PDFs and query documents in real time.
Optimized retrieval by selecting top-k relevant chunks (k=3) to balance performance and response quality.
Handled edge cases by implementing fallback logic when answers are not present in the document.
Demonstrated understanding of core GenAI concepts: embeddings, vector databases, semantic search, and LLM pipelines.
