# RAG-Based-Academic-Assistant
🎓 A Retrieval-Augmented Generation (RAG) based academic assistant that lets you upload PDFs, ask questions, and get smart answers from your documents.
It built with **Streamlit**, **FastAPI**, and **LangGraph**, powered by **local document processing** and conversational memory.

## 🚀 Features

- 💬 Conversational AI powered by LangGraph
- 📄 Upload and index PDF academic documents
- 🔎 Contextual Q&A from uploaded files
- 📎 Attachment-style file upload next to chat input
- 🧠 Memory support per conversation thread

## 🛠️ Requirements

Make sure you have the following in your `requirements.txt`:


## 📂 Folder Structure

📁 RAG-Academic-Assistant/
├── 📂 books/                    # Folder to store academic PDFs
├── 📂 vectorstore/              # Saved FAISS vector database
├── 📂 .streamlit/               # Streamlit config (optional)
├── 📄 main.py                   # Entry point (FastAPI + Streamlit UI)
├── 📄 llm_handler.py            # LLM logic, retrieval, and graph setup
├── 📄 knowledge_base.py         # PDF loading, splitting, embedding
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env                      # Environment variables (e.g., GROQ_API_KEY)
└── 📄 README.md                 # Project documentation


## ⚙️ Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

2. **Run the app**  
   ```bash
   streamlit run main.py
