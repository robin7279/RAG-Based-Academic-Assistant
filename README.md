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

├── main.py # Main application file 
├── llm_handler.py # LangGraph configuration and LLM setup 
├── knowledge_base.py # Create embeddings and vectorstore 
├── books/  # PDF data 
├── vectorstore/ # Saved FAISS vectorstore 
├── .env  # For environment variable for Groq API
└──requirements.txt 


## ⚙️ Usage

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

2. **Run the app**  
   ```bash
   streamlit run main.py
