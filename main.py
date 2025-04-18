import uuid
import streamlit as st
import requests
import os
from fastapi import FastAPI
from pydantic import BaseModel
from threading import Thread
import uvicorn
from llm_handler import graph
from knowledge_base import load_documents, create_chunks, get_embedding_model
from langchain_community.vectorstores import FAISS

# FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    response = None
    for step in graph.stream(
        {"messages": [{"role": "user", "content": request.message}]},
        stream_mode="values",
        config=config,
    ):
        response = step["messages"][-1].content
    return {"response": response}

# Document Upload and Indexing
def handle_uploaded_files(uploaded_files):
    os.makedirs("books", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("books", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    
    # Reload and re-index documents
    docs = load_documents("books/")
    chunks = create_chunks(docs)
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/faiss_index")
    st.success("Documents uploaded and indexed successfully!")

# Streamlit UI
def run_streamlit():
    st.set_page_config(page_title="LangGraph Chatbot", layout="centered")
    st.title("📚 Academic Assistant")

    # Sidebar uploader
    st.sidebar.header("📁 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.sidebar.button("Index Documents"):
        if uploaded_files:
            handle_uploaded_files(uploaded_files)
        else:
            st.sidebar.warning("Please upload at least one PDF.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            res = requests.post("http://localhost:8000/chat", json={
                "message": user_input,
                "thread_id": st.session_state.thread_id
            })
            reply = res.json().get("response", "Something went wrong.")
            st.session_state.chat_history.append(("bot", reply))

    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)

# Run API and UI
def start():
    def run_api():
        uvicorn.run(app, host="127.0.0.1", port=8000)
    Thread(target=run_api, daemon=True).start()
    run_streamlit()

if __name__ == "__main__":
    start()
