from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load documents from a directory 
FILE_PATH = "books/"
def load_documents(data):
    loader = DirectoryLoader(
        path=FILE_PATH,
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()

    return documents

documents = load_documents(data=FILE_PATH)
# print(f"Loaded {len(documents)} documents from {FILE_PATH}")

# Split documents into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
   
text_chunks=create_chunks(extracted_data=documents)
# print(f"Created {len(text_chunks)} chunks from {len(documents)} documents")

# Create embeddings for the text chunks
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embedding_model

embeddings = get_embedding_model()

# Store embeddings in FAISS
DB_PATH = "vectorstore/faiss_index"
vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings,
)

vectorstore.save_local(DB_PATH)
print(f"Saved FAISS index to {DB_PATH}")
