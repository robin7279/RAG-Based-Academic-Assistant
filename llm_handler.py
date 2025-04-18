import getpass
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Initialize the Groq model
load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

MODEL_ID = "llama-3.3-70b-versatile"

def load_llm(MODEL_ID):
    llm = ChatGroq(
        model=MODEL_ID,
        temperature=0.1,
    )

    return llm

llm = load_llm(MODEL_ID)
# response = llm.invoke("What is the capital of Bangladesh?")
# print(response.content)

# Load Database
DB_PATH = "vectorstore/faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    DB_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
# print(f"Loaded FAISS index from {DB_PATH}")


# create a state graph for the messages
graph_builder = StateGraph(MessagesState)


# Create retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retriever = db.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)
    serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
    return serialized, retrieved_docs
   
# Create nodes for the graph
def query_or_respond(state: MessagesState):
   """Generate tool call for retrieval or respond."""
   llm_with_tools = llm.bind_tools([retrieve])
   response = llm_with_tools.invoke(state["messages"])

   return {"messages": state["messages"] + [response]}


tools = ToolNode([retrieve])

def generate_response(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": state["messages"] + [response]}


def create_graph():
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate_response)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
    ) 
    
    graph_builder.add_edge("tools", "generate_response")
    graph_builder.add_edge("generate_response", END)

    # Add memory saver to the graph
    memory = MemorySaver()

    graph = graph_builder.compile(checkpointer=memory)

    return graph


graph = create_graph()


input_message = "Write me XOR Properties and its Applications."

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    step["messages"][-1].pretty_print()

    