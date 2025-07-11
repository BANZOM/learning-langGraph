import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Configuration
MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
EMBEDDING_MODEL = "models/text-embedding-004"
COLLECTION_NAME = "rag_agent_collection"
DOCUMENTS_DIR = os.path.join(os.getcwd(), "AIAgents", "Documents")
VECTOR_STORE_DIR = os.path.join(os.getcwd(), "AIAgents", "VectorStore")
GRAPH_IMAGE_PATH = os.path.join(os.getcwd(), "AIAgents", "imgs", "agent-5-graph.png")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_KWARGS = {"k": 5}

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def create_vector_store():
    """Creates a new Chroma vector store from documents."""
    print("Creating a new vector store...")

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)

    documents = []
    for file in os.listdir(DOCUMENTS_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCUMENTS_DIR, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
        
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR
    )

    print(f"Vector store created and persisted at {VECTOR_STORE_DIR} with collection name '{COLLECTION_NAME}'.")
    return vector_db

def load_existing_vector_store():
    """Loads an existing Chroma vector store."""
    print("Using existing vector store...")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )

# Create or load vector store
if input("Do you want to create a new vector store? (yes/no): ").strip().lower() == "yes":
    vector_db = create_vector_store()
else:
    vector_db = load_existing_vector_store()

# Retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs=SEARCH_KWARGS
)

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieves relevant documents based on the provided query.
    This tool is used to fetch documents from the vector store that are relevant to the user's query.
    It uses a retriever to find the most similar documents based on the embeddings.
    
    Args:
        query (str): The query to search for relevant documents.
    
    Returns:
        str: A string containing the retrieved documents.
    """
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results]) 

tools_dict = {"retrieve_documents": retrieve_documents}
llm = llm.bind_tools([retrieve_documents])

class AgentState(TypedDict):
    """
    AgentState defines the structure of the agent's state, which includes a sequence of messages.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Conditional Node
def should_continue(state: AgentState) -> bool:
    """
    Determines whether to continue the conversation based on the last AI message.
    If the last AI message is a ToolMessage, it returns True to continue.
    Otherwise, it returns False to end the conversation.
    
    Args:
        state (AgentState): The current state of the agent.
    
    Returns:
        bool: True if the last message is a ToolMessage, False otherwise.
    """
    last_message = state["messages"][-1]
    return bool(last_message.tool_calls)

# Node 1:
def start_node(state: AgentState) -> AgentState:
    """
    The starting node of the agent's state graph.
    It initializes the conversation with a system message.
    
    Args:
        state (AgentState): The current state of the agent.
    
    Returns:
        AgentState: The updated state with the initial messages.
    """
    system_message = SystemMessage(content="""You are an intelligent AI assistant who answers questions based on the context loaded into your knowledge base.
    Use the retriever tool "retrieve_documents" available to fetches the similar content based on user's query. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
    If you couldn't find the answer from context, just say you cannot get the answer from context, and give a general overview about the query from your understanding of question.
    """)
    return {"messages": [system_message]}

# Node 2:
def query_node(state: AgentState) -> AgentState:
    """
    This is the query node where the agent query the LLM with the current messages on the state.
    If needed, it retrieves relevant documents using the 'retrieve_documents' tool.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        AgentState: The updated state with the new messages after querying the LLM.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Node 3: Retrieve Node
def retrieve_node(state: AgentState) -> AgentState:
    """
    This node retrieves relevant documents based on the last AI message.
    It uses the 'retrieve_documents' tool to fetch documents related to the user's query.
    
    Args:
        state (AgentState): The current state of the agent.
    
    Returns:
        AgentState: The updated state with the retrieved documents appended to the messages.
    """
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_name = t['name']
        query = t['args'].get('query', '')
        print(f"Calling Tool: {tool_name} with query: {query}")
        
        if tool_name not in tools_dict:
            print(f"\nTool: {tool_name} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[tool_name].invoke(query)
            print(f"Result length: {len(str(result))}")
        
        results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

# State Graph Structure
graph = StateGraph(AgentState)
graph.add_node("INIT", start_node)
graph.add_node("QUERY", query_node)
graph.add_node("RETRIEVE", retrieve_node)

graph.add_edge(START, "INIT")
graph.add_edge("INIT", "QUERY")
graph.add_conditional_edges(
    "QUERY",
    should_continue,
    {
        True: "RETRIEVE",
        False: END
    }
)
graph.add_edge("RETRIEVE", "QUERY")

app = graph.compile()

if __name__ == "__main__":
    initial_state = AgentState(messages=[])

    while True:
        query = input("You: ")
        if query.strip().lower() == "exit":
            print("Exiting the conversation. Goodbye!")
            break
        initial_state["messages"].append(HumanMessage(content=query))
        state = app.invoke(initial_state)
        response = state["messages"][-1]
        
        print(f"AI: {response.content}")

        initial_state = state

    # Save the graph image to a file
    image_bytes = app.get_graph().draw_mermaid_png()
    with open(GRAPH_IMAGE_PATH, "wb") as f:
        f.write(image_bytes)