### RAG Agent
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
from langchain_community.vectorstores import Chroma

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

if input("Do you want to create a new vector store? (yes/no): ").strip().lower() == "yes":
    print("Creating a new vector store...")
    collection_name = "rag_agent_collection"
    documents_dir = os.path.join(os.getcwd(), "AIAgents", "Documents")
    vector_store_dir = os.path.join(os.getcwd(), "AIAgents", "VectorStore")

    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)

    documents = []
    for file in os.listdir(documents_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(documents_dir, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
        
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=vector_store_dir
    )

    print(f"Vector store created and persisted at {vector_store_dir} with collection name '{collection_name}'.")
else:
    print("Using existing vector store...")


