## Imports
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool #annotated with @tool decorator to define a tool that can be used by the LLM

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # Merges two lists of messages, updating existing messages by ID
from langgraph.prebuilt import ToolNode  # Node that runs the tools called in the last AIMessage

import os
from dotenv import load_dotenv
load_dotenv()

# global variable to hold the document content
document_content = ""

class AgentState(TypedDict):
    """
    AgentState defines the structure of the agent's state, which includes a sequence of messages.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tool 1: Update Tool
@tool
def update_document(content: str) -> str:
    """ 
    Keeps the document content updated with the provided content.
    This tool is used to update the document with new content.
    It can be called multiple times to modify the document as needed.
    It does not save the document, just updates the in-memory content.
    This is useful for iterative updates before finalizing the document. 
    Args:
        content (str): The content to update the document with.
    Returns:
        str: A confirmation message indicating the document has been updated.
    """
    global document_content
    document_content = content
    return f"The document has been updated. and the content is now: {document_content}"

@tool
def save_document(filename: str) -> str:
    """
    Saves the current document content to a file.
    This tool is used to save the document content to a specified file.
    It creates a directory if it does not exist and writes the content to the file.
    This is the final step to persist the document after all updates are done.
    Args:
        filename (str): The name of the file to save the document content to.
    Returns:
        str: A confirmation message indicating the document has been saved.
    Raises:
        Exception: If there is an error while saving the document.
    """
    global document_content
    if not filename:
        raise Exception("Filename cannot be empty.")
    
    try:
        file_path = os.path.join(os.getcwd(), "AIAgents", "Documents")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename = os.path.join(file_path, filename)

        with open(filename, "w") as file:
            file.write(document_content)
        return f"The document has been saved to {filename}."
    except Exception as e:
        raise Exception(f"An error occurred while saving the document: {str(e)}")

tools = [
    update_document,
    save_document
]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=1,
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)

## Node 1: Init Node
def init_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and save documents.
    
    - If the user wants to update or modify content, use the 'update_document' tool with the complete updated content.
    - If the user wants to view the current document content, you can directly provide them with the content.
    - If the user wants to save and finish, you need to use the 'save_document' tool.
    - Make sure to always show the current document state after modifications.
    - Always respond in a friendly and helpful manner.
    - If the user asks for help, provide clear instructions on how to use the tools.
    - If the user asks to save the document, confirm the filename and save the content.
    - If the user asks to finish, confirm that the document has been saved and end the conversation.
    - Always remember to keep the conversation engaging and interactive.
                                  
    - Current Document Content: 
    \n{document_content}\n
    """)

    return {
        "messages": [system_prompt]
    }

## Node 2: Query Node
def query_node(state: AgentState) -> AgentState:
    query = input("You: ")
    history = list(state["messages"]) + [HumanMessage(query)]
    response = model.invoke(history)

    print("AI: " + response.content)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\nTools to use: {[ tool["name"] for tool in response.tool_calls]} \n")

    return {
        "messages": history + [response]
    }

## Condition Node
def condition_node(state: AgentState) -> str:
    """
    Checks the messages in reverse order to determine if the last message contains tool calls.
    if it use save_document tool, it returns "end" to indicate that the conversation should end.
    If it contains any other tool calls, it returns "continue" to get back to the query node.
    """
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage):
            if message.name == "save_document":
                return "end"
            else:
                return "continue"
    return "continue"


### Helper function to preety print the tool calls
def pretty_print_tool_calls(messages):
    print(messages)


# State Graph Structure
graph = StateGraph(AgentState)
graph.add_node("INIT", init_node)
graph.add_node("QUERY", query_node)
graph.add_node("TOOL", ToolNode(tools=tools))  
graph.add_conditional_edges(
    "TOOL", 
    condition_node, 
    {
        "continue": "QUERY",
        "end": END
    }
)
graph.add_edge(START, "INIT") 
graph.add_edge("INIT", "QUERY")  # Start the conversation with the query node
graph.add_edge("QUERY", "TOOL")  # After querying, go to the tool

app = graph.compile()

# Save the graph image to a file
image_bytes = app.get_graph().draw_mermaid_png()
image_path = os.path.join(os.getcwd(), "AIAgents", "imgs", "agent-4-graph.png")
with open(image_path, "wb") as f:
    f.write(image_bytes)


if __name__ == "__main__":
    initial_state = AgentState(messages=[])
    while True:
        try:
            final_state = app.invoke(initial_state)
            initial_state = final_state 

        except KeyboardInterrupt:
            print("\nExiting the chat.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break
