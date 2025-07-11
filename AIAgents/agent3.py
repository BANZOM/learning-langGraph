"""
Agent 3: ReAct Agent - Uses Tools and Memory
"""

from typing import Annotated, Sequence, TypedDict
"""
Annotated: Used to specify types for function parameters and return values. Example: Annotated[str, "This is a string parameter"].
Sequence: A generic type for a sequence of items, to automatically handle the state updates for Sequence such as Adding new messages to chat history. Example: Sequence[int] for a sequence of integers.
TypedDict: A way to define a dictionary with specific key-value pairs and types.
"""

from dotenv import load_dotenv
load_dotenv()  
from langchain_core.messages import BaseMessage # foundation for all message types
from langchain_core.messages import ToolMessage # passes the tool response to the LLM like content and tall_call_id
from langchain_core.messages import SystemMessage # system message that provides context to the LLM
from langchain_core.tools import tool # Tool that takes in function or coroutine directly.
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode # node that runs the tools called in the last AIMessage.
from langgraph.graph.message import add_messages # Merges two lists of messages, updating existing messages by ID.

import os


# State Schema Definition
class AgentState(TypedDict):
    """
    AgentState defines the structure of the agent's state, which includes a sequence of messages.
    This state is used to maintain the conversation history and manage interactions with the LLM.
    It is a TypedDict, which allows for type checking and validation of the state structure.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

## Tool 1: Addition Tool
@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of the two numbers.
    """
    print(f"Adding {a} and {b}")
    return a + b

## Tool 2: Subtraction Tool
@tool
def subtract_numbers(a: float, b: float) -> float:
    """
    Subtracts the second number from the first.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The result of subtracting b from a.
    """
    print(f"Subtracting {b} from {a}")
    return a - b

## Tool 3: Multiplication Tool
@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiplies two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The product of the two numbers.
    """
    print(f"Multiplying {a} and {b}")
    return a * b

tools = [
    add_numbers, subtract_numbers, multiply_numbers
]
tool_node = ToolNode(tools=tools)

# DEBUG 
#print(f"DEBUG: GEMINI_API_KEY loaded: {os.getenv('GEMINI_API_KEY')}")
    
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)


# Node 1
def model_node(state: AgentState) -> AgentState:
    # response = model.invoke(["You are a helpful assistant."])
    
    # OR
    
    system_prompt = SystemMessage(
        content="You are a banzobot assistant created by Aditya Panwar which reply descriptively. You can do anything + More better with tools. "
    )
    response = model.invoke([system_prompt] + state["messages"])  # Use the messages from the state to maintain context
    
    
    return {
        "messages": [response]
    } # this type of return can be used to update the state with new messages along with the existing ones. Because of the add_messages decorator, it will automatically merge the new messages with the existing ones in the state.
    

# Condition Node
def condition_node(state: AgentState) -> str:
    """
    Checks the last message in the state to determine if it contains tool calls.
    If it does, it returns "continue_with_tool" to indicate that the tool should be executed.
    If not, it returns "end" to indicate that the conversation should end.
    """
    last_message = state["messages"][-1]  # Get the last message in the sequence
    
    if last_message.tool_calls:
        return "continue_with_tool"
    else:
        return "end"
    

# State Graph Structure
graph = StateGraph(AgentState)

graph.add_node("MODEL", model_node)  
graph.add_node("TOOL", tool_node)

graph.add_edge(START, "MODEL")
graph.add_conditional_edges(
    "MODEL", 
    condition_node, 
    {
        "continue_with_tool": "TOOL",
        "end": END
    }
)
graph.add_edge("TOOL", "MODEL")  # Loop back to the model node after tool execution

# Compile the agent
agent = graph.compile()

# Save the graph image to a file
image_bytes = agent.get_graph().draw_mermaid_png()
image_path = os.path.join(os.getcwd(), "AIAgents", "imgs", "agent-3-graph.png")
with open(image_path, "wb") as f:
    f.write(image_bytes)

# Interactive loop to chat with the agent
def format_user_message(text):
    # Convert user input to the expected message format
    return ("user", text)

print("Welcome to Banzobot! Type 'exit' to quit.")
messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break
    messages.append(format_user_message(user_input))
    inputs = {"messages": messages}
    # Get the agent's response (not streaming)
    result = agent.invoke(inputs)
    # The agent returns a dict with "messages" as a list; get the last message
    last_message = result["messages"][-1]
    # If it's a tuple, print the content; else, try to access .content
    if isinstance(last_message, tuple):
        print("Banzobot:", last_message[1])
    else:
        print("Banzobot:", getattr(last_message, "content", str(last_message)))
    # Add the agent's message to the conversation history
    messages.append(last_message)

### End of the interactive loop

## Streaming version
# Uncomment the following lines to use streaming instead of non-streaming
# Helper function to convert inputs to BaseMessage format
# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]  # Get the last message in the stream
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()


# # Run the graph
# inputs = {
#     "messages": [
#         (
#             "user", "Hello! Can you help me with some math? What is the Additon fo 3 and 9.0009 and what is subtraction for 8 and 0.999. then multiply the result by 2.0.. Then tell me about yourself",
#         )
#     ]
# }
# print_stream(agent.stream(inputs, stream_mode="values"))
## End of streaming version
