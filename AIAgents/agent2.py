"""
A Conversational Agent that uses Google Gemini to chat with the user.
It supports multiple inputs and outputs, allowing for a more complex interaction.
"""

from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import json
load_dotenv()

# Initialize the Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# Agent State Definition
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Node function for the chat interaction
def chat(state: AgentState) -> AgentState:
    """
    Chat node that takes the current state, invokes the LLM, and appends the response to the messages.
    Args:
        state (AgentState): The current state containing messages.
    Returns:
        AgentState: The updated state with the new message appended.
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# State graph definition
graph = StateGraph(AgentState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile the agent
agent = graph.compile()

# Save the graph to a file
image_bytes = agent.get_graph().draw_mermaid_png()
image_path = os.path.join(os.getcwd(), "AIAgents", "imgs", "agent-2-graph.png")
with open(image_path, "wb") as f:
    f.write(image_bytes)

# Run the graph 
conversation_history = []

# Load previous conversation history if it exists
history_path = os.path.join(os.getcwd(), "AIAgents", "ConvoHistory", "conversation_history_agent_2.json")
if os.path.exists(history_path):
    with open(history_path, "r") as f:
        loaded_history = json.load(f)
        for message_data in loaded_history:
            if message_data["type"] == "human":
                conversation_history.append(HumanMessage(content=message_data["content"]))
            elif message_data["type"] == "ai":
                conversation_history.append(AIMessage(content=message_data["content"]))

# Initial Conversation State
# print("Initial Convo History : ", conversation_history)

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break

    conversation_history.append(HumanMessage(content=query))

    initial_state = AgentState(messages=conversation_history)
    final_state = agent.invoke(initial_state)

    conversation_history = final_state["messages"]
    if final_state["messages"]:
        # Print the last AI message in the conversation
        if isinstance(final_state["messages"][-1], AIMessage):
            print("AI: ", final_state["messages"][-1].content)

    else:
        print("AI: No response generated.")

    # print("\n--- Conversation History ---")
    # print(conversation_history)


#Save the conversation history to a file as a JSON object
history_path = os.path.join(os.getcwd(), "AIAgents", "ConvoHistory", "conversation_history_agent_2.json")
with open(history_path, "w") as f:
    # Convert messages to dictionaries for JSON serialization
    serializable_history = []
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            serializable_history.append({"type": "human", "content": message.content})
        elif isinstance(message, AIMessage):
            serializable_history.append({"type": "ai", "content": message.content})
    json.dump(serializable_history, f)
