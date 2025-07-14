import os
from typing import Dict, List, Any, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Define the state structure
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    user_input: Annotated[str, "The current user input"]

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_user_input(state: ChatState) -> ChatState:
    """Get user input from the console."""
    if "user_input" not in state or not state["user_input"]:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            return END
        return {"user_input": user_input}
    return state

def process_message(state: ChatState) -> ChatState:
    """Process the user message and generate a response."""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    # Add user message to history
    messages.append(HumanMessage(content=user_input))
    
    # Generate response using the language model
    response = llm.invoke(messages)
    
    # Add AI response to history
    messages.append(AIMessage(content=response.content))
    
    # Print the response
    print(f"\nAssistant: {response.content}")
    
    # Clear user input for next iteration
    return {
        "messages": messages,
        "user_input": ""
    }

def create_chatbot_graph():
    """Create the LangGraph workflow for the chatbot."""
    # Create the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("process_message", process_message)
    
    # Add edges
    workflow.add_edge("get_input", "process_message")
    workflow.add_edge("process_message", "get_input")
    
    # Set entry point
    workflow.set_entry_point("get_input")
    
    # Compile the graph
    return workflow.compile()

def main():
    """Main function to run the chatbot."""
    print("🤖 Welcome to the LangGraph Chatbot!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("-" * 50)
    
    # Create the chatbot graph
    chatbot = create_chatbot_graph()
    
    # Initialize state
    initial_state = {
        "messages": [],
        "user_input": ""
    }
    
    # Run the chatbot
    try:
        for event in chatbot.stream(initial_state):
            # The graph will handle the conversation flow
            pass
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 