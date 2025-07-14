import os
import json
import datetime
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Define the enhanced state structure
class EnhancedChatState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    user_input: Annotated[str, "The current user input"]
    conversation_id: Annotated[str, "Unique identifier for the conversation"]
    metadata: Annotated[Dict[str, Any], "Additional conversation metadata"]

class ConversationManager:
    """Manages conversation persistence and metadata."""
    
    def __init__(self, save_dir: str = "conversations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_conversation(self, state: EnhancedChatState) -> None:
        """Save conversation to a JSON file."""
        conversation_data = {
            "conversation_id": state["conversation_id"],
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": state["metadata"],
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content
                }
                for msg in state["messages"]
            ]
        }
        
        filename = f"{state['conversation_id']}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, conversation_id: str) -> Optional[EnhancedChatState]:
        """Load a conversation from a JSON file."""
        filename = f"{conversation_id}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to LangChain message objects
            messages = []
            for msg_data in data["messages"]:
                if msg_data["type"] == "HumanMessage":
                    messages.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["type"] == "AIMessage":
                    messages.append(AIMessage(content=msg_data["content"]))
                elif msg_data["type"] == "SystemMessage":
                    messages.append(SystemMessage(content=msg_data["content"]))
            
            return {
                "messages": messages,
                "user_input": "",
                "conversation_id": data["conversation_id"],
                "metadata": data["metadata"]
            }
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None

# Initialize components
conversation_manager = ConversationManager()

# Initialize the language model with error handling
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing language model: {e}")
    print("Please check your OpenAI API key in the .env file")
    exit(1)

def get_user_input(state: EnhancedChatState) -> EnhancedChatState:
    """Get user input from the console with enhanced command handling."""
    if "user_input" not in state or not state["user_input"]:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                # Save conversation before exiting
                conversation_manager.save_conversation(state)
                return END
            elif user_input.lower() == 'save':
                conversation_manager.save_conversation(state)
                print("💾 Conversation saved!")
                return {"user_input": ""}
            elif user_input.lower() == 'history':
                print_conversation_history(state)
                return {"user_input": ""}
            elif user_input.lower() == 'clear':
                print("🗑️ Conversation history cleared!")
                return {
                    "messages": [SystemMessage(content="You are a helpful AI assistant.")],
                    "user_input": "",
                    "conversation_id": state["conversation_id"],
                    "metadata": state["metadata"]
                }
            elif user_input.lower().startswith('load '):
                conversation_id = user_input[5:].strip()
                loaded_state = conversation_manager.load_conversation(conversation_id)
                if loaded_state:
                    print(f"📂 Loaded conversation: {conversation_id}")
                    return loaded_state
                else:
                    print(f"❌ Conversation {conversation_id} not found")
                    return {"user_input": ""}
            
            return {"user_input": user_input}
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            conversation_manager.save_conversation(state)
            return END
        except EOFError:
            print("\n\nGoodbye! 👋")
            conversation_manager.save_conversation(state)
            return END
    
    return state

def print_conversation_history(state: EnhancedChatState) -> None:
    """Print the conversation history in a formatted way."""
    print("\n📜 Conversation History:")
    print("-" * 50)
    
    for i, message in enumerate(state["messages"], 1):
        if isinstance(message, HumanMessage):
            print(f"{i}. You: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"{i}. Assistant: {message.content}")
        elif isinstance(message, SystemMessage):
            print(f"{i}. System: {message.content}")
        print()
    
    print(f"Total messages: {len(state['messages'])}")
    print(f"Conversation ID: {state['conversation_id']}")

def process_message(state: EnhancedChatState) -> EnhancedChatState:
    """Process the user message and generate a response with enhanced error handling."""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    # Add user message to history
    messages.append(HumanMessage(content=user_input))
    
    try:
        # Generate response using the language model
        response = llm.invoke(messages)
        
        # Add AI response to history
        messages.append(AIMessage(content=response.content))
        
        # Print the response
        print(f"\nAssistant: {response.content}")
        
        # Update metadata
        metadata = state.get("metadata", {})
        metadata["last_updated"] = datetime.datetime.now().isoformat()
        metadata["message_count"] = len(messages)
        
        # Clear user input for next iteration
        return {
            "messages": messages,
            "user_input": "",
            "conversation_id": state["conversation_id"],
            "metadata": metadata
        }
    
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        print(f"\nAssistant: {error_message}")
        
        # Add error message to history
        messages.append(AIMessage(content=error_message))
        
        return {
            "messages": messages,
            "user_input": "",
            "conversation_id": state["conversation_id"],
            "metadata": state.get("metadata", {})
        }

def create_enhanced_chatbot_graph():
    """Create the enhanced LangGraph workflow for the chatbot."""
    # Create the graph
    workflow = StateGraph(EnhancedChatState)
    
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
    """Main function to run the enhanced chatbot."""
    print("🤖 Welcome to the Enhanced LangGraph Chatbot!")
    print("Commands:")
    print("  - Type 'save' to save the conversation")
    print("  - Type 'history' to view conversation history")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'load <conversation_id>' to load a saved conversation")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("-" * 60)
    
    # Create the chatbot graph
    chatbot = create_enhanced_chatbot_graph()
    
    # Generate unique conversation ID
    conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize state with system message
    initial_state = {
        "messages": [SystemMessage(content="You are a helpful AI assistant. Be conversational and engaging.")],
        "user_input": "",
        "conversation_id": conversation_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "model": "gpt-3.5-turbo",
            "message_count": 1
        }
    }
    
    # Run the chatbot
    try:
        for event in chatbot.stream(initial_state):
            # The graph will handle the conversation flow
            pass
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        conversation_manager.save_conversation(initial_state)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        conversation_manager.save_conversation(initial_state)

if __name__ == "__main__":
    main() 