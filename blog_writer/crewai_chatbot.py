import os
import datetime
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from blog_agent import blog_agent
from pydantic import BaseModel
from IPython.display import Image, display

# Load environment variables
load_dotenv()

class BlogRequestResponse(BaseModel):
    content: str

# Define the enhanced state structure
class CrewAIChatState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    user_input: Annotated[str, "The current user input"]
    conversation_id: Annotated[str, "Unique identifier for the conversation"]
    metadata: Annotated[Dict[str, Any], "Additional conversation metadata"]
    blog_content: Annotated[Optional[str], "Generated blog content"]
    should_end: Annotated[bool, "Flag to indicate if conversation should end"]

# Initialize the language model
try:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing language model: {e}")
    print("Please check your OpenAI API key in the .env file")
    exit(1)

def get_user_input(state: CrewAIChatState) -> CrewAIChatState:
    """Get user input from the console with enhanced command handling."""
    if "user_input" not in state or not state["user_input"]:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! 👋")
                return {
                    "user_input": "",
                    "should_end": True
                }
            elif user_input.lower() == 'help':
                print_help()
                return {"user_input": "", "should_end": False}
            elif user_input.lower() == 'history':
                print_conversation_history(state)
                return {"user_input": "", "should_end": False}
            elif user_input.lower() == 'clear':
                print("🗑️ Conversation history cleared!")
                return {
                    "messages": [SystemMessage(content="You are a helpful AI assistant with blog writing capabilities.")],
                    "user_input": "",
                    "conversation_id": state["conversation_id"],
                    "metadata": state["metadata"],
                    "blog_content": None,
                    "should_end": False
                }
            
            return {"user_input": user_input, "should_end": False}
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            return {"user_input": "", "should_end": True}
        except EOFError:
            print("\n\nGoodbye! 👋")
            return {"user_input": "", "should_end": True}
    
    return state

def print_help():
    """Print help information about available commands and blog writing capabilities."""
    help_text = """
🤖 CrewAI Chatbot Help
----------------------------------------
Commands:
  - help     Show this help message
  - history  View conversation history  
  - clear    Clear conversation history
  - quit     End conversation (also: exit, bye)

Blog Writing:
  - Ask me to write a blog about any topic
  Examples:
  • "Write a blog about artificial intelligence"
  • "Blog about climate change" 
  • "Create a blog post about healthy eating"

  I'll use CrewAI agents to research and write the blog post."""

    print(help_text)

def print_conversation_history(state: CrewAIChatState) -> None:
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

def process_message(state: CrewAIChatState) -> CrewAIChatState:
    """Process the user message and generate a response."""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    # Add user message to history
    messages.append(HumanMessage(content=user_input))
    
    try:
        # Check if this is a blog writing request using LLM
        is_blog_request_prompt = HumanMessage(content=f"""
        Determine if this is a request to write a blog post. 
        Only respond with 'true' or 'false'.
        User message: {user_input}
        """)
        
            
        is_blog_response = BlogRequestResponse(content=llm.invoke([is_blog_request_prompt]).content)
        is_blog_request = is_blog_response.content.strip().lower() == 'true'
        
        if is_blog_request:
            # Use CrewAI to write the blog
            print("\n🤖 Researching and writing your blog post...")
            blog_content = blog_agent.write_blog(user_input)
            
            # Add AI response to history
            messages.append(AIMessage(content=blog_content.raw))
            
            # Print the blog post with formatting
            print("\n📝 Blog Post:")
            print("=" * 60)
            print(blog_content)
            print("=" * 60)
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["last_updated"] = datetime.datetime.now().isoformat()
            metadata["message_count"] = len(messages)
            metadata["blogs_written"] = metadata.get("blogs_written", 0) + 1
            
            return {
                "messages": messages,
                "user_input": "",
                "conversation_id": state["conversation_id"],
                "metadata": metadata,
                "blog_content": blog_content,
                "should_end": False
            }
        else:
            # Regular conversation
            response = llm.invoke(messages)
            
            # Add AI response to history
            messages.append(AIMessage(content=response.content))
            
            # Print the response
            print(f"\nAssistant: {response.content}")
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["last_updated"] = datetime.datetime.now().isoformat()
            metadata["message_count"] = len(messages)
            
            return {
                "messages": messages,
                "user_input": "",
                "conversation_id": state["conversation_id"],
                "metadata": metadata,
                "blog_content": state.get("blog_content"),
                "should_end": False
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
            "metadata": state.get("metadata", {}),
            "blog_content": state.get("blog_content"),
            "should_end": False
        }

def should_continue(state: CrewAIChatState) -> str:
    """Determine whether to continue the conversation or end it."""
    if state.get("should_end", False):
        return "end"
    return "continue"

def create_crewai_chatbot_graph():
    """Create the CrewAI-enhanced LangGraph workflow for the chatbot."""
    # Create the graph
    workflow = StateGraph(CrewAIChatState)
    
    # Add nodes
    workflow.add_node("get_input", get_user_input)
    workflow.add_node("process_message", process_message)
    
    # Add conditional edge from get_input
    workflow.add_conditional_edges(
        "get_input",
        should_continue,
        {
            "continue": "process_message",
            "end": END
        }
    )
    
    # Add edge from process_message back to get_input
    workflow.add_edge("process_message", "get_input")
    
    # Set entry point
    workflow.set_entry_point("get_input")
    
    # Compile the graph
    return workflow.compile()

def main():
    """Main function to run the CrewAI-enhanced chatbot."""
    welcome_message = """
    🤖 Welcome to the CrewAI-Enhanced LangGraph Chatbot!
    I can help you with conversations and write blog posts using AI agents!
    Type 'help' for available commands.
    """ 
    print(welcome_message)
    print("-" * 60)
    # Create the chatbot graph
    chatbot = create_crewai_chatbot_graph()

    # Save the graph as a PNG file
    with open("graph.png", "wb") as f:
        f.write(chatbot.get_graph().draw_mermaid_png())

    # Generate unique conversation ID
    conversation_id = f"crewai_conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize state with system message
    initial_state = {
        "messages": [SystemMessage(content="You are a helpful AI assistant with blog writing capabilities using CrewAI agents.")],
        "user_input": "",
        "conversation_id": conversation_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "model": "gpt-3.5-turbo",
            "message_count": 1,
            "blogs_written": 0
        },
        "blog_content": None,
        "should_end": False
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