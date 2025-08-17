import os
import datetime
import uuid
from typing import Dict, List, Any, TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from blog_agent import blog_agent
from pydantic import BaseModel
from langgraph.types import Command, interrupt

# Load environment variables
load_dotenv()


# Define the enhanced state structure
class CrewAIChatState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    user_input: Annotated[str, "The current user input"]
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

def detect_exit_with_llm(user_input: str, conversation_context: List[BaseMessage]) -> bool:
    """Use LLM to intelligently detect if user wants to exit the conversation."""
    
    # Create a prompt for exit detection
    exit_detection_prompt = f"""
    Analyze if the user wants to end the conversation based on their input and context.
    
    User's latest input: "{user_input}"
    
    Recent conversation context (last 3 exchanges):
    {format_recent_context(conversation_context[-6:])}
    
    Consider these factors:
    1. Direct exit commands (quit, exit, bye, goodbye)
    2. Gratitude + completion signals (thanks, thank you, that's all, i'm done)
    3. Task completion acknowledgment (got what i needed, perfect, great)
    4. Natural conversation endings (no more questions, all set)
    5. Context clues from the conversation flow
    
    Respond with ONLY "EXIT" if the user wants to end the conversation, or "CONTINUE" if they want to keep chatting.
    
    Response:"""
    
    try:
        response = llm.invoke([HumanMessage(content=exit_detection_prompt)])
        decision = response.content.strip().upper()
        
        return decision == "EXIT"
        
    except Exception as e:
        # Fallback to keyword detection
        fallback_exit_words = ['quit', 'exit', 'bye', 'goodbye', 'end', 'stop']
        return any(word in user_input.lower() for word in fallback_exit_words)

def format_recent_context(messages: List[BaseMessage]) -> str:
    """Format recent conversation context for the LLM."""
    if not messages:
        return "No recent context"
    
    formatted = []
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            formatted.append(f"User {i}: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant {i}: {msg.content}")
    
    return "\n".join(formatted[-6:])  # Last 6 messages (3 exchanges)

def get_user_input(state: CrewAIChatState) -> CrewAIChatState:
    """Get user input using LangGraph interrupt."""
    
    # If we already have user input, process it
    if state.get("user_input") and not state.get("should_end", False):
        return state
    
    # Use interrupt to get user input
    user_input = interrupt(
        {
            "prompt": "Please enter your message:",
            "conversation_context": format_recent_context(state.get("messages", [])),
            "message_count": len(state.get("messages", []))
        }
    )
    
    
    # Handle clear command
    if user_input.lower() == 'clear':
        return {
            "messages": [SystemMessage(content="You are a helpful AI assistant with blog writing capabilities.")],
            "user_input": "",
            "metadata": state["metadata"],
            "blog_content": None,
            "should_end": False
        }
    
    # Check for exit intention using LLM
    if detect_exit_with_llm(user_input, state.get("messages", [])):
        return {"user_input": "", "should_end": True, "blog_content": state.get("blog_content"), "messages": state.get("messages", [])}
    
    return {"user_input": user_input, "should_end": False}

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
        
        is_blog_response = llm.invoke([is_blog_request_prompt])
        is_blog_request = is_blog_response.content.strip().lower() == 'true'
        
        if is_blog_request:
            # Use CrewAI to write the blog
            blog_content = blog_agent.write_blog(user_input)
            
            # Add AI response to history
            messages.append(AIMessage(content=str(blog_content)))
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["last_updated"] = datetime.datetime.now().isoformat()
            metadata["message_count"] = len(messages)
            metadata["blogs_written"] = metadata.get("blogs_written", 0) + 1
            
            return {
                "messages": messages,
                "user_input": "",
                "metadata": metadata,
                "blog_content": str(blog_content),
                "should_end": False
            }
        else:
            # Regular conversation
            response = llm.invoke(messages)
            
            # Add AI response to history
            messages.append(AIMessage(content=response.content))
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata["last_updated"] = datetime.datetime.now().isoformat()
            metadata["message_count"] = len(messages)
            
            return {
                "messages": messages,
                "user_input": "",
                "metadata": metadata,
                "blog_content": state.get("blog_content"),
                "should_end": False
            }
    
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        
        # Add error message to history
        messages.append(AIMessage(content=error_message))
        
        return {
            "messages": messages,
            "user_input": "",
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
    """Create the CrewAI-enhanced LangGraph workflow."""
    
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
    
    # Create checkpointer for state persistence
    checkpointer = InMemorySaver()

    # Compile the graph with checkpointer
    return workflow.compile(checkpointer=checkpointer)


def main():
    """Main function to demonstrate streaming conversation."""
    
    # Example usage
    initial_state = {
        "messages": [SystemMessage(content="You are a helpful AI assistant with blog writing capabilities using CrewAI agents.")],
        "user_input": "",
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "model": "gpt-4o",
            "message_count": 1,
            "blogs_written": 0
        },
        "blog_content": None,
        "should_end": False
    }
        

    print("Starting conversation stream...")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Create the chatbot graph
    chatbot = create_crewai_chatbot_graph()

    for event in chatbot.stream(initial_state,
                                        config=config,
                                        stream_mode="updates"):
        print(f"Event: {event}")
        
    for event in chatbot.stream(
                            Command(resume="Write a blog about artificial intelligence"),
                                        config=config,
                                        stream_mode="updates"):
        print(f"Event: {event}")

    for event in chatbot.stream(
                            Command(resume="Looks good, thanks!"),
                                        config=config,
                                        stream_mode="updates"):
        print(f"Event: {event}")

if __name__ == "__main__":
    main() 