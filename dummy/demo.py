#!/usr/bin/env python3
"""
Demo script for the LangGraph Chatbot
This script demonstrates how to use the chatbot programmatically.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dummy.enhanced_chatbot import create_enhanced_chatbot_graph, EnhancedChatState
from langchain_core.messages import SystemMessage
import datetime

def run_demo():
    """Run a demo conversation with the chatbot."""
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return
    
    print("🚀 Starting LangGraph Chatbot Demo")
    print("=" * 50)
    
    # Create the chatbot
    chatbot = create_enhanced_chatbot_graph()
    
    # Generate conversation ID
    conversation_id = f"demo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize state
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful AI assistant. Be conversational and engaging. Keep responses concise.")
        ],
        "user_input": "",
        "conversation_id": conversation_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "model": "gpt-3.5-turbo",
            "message_count": 1,
            "demo_mode": True
        }
    }
    
    # Demo conversation
    demo_messages = [
        "Hello! Can you tell me about LangGraph?",
        "That's interesting! How does it compare to regular chatbots?",
        "Can you give me a simple example of how to use it?",
        "Thanks! That was very helpful."
    ]
    
    current_state = initial_state
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n--- Demo Message {i} ---")
        print(f"User: {message}")
        
        # Set the user input
        current_state["user_input"] = message
        
        # Process the message
        try:
            # Run one step of the graph
            for event in chatbot.stream(current_state):
                # Get the final state from the event
                if hasattr(event, 'values'):
                    current_state = event.values
                elif isinstance(event, dict):
                    current_state = event
                break
        except Exception as e:
            print(f"Error processing message: {e}")
            break
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print(f"Conversation ID: {conversation_id}")
    print(f"Total messages: {len(current_state['messages'])}")

def interactive_demo():
    """Run an interactive demo where you can chat with the bot."""
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return
    
    print("🎮 Interactive LangGraph Chatbot Demo")
    print("Type your messages and press Enter to chat!")
    print("Type 'quit' to exit the demo.")
    print("=" * 50)
    
    # Import and run the enhanced chatbot
    from dummy.enhanced_chatbot import main as run_chatbot
    run_chatbot()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Chatbot Demo")
    parser.add_argument(
        "--mode", 
        choices=["auto", "interactive"], 
        default="auto",
        help="Demo mode: 'auto' for automated demo, 'interactive' for manual chat"
    )
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        run_demo()
    else:
        interactive_demo() 