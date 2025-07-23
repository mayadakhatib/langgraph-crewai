from typing import Dict, Any, Optional
import asyncio
import uuid
from quart import Quart, request, jsonify
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


# Define the state structure
class GraphState(TypedDict):
    messages: list
    processing_complete: bool


def request_human_input(state: GraphState) -> Dict[str, Any]:
    """Node that requests human input using interrupt()"""
    print("Requesting human input...")
    
    # Add a message requesting input
    messages = state.get("messages", [])
    # messages.append({
    #     "role": "assistant",
    #     "content": "I need your input to continue. Please provide your response."
    # })
    
    # Use interrupt to pause execution and wait for human input
    user_input = interrupt("Please provide your response:")
    
    # Process the received input
    messages.append({
        "role": "user", 
        "content": user_input
    })
    
    messages.append({
        "role": "assistant",
        "content": f"Thank you for your input: '{user_input}'. Processing complete!"
    })
    
    return {
        "messages": messages,
        "processing_complete": True
    }


# Create the graph
def create_graph():
    workflow = StateGraph(GraphState)
    
    # Add the single node that handles human input
    workflow.add_node("request_input", request_human_input)
    
    # Add edges
    workflow.add_edge(START, "request_input")
    workflow.add_edge("request_input", END)
    
    # Set up checkpointer - required for interrupts
    checkpointer = MemorySaver()
    
    # Compile the graph with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


# Global variables to store graph and active threads
graph_app = create_graph()
active_threads = {}


# Create Quart application
app = Quart(__name__)


@app.route('/chat', methods=['POST'])
async def chat():
    """
    Main endpoint for chat interactions.
    Dynamically determines whether to start new conversation or resume based on thread state.
    """
    try:
        data = await request.get_json()
        
        thread_id = data.get('thread_id')
        user_input = data.get('user_input', '')
        
        # If no thread_id provided, start new conversation
        if thread_id is None:
            return await start_new_conversation()
        
        # Thread ID provided - check if it exists and its state
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get the current state of the thread
            current_state = graph_app.get_state(config)
            
            # If state doesn't exist, start new conversation with this thread_id
            if current_state.values is None:
                return await start_new_conversation(thread_id)
            
            # If state exists, check if it's interrupted (next steps exist)
            if current_state.next:
                # Thread is interrupted, resume with user input
                return await resume_conversation(thread_id, user_input, config)
            else:
                # Thread is completed, could start a new one or return completed status
                return jsonify({
                    "thread_id": thread_id,
                    "status": "already_completed",
                    "message": "This conversation has already been completed",
                    "messages": current_state.values.get("messages", []),
                    "processing_complete": current_state.values.get("processing_complete", False)
                })
                
        except Exception as e:
            print(f"Error checking thread state: {e}")
            # If we can't get state, try to start new conversation
            return await start_new_conversation(thread_id)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


async def start_new_conversation(thread_id: str = None):
    """Start a new conversation, optionally with specified thread_id"""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial state
    initial_state = {
        "messages": [],
        "processing_complete": False
    }
    
    print(f"Starting new conversation with thread_id: {thread_id}")
    
    # Run the graph until interrupt
    try:
        for event in graph_app.stream(initial_state, config=config, stream_mode="updates"):
            print(f"Event: {event}")
            
            # Check if we hit an interrupt
            if "__interrupt__" in event:
                interrupt_info = event["__interrupt__"][0]
                print(f"Hit interrupt: {interrupt_info.value}")
                
                # Store the thread info
                active_threads[thread_id] = {
                    "config": config,
                    "interrupt_message": interrupt_info.value
                }
                
                return jsonify({
                    "thread_id": thread_id,
                    "status": "interrupted",
                    "message": interrupt_info.value,
                    "requires_input": True
                })
                
    except Exception as e:
        print(f"Stream error: {e}")
        return jsonify({"error": f"Stream error: {str(e)}"}), 500
    
    # If we get here, no interrupt occurred (shouldn't happen in this example)
    return jsonify({
        "thread_id": thread_id,
        "status": "completed",
        "message": "No interrupt occurred"
    })


async def resume_conversation(thread_id: str, user_input: str, config: dict):
    """Resume an interrupted conversation with user input"""
    if not user_input:
        return jsonify({
            "error": "user_input is required to resume conversation"
        }), 400
    
    print(f"Resuming conversation with thread_id: {thread_id}")
    print(f"User input: {user_input}")
    
    # Resume execution with Command(resume=user_input)
    messages = []
    
    try:
        for event in graph_app.stream(
            Command(resume=user_input), 
            config=config, 
            stream_mode="updates"
        ):
            print(f"Resume event: {event}")
            
            # Extract messages from the final state
            if "request_input" in event and event["request_input"]:
                messages = event["request_input"].get("messages", [])
        
        # Clean up the thread since processing is complete
        if thread_id in active_threads:
            del active_threads[thread_id]
        
        return jsonify({
            "thread_id": thread_id,
            "status": "completed",
            "message": "Processing complete",
            "messages": messages,
            "requires_input": False,
            "processing_complete": True
        })
        
    except Exception as e:
        print(f"Resume error: {e}")
        return jsonify({"error": f"Resume error: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


@app.route('/threads/<thread_id>/state', methods=['GET'])
async def get_thread_state(thread_id: str):
    """Get the current state of a thread"""
    if thread_id not in active_threads:
        return jsonify({"error": "Thread not found"}), 404
    
    config = active_threads[thread_id]["config"]
    
    try:
        state = graph_app.get_state(config)
        
        return jsonify({
            "thread_id": thread_id,
            "state": state.values,
            "next_steps": state.next,
            "interrupt_message": active_threads[thread_id].get("interrupt_message", "")
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get state: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
async def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "active_threads": len(active_threads),
        "thread_ids": list(active_threads.keys())
    })


if __name__ == '__main__':
    print("Starting LangGraph Human-in-the-Loop Quart Application...")
    print("\nThis app demonstrates LangGraph's interrupt() function for human input")
    print("\nAPI Endpoints:")
    print("POST /chat - Main chat endpoint (dynamically detects start/resume)")
    print("  - Start new: {}")
    print("  - Start with specific thread: {\"thread_id\": \"<id>\"}")
    print("  - Resume: {\"thread_id\": \"<id>\", \"user_input\": \"<input>\"}")
    print("GET /threads/<thread_id>/state - Get thread state")
    print("GET /health - Health check")
    
    print("\nExample usage:")
    print("1. Start new conversation:")
    print("   curl -X POST http://localhost:5000/chat \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{}'")
    print()
    print("2. Resume conversation (use thread_id from step 1):")
    print("   curl -X POST http://localhost:5000/chat \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"thread_id\": \"<thread_id>\", \"user_input\": \"Hello World\"}'")
    print()
    print("3. Start with specific thread ID:")
    print("   curl -X POST http://localhost:5000/chat \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"thread_id\": \"my-custom-thread\"}'")
    print()
    print("4. Check thread state:")
    print("   curl http://localhost:5000/threads/<thread_id>/state")
    
    app.run(debug=True, host='0.0.0.0', port=5000)