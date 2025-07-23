from typing import Dict, Any, Optional
import asyncio
import uuid
from quart import Quart, request, jsonify
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
import sqlite3
import os


# Define the state structure
class GraphState(TypedDict):
    messages: list
    processing_complete: bool


def request_human_input(state: GraphState) -> Dict[str, Any]:
    """Node that requests human input using interrupt()"""
    print("Requesting human input...")
    
    # Add a message requesting input
    messages = state.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": "I need your input to continue. Please provide your response."
    })
    
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


# Create the graph with SQLite checkpointer
def create_graph(db_path: str = "checkpoints.db"):
    """Create LangGraph with SQLite persistence"""
    workflow = StateGraph(GraphState)
    
    # Add the single node that handles human input
    workflow.add_node("request_input", request_human_input)
    
    # Add edges
    workflow.add_edge(START, "request_input")
    workflow.add_edge("request_input", END)
    
    # Set up SQLite checkpointer
    # Create database directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Initialize SQLite connection and create tables
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # SqliteSaver will create necessary tables automatically
    checkpointer = SqliteSaver(conn)
    
    # Compile the graph with SQLite checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    
    return app, checkpointer


# Global variables to store graph and database connection
DB_PATH = "data/checkpoints.db"  # You can customize this path
graph_app, sqlite_checkpointer = create_graph(DB_PATH)


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
            # Get the current state of the thread from SQLite
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
                
                # State is automatically saved to SQLite by the checkpointer
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
        
        # State is automatically saved to SQLite, no manual cleanup needed
        
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


@app.route('/threads/<thread_id>/state', methods=['GET'])
async def get_thread_state(thread_id: str):
    """Get the current state of a thread from SQLite"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = graph_app.get_state(config)
        
        if state.values is None:
            return jsonify({"error": "Thread not found"}), 404
        
        return jsonify({
            "thread_id": thread_id,
            "state": state.values,
            "next_steps": state.next,
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "step": state.step,
            "tasks": [task.name for task in state.tasks] if state.tasks else []
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get state: {str(e)}"}), 500


@app.route('/threads', methods=['GET'])
async def list_threads():
    """List all threads stored in the database"""
    try:
        # Get all threads from the SQLite database
        # Note: This requires direct database access as LangGraph doesn't provide a list method
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query to get unique thread IDs
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_ts DESC")
        thread_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            "threads": thread_ids,
            "count": len(thread_ids)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to list threads: {str(e)}"}), 500


@app.route('/threads/<thread_id>', methods=['DELETE'])
async def delete_thread(thread_id: str):
    """Delete a specific thread from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete all checkpoints for this thread
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return jsonify({
                "message": f"Thread {thread_id} deleted successfully",
                "deleted_checkpoints": deleted_count
            })
        else:
            return jsonify({"error": "Thread not found"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to delete thread: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
async def health():
    """Health check endpoint with database info"""
    try:
        # Get database statistics
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
        thread_count = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        conn.close()
        
        return jsonify({
            "status": "healthy",
            "database_path": DB_PATH,
            "database_exists": os.path.exists(DB_PATH),
            "total_threads": thread_count,
            "total_checkpoints": checkpoint_count
        })
    except Exception as e:
        return jsonify({
            "status": "healthy",
            "database_path": DB_PATH,
            "database_exists": os.path.exists(DB_PATH),
            "error": f"Database query failed: {str(e)}"
        })


if __name__ == '__main__':
    print("Starting LangGraph Human-in-the-Loop Quart Application with SQLite Storage...")
    print(f"Database path: {DB_PATH}")
    print("\nThis app demonstrates LangGraph's interrupt() function with SQLite persistence")
    print("\nAPI Endpoints:")
    print("POST /chat - Main chat endpoint (dynamically detects start/resume)")
    print("  - Start new: {}")
    print("  - Start with specific thread: {\"thread_id\": \"<id>\"}")
    print("  - Resume: {\"thread_id\": \"<id>\", \"user_input\": \"<input>\"}")
    print("GET /threads/<thread_id>/state - Get thread state")
    print("GET /threads - List all threads")
    print("DELETE /threads/<thread_id> - Delete a specific thread")
    print("GET /health - Health check with database info")
    
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
    print("3. List all threads:")
    print("   curl http://localhost:5000/threads")
    print()
    print("4. Check thread state:")
    print("   curl http://localhost:5000/threads/<thread_id>/state")
    print()
    print("5. Delete a thread:")
    print("   curl -X DELETE http://localhost:5000/threads/<thread_id>")
    
    app.run(debug=True, host='0.0.0.0', port=5000)