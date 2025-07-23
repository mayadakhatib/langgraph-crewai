# LangGraph Human-in-the-Loop Support Agent

This directory contains multiple applications that demonstrate LangGraph's `interrupt()` function for human input in Quart-based web services.

## Overview

Both applications implement LangGraph workflows that use interrupts to request human input during execution. They manage conversation threads and allow for dynamic conversation flow control, but with different API designs.

## Interrupt in Chat V2 (`chat_with_interrupt_v2.py`)

The `chat_with_interrupt_v2.py` file implements a LangGraph workflow that uses interrupts to request human input during execution. The application manages conversation threads and allows for dynamic conversation flow control.

### API Endpoints

#### POST /chat
Main chat endpoint that dynamically detects whether to start a new conversation or resume an existing one.

**Request Body Options:**
- Start new conversation: `{}`
- Start with specific thread ID: `{"thread_id": "<id>"}`
- Resume conversation: `{"thread_id": "<id>", "user_input": "<input>"}`

#### GET /threads/<thread_id>/state
Get the current state of a specific thread.

#### GET /health
Health check endpoint that returns application status and active thread information.

### Example Usage

#### 1. Start New Conversation
```bash
curl -X POST http://localhost:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{}'
```

#### 2. Resume Conversation
Use the thread_id returned from step 1:
```bash
curl -X POST http://localhost:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{"thread_id": "<thread_id>", "user_input": "Hello World"}'
```

#### 3. Start with Specific Thread ID
```bash
curl -X POST http://localhost:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{"thread_id": "my-custom-thread"}'
```

#### 4. Check Thread State
```bash
curl http://localhost:5000/threads/<thread_id>/state
```

### How It Works

1. **Graph Structure**: The application uses a simple StateGraph with a single node that requests human input using `interrupt()`.

2. **Thread Management**: Each conversation is assigned a unique thread ID and managed through LangGraph's checkpointing system.

3. **Interrupt Handling**: When the graph reaches the `interrupt()` call, execution pauses and waits for human input.

4. **Resume Capability**: Conversations can be resumed by providing the thread ID and user input.

### Running the Basic Chat Application

```bash
cd support_agent
python chat_with_interrupt_v2.py
```

The application will start on `http://localhost:5000` with debug mode enabled.

---

## Zero Shot Agent (`chat_with_interrupt_v1.py`)

The `chat_with_interrupt_v1.py` file implements a similar LangGraph workflow but with a different API design using explicit commands for starting and resuming conversations.

### API Endpoints

#### POST /chat
Main chat endpoint with explicit command-based approach.

**Request Body Options:**
- Start conversation: `{"command": "start"}`
- Resume conversation: `{"command": "resume", "thread_id": "<id>", "user_input": "<input>"}`

#### GET /threads/<thread_id>/state
Get the current state of a specific thread.

#### GET /health
Health check endpoint that returns application status and active thread information.

### Example Usage

#### 1. Start Conversation
```bash
curl -X POST http://localhost:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{"command": "start"}'
```

#### 2. Resume with User Input
Use the thread_id returned from step 1:
```bash
curl -X POST http://localhost:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{"command": "resume", "thread_id": "<thread_id>", "user_input": "Hello World"}'
```

#### 3. Check Thread State
```bash
curl http://localhost:5000/threads/<thread_id>/state
```

### Key Differences from Basic Chat

- Uses explicit `command` field in requests (`"start"` or `"resume"`)
- More structured API design with clear command separation
- Same underlying LangGraph functionality but different request format

## Interrupt in Chat V1

```bash
cd support_agent
python chat_with_interrupt_v1.py
```

The application will start on `http://localhost:5000` with debug mode enabled. 