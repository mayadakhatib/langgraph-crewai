# LangGraph Chatbot

A conversational chatbot built with LangGraph that maintains conversation history and provides an interactive chat experience.

## Features

- 🤖 **Interactive Chat Interface**: Natural conversation flow with user input prompts
- 💾 **Conversation History**: Maintains context across the entire conversation
- 🔄 **LangGraph Workflow**: Uses LangGraph for robust conversation management
- 🎯 **Easy Exit**: Type 'quit', 'exit', or 'bye' to end the conversation
- 🛡️ **Error Handling**: Graceful handling of interruptions and errors

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   - Copy `env_example.txt` to `.env`
   - Replace `your_openai_api_key_here` with your actual OpenAI API key
   ```bash
   cp env_example.txt .env
   # Then edit .env with your API key
   ```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

### Example Conversation

```
🤖 Welcome to the LangGraph Chatbot!
Type 'quit', 'exit', or 'bye' to end the conversation.
--------------------------------------------------

You: Hello! How are you today?

Assistant: Hello! I'm doing well, thank you for asking! I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?

You: Can you tell me about LangGraph?

Assistant: LangGraph is a library for building stateful, multi-actor applications with LLMs. It's built on top of LangChain and provides a way to create complex workflows and applications that can maintain state across multiple interactions.

Key features of LangGraph include:
- State management across conversation turns
- Multi-actor systems
- Conditional logic and branching
- Integration with LangChain tools and models
- Support for both synchronous and asynchronous operations

It's particularly useful for building chatbots, agents, and other applications that need to maintain context and state over time.

You: That's helpful! Can you give me an example of how it works?

Assistant: Certainly! Let me show you a simple example of how LangGraph works. In our chatbot, we have a basic workflow with two main nodes:

1. **get_input**: This node prompts the user for input and handles the conversation flow
2. **process_message**: This node processes the user's message and generates a response

The workflow looks like this:
```
get_input → process_message → get_input → process_message → ...
```

Here's a simplified version of how it's structured:

```python
# Define the state
class ChatState(TypedDict):
    messages: List[BaseMessage]  # Conversation history
    user_input: str             # Current user input

# Create the graph
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("get_input", get_user_input)
workflow.add_node("process_message", process_message)

# Connect the nodes
workflow.add_edge("get_input", "process_message")
workflow.add_edge("process_message", "get_input")

# Set entry point and compile
workflow.set_entry_point("get_input")
chatbot = workflow.compile()
```

The key benefits are:
- **State Management**: The conversation history is automatically maintained
- **Modular Design**: Each function handles a specific part of the conversation
- **Extensibility**: Easy to add new nodes for different functionalities
- **Error Handling**: Built-in error handling and graceful exits

You: Thanks for the explanation! I think I understand it better now.

Assistant: You're very welcome! I'm glad I could help you understand LangGraph better. It's a powerful tool for building conversational AI applications, and the way it manages state and workflow makes it much easier to create complex, multi-turn interactions.

If you have any more questions about LangGraph, LangChain, or want to explore other features, feel free to ask! I'm here to help you learn and experiment with these technologies.

You: quit

Goodbye! 👋
```

## Project Structure

```
langgraph-crewai/
├── chatbot.py          # Main chatbot implementation
├── requirements.txt    # Python dependencies
├── env_example.txt     # Environment variables template
└── README.md          # This file
```

## How It Works

### State Management
The chatbot uses a `ChatState` TypedDict to maintain:
- **messages**: List of conversation messages (both user and AI)
- **user_input**: Current user input being processed

### LangGraph Workflow
1. **get_input**: Prompts user for input and handles exit commands
2. **process_message**: Processes the input, generates a response, and updates conversation history
3. **Loop**: The workflow continuously cycles between these nodes

### Key Components

- **StateGraph**: Manages the conversation flow and state
- **ChatOpenAI**: Handles AI response generation
- **Message Types**: Uses LangChain's `HumanMessage` and `AIMessage` for structured conversation

## Customization

You can easily customize the chatbot by:

1. **Changing the model**:
   ```python
   llm = ChatOpenAI(
       model="gpt-4",  # Change to different model
       temperature=0.5  # Adjust creativity
   )
   ```

2. **Adding new nodes** for specific functionalities
3. **Modifying the conversation flow** by changing the graph structure
4. **Adding tools** for external API calls or data processing

## Troubleshooting

- **API Key Error**: Make sure your `.env` file contains a valid OpenAI API key
- **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Model Errors**: Check that you have access to the specified OpenAI model

## License

This project is open source and available under the MIT License. 