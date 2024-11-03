# CSV Agent with Memory - FastAPI Implementation

## Overview

This repository contains a FastAPI implementation of a CSV agent that utilizes memory for chat interactions, powered by LangChain and Redis. The goal of this project is to provide a comprehensive starting point for developers looking to integrate CSV report querying capabilities into their applications via a well-structured API. 

## Pain Points Addressed

In the landscape of existing CSV agent implementations, many examples lack a cohesive API layer and fail to incorporate memory management, which are critical for creating responsive and context-aware applications. This project aims to bridge that gap by offering a robust solution that combines:

- **FastAPI Framework**: A modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **LangChain Integration**: Using the powerful capabilities of LangChain for natural language processing and interaction with CSV data.
- **Memory Management**: Leveraging Redis for persistent chat message history, enabling more meaningful interactions by retaining context across user sessions.

## Features

- **API Endpoint**: A simple and secure `/query` endpoint for querying CSV data with user-defined messages.
- **Session Management**: Unique session IDs for tracking user interactions and maintaining context through Redis.
- **Configurable Agent**: Ability to define the CSV URL and user message dynamically in requests, facilitating flexible data queries.
- **Secure API Key Handling**: Basic API key authentication to protect your endpoint from unauthorized access.

## Usage

This implementation serves as a starting point for developers to build upon. It can be easily integrated into applications that need to handle CSV reports, allowing users to interact with their data in a conversational manner. You can extend this codebase to include additional features such as:

- Enhanced authentication mechanisms.
- Improved error handling and logging.
- Custom logic for processing CSV data beyond simple queries.

## Getting Started

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/VARMASANGARAJU/langchain-csv-agent.git
   cd langchain-csv-agent
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama**: 
   Follow the official [Ollama installation guide](https://ollama.com/docs/getting-started/installation) to install Ollama on your machine.

4. **Pull the Gemma2 Model**:
   After installing Ollama, run the following command to pull the Gemma2 model:
   ```bash
   ollama pull gemma2
   ```

5. **Set up Redis using Docker**: 
   You can easily run Redis in a Docker container. Use the following command to pull and run Redis:
   ```bash
   docker run -d --name redis -p 6379:6379 redis
   ```
   This command starts a Redis server running on port 6379.

6. Set the necessary environment variables, including `REDIS_URL`, if you're using a non-default Redis setup:
   ```bash
   export REDIS_URL=redis://localhost:6379/0

7. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8065 --reload
   ```

8. Test the API using a tool like Postman or cURL.

## Contribution

Contributions are welcome! If you encounter issues, have suggestions for improvements, or want to add new features, feel free to open an issue or submit a pull request.
