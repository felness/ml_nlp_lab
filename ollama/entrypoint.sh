#!/bin/bash
set -e

# Start Ollama server in background
ollama serve &
SERVER_PID=$!

# Wait for the server to become ready
echo "Waiting for Ollama server..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

# Pull the model if not already cached
echo "Pulling qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b
echo "Model ready."

# Keep server running
wait $SERVER_PID
