#!/bin/bash

echo "🚀 Setting up DeFi Q&A Bot environment..."
echo

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    echo "   You can either:"
    echo "   1. Edit the existing .env file manually"
    echo "   2. Delete it and run this script again"
    echo "   3. Copy the template below and add your OpenAI API key"
    echo
    exit 1
fi

# Create .env file
cat > .env << 'EOF'
# Essential environment variables for DeFi Q&A Bot

# Required - OpenAI API key (GET THIS FROM https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
DEBUG=true
RELOAD=true
SECRET_KEY=your-secure-32-character-secret-key-here

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=200
RATE_LIMIT_WINDOW_SECONDS=60

# Agent Settings
AGENT_SIMILARITY_THRESHOLD=0.6
AGENT_MAX_RESULTS=3
AGENT_CACHE_ENABLED=true
AGENT_LLM_MODEL=gpt-4o-mini
AGENT_EMBEDDING_MODEL=text-embedding-3-small

# Cache
CACHE_DIR=./cache
CACHE_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
ENABLE_JSON_LOGS=true
ENABLE_CONSOLE_LOGS=true

# Performance
MAX_CONCURRENT_REQUESTS=10
MONITORING_ENABLED=true

# Security (for production)
USE_HTTPS=false
TRUST_PROXY=false
EOF

echo "✅ Created .env file with default configuration"
echo
echo "🔑 IMPORTANT: You need to set your OpenAI API key!"
echo "   1. Get your API key from: https://platform.openai.com/api-keys"
echo "   2. Edit the .env file and replace 'your_openai_api_key_here' with your actual key"
echo "   3. Also replace 'your-secure-32-character-secret-key-here' with a secure secret"
echo
echo "📝 To edit the file:"
echo "   nano .env    (or use your preferred editor)"
echo
echo "🚀 After setting your API key, restart your application!"

# Make the script executable
chmod +x setup-env.sh 