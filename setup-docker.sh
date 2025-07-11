#!/bin/bash

# DeFi Q&A Bot Docker Setup Script
# This script helps you quickly set up and run the application with Docker

set -e  # Exit on any error

echo "🚀 DeFi Q&A Bot Docker Setup"
echo "============================="

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker is installed and running"

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f "docker.env.example" ]; then
        echo "📝 Creating .env file from template..."
        cp docker.env.example .env
        echo "⚠️  IMPORTANT: Please edit the .env file and add your OpenAI API key!"
        echo "   You can do this with: nano .env"
        read -p "   Press Enter after you've set your OPENAI_API_KEY..."
    else
        echo "❌ No environment template found. Please create a .env file manually."
        exit 1
    fi
else
    echo "✅ .env file found"
fi

# Validate that OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=.*[^=]" .env; then
    echo "⚠️  WARNING: OPENAI_API_KEY appears to be empty in .env file"
    echo "   The application may not work without a valid OpenAI API key."
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Please edit .env file and set your OPENAI_API_KEY"
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/cache backend/logs

# Set proper permissions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🔧 Setting directory permissions..."
    chmod 755 backend/cache backend/logs
fi

# Ask user about build preference
echo ""
echo "🔨 Build Options:"
echo "1. Quick start (use cached layers if available)"
echo "2. Fresh build (rebuild everything from scratch)"
read -p "Choose option (1-2): " -n 1 -r build_option
echo

case $build_option in
    2)
        echo "🧹 Cleaning Docker cache..."
        docker system prune -f
        FRESH_BUILD=true
        ;;
    *)
        FRESH_BUILD=false
        ;;
esac

# Ask about running mode
echo ""
echo "🎯 Running Mode:"
echo "1. Foreground (see logs in terminal)"
echo "2. Background (detached mode)"
read -p "Choose option (1-2): " -n 1 -r run_mode
echo

case $run_mode in
    2)
        RUN_ARGS="-d"
        ;;
    *)
        RUN_ARGS=""
        ;;
esac

# Build and start the application
echo ""
if [ "$FRESH_BUILD" = true ]; then
    echo "🏗️  Building application from scratch..."
    docker-compose build --no-cache
    echo "🚀 Starting the application..."
    docker-compose up $RUN_ARGS
else
    echo "🏗️  Building and starting the application..."
    docker-compose up --build $RUN_ARGS
fi

if [ "$run_mode" = "2" ]; then
    echo ""
    echo "🎉 Application is starting in the background!"
    echo ""
    echo "📊 Check status with: docker-compose ps"
    echo "📋 View logs with: docker-compose logs -f"
    echo ""
    echo "🌐 Once ready, access your application at:"
    echo "   Frontend:  http://localhost"
    echo "   Backend:   http://localhost:8000"
    echo "   API Docs:  http://localhost:8000/docs"
    echo ""
    echo "⏹️  Stop with: docker-compose down"
else
    echo ""
    echo "🎉 Application started!"
    echo "🌐 Access your application at:"
    echo "   Frontend:  http://localhost"
    echo "   Backend:   http://localhost:8000"
    echo "   API Docs:  http://localhost:8000/docs"
    echo ""
    echo "⏹️  Press Ctrl+C to stop the application"
fi 