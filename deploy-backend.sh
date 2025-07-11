#!/bin/bash

# Deploy script for DeFi Q&A Backend to Fly.io
# This script sets up all necessary secrets, volumes, and deploys the application

set -e  # Exit on any error

echo "🚀 Starting DeFi Q&A Backend deployment to Fly.io..."

# Check if user is logged in
if ! fly auth whoami > /dev/null 2>&1; then
    echo "❌ You need to log in to Fly.io first"
    echo "Run: fly auth login"
    exit 1
fi

# Navigate to backend directory
cd backend

# Check if secrets are set
echo "🔑 Checking secrets..."
if ! fly secrets list | grep -q "OPENAI_API_KEY"; then
    echo "❌ OPENAI_API_KEY is not set"
    echo "Please set it with: fly secrets set OPENAI_API_KEY=your_actual_key"
    echo "You can get your OpenAI API key from: https://platform.openai.com/api-keys"
    exit 1
fi

if ! fly secrets list | grep -q "SECRET_KEY"; then
    echo "❌ SECRET_KEY is not set"
    echo "Please set it with: fly secrets set SECRET_KEY=your-secure-32-character-secret-key"
    echo "You can generate one with: openssl rand -hex 32"
    exit 1
fi

echo "✅ Required secrets are configured"

# Create volumes if they don't exist
echo "💾 Setting up volumes..."

if ! fly volumes list | grep -q "cache_volume"; then
    echo "Creating cache volume..."
    fly volumes create cache_volume --size 3 --region mia
fi

if ! fly volumes list | grep -q "logs_volume"; then
    echo "Creating logs volume..."
    fly volumes create logs_volume --size 2 --region mia
fi

echo "✅ Volumes are ready"

# Deploy the application
echo "🚀 Deploying backend application..."
fly deploy

echo "✅ Backend deployment completed!"
echo "🌐 Your backend should be available at: https://$(fly info --json | jq -r '.Hostname')"

# Show status
echo "📊 Application status:"
fly status 