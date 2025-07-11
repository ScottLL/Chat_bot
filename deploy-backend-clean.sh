#!/bin/bash

# Clean deployment script for DeFi Q&A Backend to Fly.io
set -e  # Exit on any error

echo "🚀 Starting clean deployment of DeFi Q&A Backend to Fly.io..."

# Navigate to backend directory
cd backend

# Check if user is logged in
if ! fly auth whoami > /dev/null 2>&1; then
    echo "❌ You need to log in to Fly.io first"
    echo "Run: fly auth login"
    exit 1
fi

echo "✅ Logged in to Fly.io"

# Check if app exists and clean up if needed
echo "🧹 Checking for existing app..."
if fly apps list | grep -q "defi-qa-backend"; then
    echo "⚠️  Found existing app. Destroying it to start clean..."
    echo "Please type 'defi-qa-backend' when prompted to confirm destruction:"
    fly apps destroy defi-qa-backend
fi

# Create new app
echo "📱 Creating new app..."
fly apps create defi-qa-backend

# Set up secrets
echo "🔑 Setting up secrets..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Please set OPENAI_API_KEY environment variable"
    echo "Example: export OPENAI_API_KEY=sk-proj-your-key-here"
    exit 1
fi

fly secrets set OPENAI_API_KEY="$OPENAI_API_KEY"

# Generate and set secret key
SECRET_KEY=$(openssl rand -hex 32)
fly secrets set SECRET_KEY="$SECRET_KEY"
echo "✅ Generated and set SECRET_KEY: $SECRET_KEY"

# Create single volume
echo "💾 Creating volume..."
fly volumes create app_data --region mia --size 3

# Deploy the application
echo "🚀 Deploying application..."
fly deploy

echo "✅ Deployment completed!"
echo "🌐 Your app should be available at: https://defi-qa-backend.fly.dev"
echo "📊 Monitor your app at: https://fly.io/apps/defi-qa-backend/monitoring" 