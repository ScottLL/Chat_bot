#!/bin/bash

# Setup script for GitHub Actions deployment to Fly.io
echo "🚀 Setting up GitHub Actions deployment to Fly.io"
echo "================================================="

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed."
    echo "Please install it first: https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

echo "✅ flyctl is installed"

# Check if user is logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ You are not logged in to Fly.io"
    echo "Please run: flyctl auth login"
    exit 1
fi

echo "✅ You are logged in to Fly.io"

# Get API token
echo ""
echo "📋 Getting your Fly.io API token..."
API_TOKEN=$(flyctl auth token)

if [ -z "$API_TOKEN" ]; then
    echo "❌ Failed to get API token"
    exit 1
fi

echo "✅ API Token retrieved"
echo ""
echo "🔐 Add this token as a GitHub secret:"
echo "1. Go to your repository → Settings → Secrets and variables → Actions"
echo "2. Click 'New repository secret'"
echo "3. Name: FLY_API_TOKEN"
echo "4. Value: $API_TOKEN"
echo ""

# Check for existing Fly.io apps
echo "🔍 Checking for existing Fly.io apps..."

APPS=("defi-qa-backend" "defi-qa-frontend")
MISSING_APPS=()

for app in "${APPS[@]}"; do
    if flyctl apps list | grep -q "$app"; then
        echo "✅ App '$app' exists"
    else
        echo "❌ App '$app' does not exist"
        MISSING_APPS+=("$app")
    fi
done

if [ ${#MISSING_APPS[@]} -gt 0 ]; then
    echo ""
    echo "📝 Create missing apps:"
    for app in "${MISSING_APPS[@]}"; do
        echo "flyctl apps create $app"
    done
else
    echo "✅ All required apps exist"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Add the FLY_API_TOKEN secret to GitHub (instructions above)"
echo "2. Create any missing Fly.io apps"
echo "3. Push changes to the 'deploy' branch to trigger deployment"
echo ""
echo "📚 For detailed documentation, see: .github/DEPLOYMENT.md"
echo ""
echo "🎉 Setup complete! Your repository is ready for automated deployment." 