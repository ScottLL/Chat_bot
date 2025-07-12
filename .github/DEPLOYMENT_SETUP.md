# GitHub Actions Deployment Setup for Fly.io

This guide explains how to set up automatic deployment to Fly.io using GitHub Actions.

## Prerequisites

1. **Fly.io Account**: You need accounts for both backend and frontend apps
2. **GitHub Repository**: This repository with the workflow file
3. **Fly.io Apps**: Both `defi-qa-backend` and `defi-qa-frontend` apps created on Fly.io

## Setup Steps

### 1. Create Fly.io Apps (if not already created)

```bash
# Backend
cd backend
flyctl apps create defi-qa-backend

# Frontend  
cd frontend
flyctl apps create defi-qa-frontend
```

### 2. Get Fly.io API Token

```bash
# Generate a new token
flyctl auth token

# Copy the token - you'll need it for GitHub secrets
```

### 3. Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these repository secrets:

| Secret Name | Description | How to get |
|-------------|-------------|------------|
| `FLY_API_TOKEN` | Fly.io API token for deployments | Run `flyctl auth token` |
| `OPENAI_API_KEY` | OpenAI API key for the backend | Get from https://platform.openai.com/api-keys |

### 4. Update Fly.io App Names (if different)

If your Fly.io app names are different from `defi-qa-backend` and `defi-qa-frontend`, update:

1. **Backend fly.toml**: Update `app = "your-backend-app-name"`
2. **Frontend fly.toml**: Update `app = "your-frontend-app-name"`
3. **Workflow file**: Update health check URLs in `.github/workflows/deploy-fly.yml`
4. **Frontend environment**: Update `REACT_APP_API_URL` in `frontend/fly.toml`

### 5. Create Deploy Branch

```bash
# Create and push deploy branch
git checkout -b deploy
git push -u origin deploy
```

## How It Works

### Trigger Deployment

The GitHub Action will automatically deploy when:
- Code is pushed to the `deploy` branch
- Manual trigger via GitHub Actions UI (workflow_dispatch)

### Deployment Flow

1. **Backend Deployment**: Deploys the FastAPI backend first
2. **Frontend Deployment**: Deploys the React frontend after backend succeeds
3. **Health Checks**: Verifies both services are running correctly

### Environment Variables

The backend will automatically receive:
- All environment variables defined in `backend/fly.toml`
- `OPENAI_API_KEY` from GitHub secrets
- Production-optimized settings

## Deployment Commands

### Manual Deployment (if needed)

```bash
# Backend only
flyctl deploy --config backend/fly.toml --dockerfile backend/Dockerfile

# Frontend only
flyctl deploy --config frontend/fly.toml --dockerfile frontend/Dockerfile
```

### Monitor Deployments

```bash
# Check deployment status
flyctl status --app defi-qa-backend
flyctl status --app defi-qa-frontend

# View logs
flyctl logs --app defi-qa-backend
flyctl logs --app defi-qa-frontend
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify `FLY_API_TOKEN` is correct
2. **App Not Found**: Ensure app names match in fly.toml files
3. **Build Failures**: Check Dockerfile paths and build context
4. **Health Check Failures**: Verify endpoints are accessible

### Debug Steps

1. Check GitHub Actions logs for specific error messages
2. Verify Fly.io app configurations
3. Test manual deployment locally
4. Check Fly.io app status and logs

## Security Notes

- Never commit API keys or tokens to the repository
- Use GitHub secrets for sensitive environment variables
- Regularly rotate Fly.io API tokens
- Monitor deployment logs for any security issues

## App URLs (Update with your actual URLs)

- **Backend**: https://defi-qa-backend.fly.dev
- **Frontend**: https://defi-qa-frontend.fly.dev
- **Backend Health**: https://defi-qa-backend.fly.dev/health
- **Backend API Docs**: https://defi-qa-backend.fly.dev/docs