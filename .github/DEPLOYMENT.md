# GitHub Actions Deployment to Fly.io

This repository is configured with multiple GitHub Actions workflows for automated deployment to Fly.io.

## Setup

### 1. Fly.io API Token

1. Get your Fly.io API token:
   ```bash
   flyctl auth token
   ```

2. Add the token as a secret in your GitHub repository:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `FLY_API_TOKEN`
   - Value: Your Fly.io API token

### 2. Fly.io Apps

Make sure you have created the following Fly.io apps:
- `defi-qa-chatbot` (combined app)
- `defi-qa-backend` (backend only)
- `defi-qa-frontend` (frontend only)

You can create them using:
```bash
flyctl apps create defi-qa-chatbot
flyctl apps create defi-qa-backend
flyctl apps create defi-qa-frontend
```

## Available Workflows

### 1. Combined Deployment (`deploy-combined.yml`)
- **Triggers**: Push to `deploy` branch, manual dispatch
- **Description**: Deploys the backend API service as the main application
- **Fly.io App**: `defi-qa-chatbot`
- **Configuration**: Uses root `fly.toml` (backend-focused)
- **Note**: This deploys the API service. For full-stack deployment, use individual backend + frontend deployments

### 2. Backend Only (`deploy-backend.yml`)
- **Triggers**: Push to `deploy` branch (backend changes only), manual dispatch
- **Description**: Deploys only the backend service
- **Fly.io App**: `defi-qa-backend`
- **Configuration**: Uses `backend/fly.toml`

### 3. Frontend Only (`deploy-frontend.yml`)
- **Triggers**: Push to `deploy` branch (frontend changes only), manual dispatch
- **Description**: Deploys only the frontend service
- **Fly.io App**: `defi-qa-frontend`
- **Configuration**: Uses `frontend/fly.toml`

### 4. Deploy with Tests (`deploy-with-tests.yml`)
- **Triggers**: Push to `deploy` branch, manual dispatch
- **Description**: Runs full test suite before deployment and includes health checks
- **Features**:
  - Python backend tests with pytest
  - Node.js frontend tests
  - Health check after deployment
  - Only deploys if all tests pass

## Usage

### Automatic Deployment
1. Push changes to the `deploy` branch:
   ```bash
   git checkout deploy
   git merge main  # or cherry-pick specific commits
   git push origin deploy
   ```

2. GitHub Actions will automatically:
   - Run the appropriate workflow based on which files changed
   - Deploy to Fly.io
   - Run health checks (if using the comprehensive workflow)

### Manual Deployment
1. Go to your repository → Actions
2. Select the workflow you want to run
3. Click "Run workflow"
4. Choose the `deploy` branch
5. Click "Run workflow"

## Workflow Selection Strategy

- **Use Combined Deployment** for full application releases
- **Use Individual Deployments** for service-specific updates
- **Use Deploy with Tests** for production releases where testing is critical

## Monitoring

After deployment, you can monitor your applications:
- Combined app: https://defi-qa-chatbot.fly.dev
- Backend: https://defi-qa-backend.fly.dev
- Frontend: https://defi-qa-frontend.fly.dev

Check deployment status:
```bash
flyctl status -a defi-qa-chatbot
flyctl logs -a defi-qa-chatbot
```

## Troubleshooting

### Common Issues

1. **Deployment fails with authentication error**
   - Check that `FLY_API_TOKEN` secret is correctly set
   - Verify the token is valid: `flyctl auth whoami`

2. **App not found error**
   - Ensure the Fly.io apps are created
   - Check app names in fly.toml files match your actual apps

3. **Build fails**
   - Check Dockerfile syntax
   - Verify all dependencies are listed in requirements.txt/package.json

4. **Health check fails**
   - Ensure your app responds to `/health` endpoint
   - Check if the app is properly starting

### Debug Commands

```bash
# Check app status
flyctl status -a your-app-name

# View logs
flyctl logs -a your-app-name

# Connect to app console
flyctl ssh console -a your-app-name

# Check deployment history
flyctl releases -a your-app-name
``` 