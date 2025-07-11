# Fly.io Deployment Guide

This guide provides instructions for deploying your DeFi Q&A ChatBot application to Fly.io using two different approaches.

## Prerequisites

1. Install the Fly.io CLI:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. Authenticate with Fly.io:
   ```bash
   fly auth login
   ```

3. Make sure you have a payment method configured (required for deployment).

## Deployment Options

### Option 1: Single App with Docker Compose (Recommended for Development)

This approach uses Fly.io's new Docker Compose compatibility feature to deploy both services as a single app.

#### Steps:

1. **Use the root `fly.toml` file** (already created) which references your existing `docker-compose.yml`.

2. **Launch the application:**
   ```bash
   fly launch --no-deploy
   ```

3. **Create necessary volumes:**
   ```bash
   fly volumes create data_volume --size 10 --region mia
   ```

4. **Set environment variables/secrets:**
   ```bash
   # Add your environment variables
   fly secrets set OPENAI_API_KEY=your_openai_key_here
   fly secrets set ANTHROPIC_API_KEY=your_anthropic_key_here
   # Add other secrets as needed
   ```

5. **Deploy the application:**
   ```bash
   fly deploy
   ```

### Option 2: Separate Apps (Recommended for Production)

This approach deploys the backend and frontend as separate applications, providing better isolation and scaling capabilities.

#### Backend Deployment:

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Launch the backend app:**
   ```bash
   fly launch --no-deploy
   ```

3. **Create volumes for persistent data:**
   ```bash
   fly volumes create data_volume --size 5 --region mia
   fly volumes create cache_volume --size 3 --region mia
   fly volumes create logs_volume --size 2 --region mia
   ```

4. **Set backend secrets:**
   ```bash
   fly secrets set OPENAI_API_KEY=your_openai_key_here
   fly secrets set ANTHROPIC_API_KEY=your_anthropic_key_here
   fly secrets set ENVIRONMENT=production
   # Add other API keys and secrets as needed
   ```

5. **Deploy the backend:**
   ```bash
   fly deploy
   ```

6. **Note the backend URL** (will be something like `https://defi-qa-backend.fly.dev`)

#### Frontend Deployment:

1. **Navigate to the frontend directory:**
   ```bash
   cd ../frontend
   ```

2. **Update the API URL in the frontend fly.toml** if your backend URL is different:
   ```toml
   [env]
     REACT_APP_API_URL = "https://your-actual-backend-url.fly.dev"
   ```

3. **Launch the frontend app:**
   ```bash
   fly launch --no-deploy
   ```

4. **Deploy the frontend:**
   ```bash
   fly deploy
   ```

## Configuration Details

### Key Features in the Fly.toml Files:

- **Health Checks**: Automated health monitoring for both services
- **Auto-scaling**: Automatic start/stop of machines based on traffic
- **HTTPS**: Force HTTPS redirection for security
- **Volumes**: Persistent storage for data, cache, and logs
- **Resource Allocation**: Optimized CPU and memory settings

### Environment Variables:

Make sure to set these environment variables using `fly secrets set`:

**Backend:**
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY` 
- `ENVIRONMENT=production`
- Any database URLs or other API keys

**Frontend:**
- `REACT_APP_API_URL` (should point to your backend URL)

## Monitoring and Management

### Useful Commands:

```bash
# Check application status
fly status

# View logs
fly logs

# Scale your application
fly scale count 2

# Scale VM resources
fly scale vm shared-cpu-2x
fly scale memory 2048

# SSH into your application
fly ssh console

# List your applications
fly apps list

# Get application information
fly info
```

### Health Checks:

Both configurations include health checks:
- **Backend**: `GET /health` endpoint
- **Frontend**: `GET /` endpoint

### Scaling:

You can scale your applications based on traffic:

```bash
# Scale backend to 2 instances
fly scale count 2 --app defi-qa-backend

# Scale frontend to 3 instances
fly scale count 3 --app defi-qa-frontend
```

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check your Dockerfile and dependencies
2. **Health Check Failures**: Ensure your application responds correctly on the specified paths
3. **Volume Mount Issues**: Make sure volumes are created before deployment
4. **Environment Variables**: Use `fly secrets list` to verify your secrets are set

### Debug Commands:

```bash
# Check machine status
fly machine status

# View detailed logs
fly logs --app your-app-name

# Connect to your application
fly ssh console --app your-app-name
```

## Cost Optimization

### Tips to Minimize Costs:

1. **Use auto-stop/auto-start**: Configured in the fly.toml files
2. **Right-size your VMs**: Start with smaller instances and scale up if needed
3. **Use shared CPU instances**: More cost-effective for most applications
4. **Monitor usage**: Check the Fly.io dashboard regularly

### Estimated Costs:

- **Backend** (shared-cpu-1x, 1GB RAM): ~$5-10/month
- **Frontend** (shared-cpu-1x, 512MB RAM): ~$3-5/month
- **Volumes**: ~$0.15/GB/month

## Next Steps

1. **Custom Domain**: Configure a custom domain for your application
2. **CDN**: Set up Cloudflare or similar for better performance
3. **Monitoring**: Implement application monitoring and alerting
4. **Backup**: Set up regular backups for your volumes
5. **CI/CD**: Integrate with GitHub Actions for automated deployments

## Support

- [Fly.io Documentation](https://fly.io/docs/)
- [Fly.io Community](https://community.fly.io/)
- [Fly.io Discord](https://fly.io/discord) 