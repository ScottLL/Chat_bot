# Environment Configuration for DeFi Q&A Bot

This document outlines all environment variables needed for deploying the DeFi Q&A Bot to cloud platforms.

## Required Environment Variables

### API Keys (REQUIRED)
```bash
# OpenAI API key for embeddings and LLM
OPENAI_API_KEY=your_openai_api_key_here
```

### Server Configuration
```bash
# Server host and port
HOST=0.0.0.0
PORT=8000

# Environment: development, staging, production
ENVIRONMENT=production

# Debug mode (true/false)
DEBUG=false

# Auto-reload for development (true/false) - should be false for production
RELOAD=false

# Secret key for sessions and security (minimum 32 characters)
SECRET_KEY=your_secret_key_here_minimum_32_characters_long
```

### CORS Configuration
```bash
# Comma-separated list of allowed origins for CORS
# Replace with your actual frontend URLs for production
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Rate Limiting
```bash
# Rate limiting configuration
RATE_LIMIT_REQUESTS_PER_MINUTE=200
RATE_LIMIT_WINDOW_SECONDS=60
```

### Agent Configuration
```bash
# Similarity threshold for semantic search (0.0-1.0)
AGENT_SIMILARITY_THRESHOLD=0.6

# Maximum number of results to return
AGENT_MAX_RESULTS=3

# Enable embedding cache (true/false)
AGENT_CACHE_ENABLED=true

# AI Models
AGENT_LLM_MODEL=gpt-4o-mini
AGENT_EMBEDDING_MODEL=text-embedding-3-small
```

### Cache Configuration
```bash
# Cache directory path
CACHE_DIR=../cache

# Cache settings
CACHE_ENABLED=true
CACHE_MAX_SIZE=1000
CACHE_EXPIRY_DAYS=30
```

### Logging Configuration
```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log directory
LOG_DIR=logs

# Enable JSON formatted logs (true/false)
ENABLE_JSON_LOGS=true

# Enable console logging (true/false)
ENABLE_CONSOLE_LOGS=true

# Log file rotation
LOG_MAX_FILE_SIZE=100MB
LOG_RETENTION_DAYS=30
LOG_COMPRESSION=gz
```

### Performance & Monitoring
```bash
# Maximum concurrent requests
MAX_CONCURRENT_REQUESTS=10

# Session management
SESSION_CLEANUP_INTERVAL=300
MAX_SESSION_HISTORY=10

# Monitoring settings
MONITORING_ENABLED=true
METRICS_ENABLED=true

# Background task intervals (seconds)
PERFORMANCE_MONITOR_INTERVAL=30
WEBSOCKET_CLEANUP_INTERVAL=60
```

### Security
```bash
# HTTPS settings for production
USE_HTTPS=true
SSL_CERT_PATH=/path/to/ssl/cert.pem
SSL_KEY_PATH=/path/to/ssl/key.pem

# Trust proxy headers (for load balancers)
TRUST_PROXY=true
```

## Cloud Platform Specific Variables

### Heroku
```bash
WEB_CONCURRENCY=4
```

### Railway/Render
```bash
RAILWAY_STATIC_URL=https://your-app.railway.app
RENDER_EXTERNAL_URL=https://your-app.onrender.com
```

### AWS/GCP/Azure
```bash
CLOUD_REGION=us-east-1
CLOUD_STORAGE_BUCKET=your-storage-bucket
```

## Future Extensions (Optional)

### Database Configuration
```bash
# Database URL (if needed for future features)
DATABASE_URL=postgresql://user:password@localhost:5432/defi_qa

# Redis URL (if needed for caching/sessions)
REDIS_URL=redis://localhost:6379/0
```

## How to Set Environment Variables

### 1. Create .env file locally
Create a `.env` file in the backend directory with your actual values:
```bash
cp environment_config_template.txt .env
# Edit .env with your actual values
```

### 2. Cloud Platform Configuration

#### Heroku
```bash
heroku config:set OPENAI_API_KEY=your_key_here
heroku config:set ENVIRONMENT=production
# ... set all other variables
```

#### Railway
```bash
railway variables set OPENAI_API_KEY=your_key_here
railway variables set ENVIRONMENT=production
# ... set all other variables
```

#### Render
Set environment variables in the Render dashboard under "Environment" tab.

#### Vercel
```bash
vercel env add OPENAI_API_KEY
vercel env add ENVIRONMENT
# ... add all other variables
```

## Security Best Practices

1. **Never commit .env files** - Always use .env.example or documentation
2. **Use strong secret keys** - Generate random 32+ character strings
3. **Rotate API keys regularly** - Especially for production environments
4. **Use different keys for different environments** - dev/staging/production
5. **Enable HTTPS in production** - Set USE_HTTPS=true
6. **Restrict CORS origins** - Don't use wildcard (*) in production
7. **Monitor rate limits** - Adjust based on your traffic patterns

## Environment-Specific Recommendations

### Development
- DEBUG=true
- RELOAD=true
- LOG_LEVEL=DEBUG
- ALLOWED_ORIGINS=*

### Staging
- DEBUG=false
- RELOAD=false
- LOG_LEVEL=INFO
- ALLOWED_ORIGINS=staging.yourdomain.com

### Production
- DEBUG=false
- RELOAD=false
- LOG_LEVEL=WARNING
- ALLOWED_ORIGINS=yourdomain.com,www.yourdomain.com
- USE_HTTPS=true
- Higher rate limits if needed 