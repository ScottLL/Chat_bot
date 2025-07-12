# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DeFi Q&A Bot - an AI-powered question-answering system specialized in Decentralized Finance topics. The system uses FastAPI backend with LangGraph orchestration, React frontend, and real-time WebSocket streaming for responses.

## Development Commands

### Backend (Python/FastAPI)
```bash
# Navigate to backend directory
cd backend

# Development server (with auto-reload)
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_api.py -v
python -m pytest tests/test_integration.py -v

# Run end-to-end tests
python -m pytest tests/e2e/ -v
```

### Frontend (React/TypeScript)
```bash
# Navigate to frontend directory  
cd frontend

# Development server
npm start

# Build for production
npm run build

# Run tests
npm test
```

### Docker Development
```bash
# Setup environment (creates .env file)
./setup-env.sh

# Build and run with Docker Compose (full stack)
docker-compose up --build -d

# Backend only
cd backend && docker build -t defi-qa-bot . && docker run -p 8000:8000 --env-file .env defi-qa-bot
```

### Deployment Scripts
```bash
# Setup deployment environment
./setup-deployment.sh

# Deploy backend (clean)
./deploy-backend-clean.sh

# Deploy backend 
./deploy-backend.sh
```

## Architecture

### Core Components

**Backend (`/backend/`)**:
- `main.py` - FastAPI application entry point with WebSocket support
- `agents/` - LangGraph agents for Q&A orchestration
  - `async_defi_qa_agent.py` - Main async agent with session isolation
  - `defi_qa_agent.py` - Synchronous agent implementation
- `services/` - Business logic services
  - `async_embedding_service.py` - OpenAI embeddings with async support
  - `session_manager.py` - User session management and isolation
  - `cache_manager.py` - Embedding cache management
  - `dataset_loader.py` - DeFi Q&A dataset loading
- `infrastructure/` - System infrastructure
  - `websocket_manager.py` - Real-time WebSocket connection management
  - `error_handlers.py` - Centralized error handling
  - `logging_config.py` - Structured logging configuration
  - `monitoring.py` - Metrics and health monitoring

**Frontend (`/frontend/`)**:
- React TypeScript application with WebSocket integration
- Real-time streaming UI for Q&A responses

### Key Technologies
- **LangGraph** - Agent orchestration and workflow management
- **FastAPI** - Async web framework with WebSocket support
- **OpenAI** - GPT models and text embeddings (text-embedding-3-small)
- **WebSockets** - Real-time bidirectional communication
- **Session Management** - User isolation and concurrent request handling

### Configuration
- Environment variables managed through `config.py`
- Template file: `backend/environment_template.txt`
- Setup script: `setup-env.sh` creates `.env` with defaults

### Testing
- Pytest configuration in `backend/pytest.ini`
- Test structure: `backend/tests/` with unit, integration, and e2e tests
- Async test support with `pytest-asyncio`

### Cache System
- Embedding cache in `cache/` and `backend/cache/` directories  
- Cache manager handles embedding storage and retrieval
- Significant performance improvement for repeated questions

### Deployment Support
- Docker configurations for both backend and frontend
- Docker Compose for full-stack deployment
- Cloud platform configs: Fly.io (`fly.toml`), Railway (`railway.toml`), Render (`render.yaml`)
- Heroku support via `Procfile`

### Important Notes
- **OpenAI API Key Required**: Must be configured in `.env` file
- **Session Isolation**: Each user gets isolated session context  
- **Concurrent Support**: Handles multiple simultaneous WebSocket connections
- **Error Resilience**: Comprehensive error handling with user-friendly messages
- **Real-time Streaming**: Word-by-word response streaming via WebSocket