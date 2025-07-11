"""
Configuration module for DeFi Q&A Bot.

This module handles all environment variable configuration for development,
staging, and production environments.
"""

import os
import secrets
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Application configuration class that reads environment variables
    and provides defaults for development/production environments.
    """
    
    # ========================================
    # CORE API CONFIGURATION
    # ========================================
    # Required OpenAI API key
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    # ========================================
    # SERVER CONFIGURATION
    # ========================================
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '8000'))
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')
    DEBUG: bool = os.getenv('DEBUG', 'true').lower() == 'true'
    RELOAD: bool = os.getenv('RELOAD', 'true').lower() == 'true'
    
    # Generate a secure secret key if not provided
    SECRET_KEY: str = os.getenv('SECRET_KEY', secrets.token_urlsafe(32))
    
    # ========================================
    # CORS CONFIGURATION
    # ========================================
    ALLOWED_ORIGINS_STR: str = os.getenv(
        'ALLOWED_ORIGINS', 
        'http://localhost:3000,http://127.0.0.1:3000'
    )
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.ENVIRONMENT == 'development':
            # Allow all origins in development
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(',') if origin.strip()]
    
    # ========================================
    # RATE LIMITING
    # ========================================
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '200'))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv('RATE_LIMIT_WINDOW_SECONDS', '60'))
    
    # ========================================
    # AGENT CONFIGURATION
    # ========================================
    AGENT_SIMILARITY_THRESHOLD: float = float(os.getenv('AGENT_SIMILARITY_THRESHOLD', '0.6'))
    AGENT_MAX_RESULTS: int = int(os.getenv('AGENT_MAX_RESULTS', '3'))
    AGENT_CACHE_ENABLED: bool = os.getenv('AGENT_CACHE_ENABLED', 'true').lower() == 'true'
    AGENT_LLM_MODEL: str = os.getenv('AGENT_LLM_MODEL', 'gpt-4o-mini')
    AGENT_EMBEDDING_MODEL: str = os.getenv('AGENT_EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # ========================================
    # CACHE CONFIGURATION
    # ========================================
    CACHE_DIR: str = os.getenv('CACHE_DIR', './cache')  # Use local cache directory instead of ../cache
    CACHE_ENABLED: bool = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_MAX_SIZE: int = int(os.getenv('CACHE_MAX_SIZE', '1000'))
    CACHE_EXPIRY_DAYS: int = int(os.getenv('CACHE_EXPIRY_DAYS', '30'))
    
    # ========================================
    # LOGGING CONFIGURATION
    # ========================================
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR: str = os.getenv('LOG_DIR', 'logs')
    ENABLE_JSON_LOGS: bool = os.getenv('ENABLE_JSON_LOGS', 'true').lower() == 'true'
    ENABLE_CONSOLE_LOGS: bool = os.getenv('ENABLE_CONSOLE_LOGS', 'true').lower() == 'true'
    LOG_MAX_FILE_SIZE: str = os.getenv('LOG_MAX_FILE_SIZE', '100MB')
    LOG_RETENTION_DAYS: str = os.getenv('LOG_RETENTION_DAYS', '30')
    LOG_COMPRESSION: str = os.getenv('LOG_COMPRESSION', 'gz')
    
    # ========================================
    # PERFORMANCE & MONITORING
    # ========================================
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    SESSION_CLEANUP_INTERVAL: int = int(os.getenv('SESSION_CLEANUP_INTERVAL', '300'))
    MAX_SESSION_HISTORY: int = int(os.getenv('MAX_SESSION_HISTORY', '10'))
    MONITORING_ENABLED: bool = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
    METRICS_ENABLED: bool = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    PERFORMANCE_MONITOR_INTERVAL: int = int(os.getenv('PERFORMANCE_MONITOR_INTERVAL', '30'))
    WEBSOCKET_CLEANUP_INTERVAL: int = int(os.getenv('WEBSOCKET_CLEANUP_INTERVAL', '60'))
    
    # ========================================
    # SECURITY CONFIGURATION
    # ========================================
    USE_HTTPS: bool = os.getenv('USE_HTTPS', 'false').lower() == 'true'
    SSL_CERT_PATH: Optional[str] = os.getenv('SSL_CERT_PATH')
    SSL_KEY_PATH: Optional[str] = os.getenv('SSL_KEY_PATH')
    TRUST_PROXY: bool = os.getenv('TRUST_PROXY', 'false').lower() == 'true'
    
    # ========================================
    # CLOUD PLATFORM SPECIFIC
    # ========================================
    WEB_CONCURRENCY: Optional[int] = int(os.getenv('WEB_CONCURRENCY', '1')) if os.getenv('WEB_CONCURRENCY') else None
    RAILWAY_STATIC_URL: Optional[str] = os.getenv('RAILWAY_STATIC_URL')
    RENDER_EXTERNAL_URL: Optional[str] = os.getenv('RENDER_EXTERNAL_URL')
    CLOUD_REGION: Optional[str] = os.getenv('CLOUD_REGION')
    CLOUD_STORAGE_BUCKET: Optional[str] = os.getenv('CLOUD_STORAGE_BUCKET')
    
    # ========================================
    # VALIDATION & UTILITIES
    # ========================================
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """
        Validate the configuration and return a list of errors.
        
        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        errors = []
        
        # Required fields validation
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # Port validation
        if not 1 <= cls.PORT <= 65535:
            errors.append(f"PORT must be between 1-65535, got {cls.PORT}")
        
        # Environment validation
        valid_environments = ['development', 'staging', 'production']
        if cls.ENVIRONMENT not in valid_environments:
            errors.append(f"ENVIRONMENT must be one of {valid_environments}, got {cls.ENVIRONMENT}")
        
        # Similarity threshold validation
        if not 0.0 <= cls.AGENT_SIMILARITY_THRESHOLD <= 1.0:
            errors.append(f"AGENT_SIMILARITY_THRESHOLD must be between 0.0-1.0, got {cls.AGENT_SIMILARITY_THRESHOLD}")
        
        # Secret key validation
        if len(cls.SECRET_KEY) < 32:
            errors.append("SECRET_KEY must be at least 32 characters long")
        
        # HTTPS validation
        if cls.USE_HTTPS:
            if not cls.SSL_CERT_PATH:
                errors.append("SSL_CERT_PATH is required when USE_HTTPS=true")
            if not cls.SSL_KEY_PATH:
                errors.append("SSL_KEY_PATH is required when USE_HTTPS=true")
        
        return errors
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of the current configuration (without sensitive data).
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "environment": cls.ENVIRONMENT,
            "debug": cls.DEBUG,
            "host": cls.HOST,
            "port": cls.PORT,
            "allowed_origins_count": len(cls().ALLOWED_ORIGINS),
            "rate_limit_rpm": cls.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "agent_similarity_threshold": cls.AGENT_SIMILARITY_THRESHOLD,
            "agent_max_results": cls.AGENT_MAX_RESULTS,
            "cache_enabled": cls.CACHE_ENABLED,
            "monitoring_enabled": cls.MONITORING_ENABLED,
            "use_https": cls.USE_HTTPS,
            "log_level": cls.LOG_LEVEL,
            "openai_api_key_configured": bool(cls.OPENAI_API_KEY),
            "cloud_platform": cls._detect_cloud_platform()
        }
    
    @classmethod
    def _detect_cloud_platform(cls) -> Optional[str]:
        """Detect which cloud platform we're running on."""
        if os.getenv('DYNO'):  # Heroku
            return 'heroku'
        elif os.getenv('RAILWAY_ENVIRONMENT'):  # Railway
            return 'railway'
        elif os.getenv('RENDER'):  # Render
            return 'render'
        elif os.getenv('VERCEL'):  # Vercel
            return 'vercel'
        elif os.getenv('AWS_LAMBDA_FUNCTION_NAME'):  # AWS Lambda
            return 'aws_lambda'
        elif os.getenv('GOOGLE_CLOUD_PROJECT'):  # Google Cloud
            return 'google_cloud'
        elif os.getenv('AZURE_CLIENT_ID'):  # Azure
            return 'azure'
        return None
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if we're running in production environment."""
        return cls.ENVIRONMENT == 'production'
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if we're running in development environment."""
        return cls.ENVIRONMENT == 'development'


# Create a global config instance
config = Config()

# Validate configuration on import
config_errors = config.validate_config()
if config_errors:
    print("❌ Configuration Errors:")
    for error in config_errors:
        print(f"  - {error}")
    print("\nPlease check your environment variables and try again.")
    # Don't exit in development, but log the errors
    if config.ENVIRONMENT == 'production':
        exit(1)
else:
    print("✅ Configuration validated successfully")
    if config.DEBUG:
        print(f"📋 Config Summary: {config.get_config_summary()}") 