#!/usr/bin/env python3
"""
Startup script for TTII Chatbot API
This script helps debug startup issues and ensures proper initialization
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set"""
    logger.info("Checking environment variables...")
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            logger.warning(f"Missing environment variable: {var}")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("All required environment variables are set")
    return True

def check_dependencies():
    """Check if all required packages are available"""
    logger.info("Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import openai
        import pydantic
        import dotenv
        logger.info("Core dependencies loaded successfully")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("Starting TTII Chatbot API...")
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    logger.info("All checks passed, starting uvicorn server...")
    
    # Import and run the app
    try:
        from main import app
        import uvicorn
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 