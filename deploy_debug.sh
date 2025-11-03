#!/bin/bash
# Quick deployment script with debugging

echo "Building Docker image with debugging..."
docker build -t pythontrogon/ttii-chatbot:latest .

echo "Pushing to Docker Hub..."
docker push pythontrogon/ttii-chatbot:latest

echo "Deployment complete! Now test the PHP question again to see debug output."
