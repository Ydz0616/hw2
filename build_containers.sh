#!/bin/bash

# Script to build Docker containers for the song recommendation system

# Set environment variables
DOCKER_REPO=${DOCKER_REPO:-"yuandongzhang"}  # Updated with your Docker Hub namespace
IMAGE_TAG=${IMAGE_TAG:-"1.0.0"}

# Create directories if they don't exist
mkdir -p docker/ml docker/frontend

# Copy necessary files for ML container
echo "Copying files for ML container..."
cp song_recommender.py docker/ml/
cp docker/ml/train_model.py docker/ml/
cp docker/ml/Dockerfile docker/ml/
cp docker/ml/requirements.txt docker/ml/

# Copy necessary files for Frontend container
echo "Copying files for Frontend container..."
cp song_recommender.py docker/frontend/
cp app.py docker/frontend/
cp -r templates docker/frontend/
cp -r static docker/frontend/
cp docker/frontend/model_watcher.py docker/frontend/
cp docker/frontend/app.py docker/frontend/
cp docker/frontend/Dockerfile docker/frontend/
cp docker/frontend/requirements.txt docker/frontend/

# Build ML container
echo "Building ML container..."
docker build -t ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG} docker/ml/

# Build Frontend container
echo "Building Frontend container..."
docker build -t ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG} docker/frontend/

echo "Build completed!"
echo "ML Container: ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
echo "Frontend Container: ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}"
echo
echo "To push these containers to Docker Hub, run:"
echo "docker push ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
echo "docker push ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}" 