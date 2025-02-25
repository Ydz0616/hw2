#!/bin/bash
# Script to build Docker containers for the song recommendation system

# Set environment variables
DOCKER_REPO="yuandongzhang"
IMAGE_TAG="1.0.0"

# Create directories if they don't exist
mkdir -p ../data

# Copy song_recommender.py to both containers
echo "Copying song_recommender.py to containers..."
cp ../song_recommender.py ml/
cp ../song_recommender.py frontend/

# Copy static and templates to frontend
echo "Copying static and templates to frontend container..."
cp -r ../static frontend/
cp -r ../templates frontend/

# Build ML container
echo "Building ML container..."
docker build -t ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG} ml/

# Build Frontend container
echo "Building Frontend container..."
docker build -t ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG} frontend/

echo "Build completed!"
echo "ML image: ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
echo "Frontend image: ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}"
echo ""
echo "To push images to Docker Hub:"
echo "docker login"
echo "docker push ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
echo "docker push ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}" 