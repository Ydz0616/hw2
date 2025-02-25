#!/bin/bash
# Script to test Docker containers locally

# Set environment variables
DOCKER_REPO="yuandongzhang"
IMAGE_TAG="1.0.0"
ML_IMAGE="${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
FRONTEND_IMAGE="${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}"
CONTAINER_NAME="song-recommender-frontend"

# Create test directories
echo "Creating test directories..."
mkdir -p test_shared/models
mkdir -p test_data
chmod -R 777 test_shared

# Copy dataset files to test_data
echo "Copying dataset files..."
cp ../data/2023_spotify_songs.csv test_data/ 2>/dev/null || echo "Warning: 2023_spotify_songs.csv not found"
cp ../data/2023_spotify_ds2.csv test_data/ 2>/dev/null || echo "Warning: 2023_spotify_ds2.csv not found"

# Run ML container to train the model
echo "Running ML container to train the model..."
docker run --rm \
  -v "$(pwd)/test_data:/data" \
  -v "$(pwd)/test_shared:/shared" \
  -e MIN_SUPPORT=0.002 \
  -e MIN_CONFIDENCE=0.01 \
  -e MAX_PLAYLISTS=1000 \
  -e MAX_SONGS_PER_PLAYLIST=30 \
  ${ML_IMAGE}

# Check if model was created
if [ -f "test_shared/models/song_recommender.pkl" ]; then
  echo "Model created successfully!"
  # Ensure permissions are correct
  chmod 666 test_shared/models/song_recommender.pkl
else
  echo "Error: Model was not created. Check the logs above."
  exit 1
fi

# Stop and remove any existing container with the same name
echo "Stopping and removing any existing container with name ${CONTAINER_NAME}..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Run Frontend container
echo "Running Frontend container..."
docker run -d --name ${CONTAINER_NAME} \
  -v "$(pwd)/test_shared:/shared" \
  -p 5000:5000 \
  ${FRONTEND_IMAGE}

echo "Frontend container started. Waiting for it to initialize..."
sleep 10  # Increased wait time to ensure the container is fully initialized

# Test the API
echo "Testing the API..."
echo "Making request to http://localhost:5000/api/recommend with songs: Ride Wit Me, Sweet Emotion"
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"songs": ["Ride Wit Me", "Sweet Emotion"]}' \
  http://localhost:5000/api/recommend

echo ""
echo "Frontend is running at http://localhost:5000"
echo "Press Ctrl+C to stop the container when done testing."
echo ""

echo "To stop the container manually:"
echo "docker stop ${CONTAINER_NAME}"
echo ""
echo "To push images to Docker Hub:"
echo "docker login"
echo "docker push ${DOCKER_REPO}/song-recommender-ml:${IMAGE_TAG}"
echo "docker push ${DOCKER_REPO}/song-recommender-frontend:${IMAGE_TAG}"

# Wait for user input
echo "Press Enter to stop the container..."
read

# Stop the container
docker stop ${CONTAINER_NAME}