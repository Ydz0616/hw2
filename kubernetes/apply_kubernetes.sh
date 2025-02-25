#!/bin/bash

# Script to apply Kubernetes configurations for the song recommendation system

# Set environment variables
DOCKER_REPO=${DOCKER_REPO:-"yuandongzhang"}
IMAGE_TAG=${IMAGE_TAG:-"1.0.0"}

echo "=== Song Recommendation System Kubernetes Deployment ==="
echo "Docker Repository: ${DOCKER_REPO}"
echo "Image Tag: ${IMAGE_TAG}"
echo ""

# Check if the project2-pv directory exists
if [ ! -d "/home/yuandong/project2-pv" ]; then
    echo "Creating project2-pv directory..."
    mkdir -p /home/yuandong/project2-pv
    chmod 777 /home/yuandong/project2-pv
fi

# Create models directory inside project2-pv
if [ ! -d "/home/yuandong/project2-pv/models" ]; then
    echo "Creating models directory inside project2-pv..."
    mkdir -p /home/yuandong/project2-pv/models
    chmod 777 /home/yuandong/project2-pv/models
fi

# Replace placeholders in YAML files
echo "Replacing placeholders in YAML files..."
sed -i "s|\${DOCKER_REPO}|${DOCKER_REPO}|g" ml-job.yaml
sed -i "s|\${IMAGE_TAG}|${IMAGE_TAG}|g" ml-job.yaml
sed -i "s|\${DOCKER_REPO}|${DOCKER_REPO}|g" frontend-deployment.yaml
sed -i "s|\${IMAGE_TAG}|${IMAGE_TAG}|g" frontend-deployment.yaml


# Apply ML job
echo "Applying ML job..."
kubectl apply -f ml-job.yaml
echo "ML job applied. This will train the model and save it to the shared volume."
echo "You can check the status with: kubectl -n yuandong get jobs"
echo "And view logs with: kubectl -n yuandong logs job/song-recommender-ml-job"

# Wait for ML job to complete
echo "Waiting for ML job to complete (this may take a few minutes)..."
kubectl -n yuandong wait --for=condition=complete --timeout=300s job/song-recommender-ml-job
if [ $? -ne 0 ]; then
    echo "ML job did not complete within timeout. Check logs for details."
    echo "Continuing with deployment anyway..."
fi

# Apply Frontend deployment and service
echo "Applying Frontend deployment and service..."
kubectl apply -f frontend-deployment.yaml
kubectl apply -f frontend-service.yaml

echo ""
echo "=== Deployment Complete ==="
echo "To check the status of your resources:"
echo "  kubectl -n yuandong get pods"
echo "  kubectl -n yuandong get jobs"
echo "  kubectl -n yuandong get services"
echo "  kubectl -n yuandong get pvc"
echo ""
echo "To access the frontend service:"
echo "  kubectl -n yuandong port-forward service/song-recommender-frontend-service 5000:30502"
echo "  Then open http://localhost:5000 in your browser"
echo ""
echo "To view logs:"
echo "  kubectl -n yuandong logs -f deployment/song-recommender-frontend"
echo "  kubectl -n yuandong logs job/song-recommender-ml-job" 