# Song Recommendation System

A recommendation system for songs that combines frequent itemset mining and NLP-based similarity search.

## Overview

This system recommends songs based on:
1. **Frequent Itemset Mining**: Analyzes patterns in playlists to find songs that frequently appear together
2. **NLP-based Similarity**: Uses TF-IDF to find songs with similar names

## Data Files

The system uses two types of data files:
- **Playlist data files** (e.g., `2023_spotify_ds1.csv` or `2023_spotify_ds2.csv`): Large files containing playlist information used for frequent itemset mining
- **Song data file** (e.g., `2023_spotify_songs.csv`): A smaller file containing unique songs used for NLP-based similarity

## Requirements

The following packages are required:
- pandas
- numpy
- scikit-learn
- mlxtend
- flask
- requests

You can install them using:
```
pip install -r requirements.txt
```

## Files

- `song_recommender.py` - Main recommendation system class
- `recommend.py` - Command-line interface for recommendations
- `test_recommender.py` - Script to test the recommendation system
- `app.py` - Flask web server for the API and web interface
- `data/` - Directory containing the Spotify datasets
- `models/` - Directory to store trained models
- `templates/` - HTML templates for the web interface
- `static/` - Static files (CSS, JavaScript) for the web interface

## Usage

### Web Interface

The easiest way to use the system is through the web interface:

1. Start the server:
```
bash start_server.sh
```

2. Open your browser and navigate to:
```
http://localhost:52010/
```

3. Enter song names, choose the number of recommendations, and click "Get Recommendations"

### REST API

The system provides a REST API for recommendations:

```
POST /api/recommend
```

Request body:
```json
{
  "songs": ["Song Name 1", "Song Name 2"]
}
```

Response:
```json
{
  "songs": ["Recommended Song 1", "Recommended Song 2", ...],
  "version": "1.0.0",
  "model_date": "2023-02-25 08:30:45"
}
```

### Command-line Interface

Train a new model and get recommendations:
```
python recommend.py --train --songs "Ride Wit Me" "Sweet Emotion" --count 5
```

Load an existing model and get recommendations:
```
python recommend.py --songs "Ride Wit Me" "Sweet Emotion" --count 5
```

Command-line options:
```
  --songs SONGS [SONGS ...]  Input song names (one or more)
  --count COUNT             Number of recommendations (max 10)
  --model MODEL             Path to the model file
  --train                   Train a new model before making recommendations
  --playlist-file FILE      Playlist dataset file for training (only used with --train)
  --songs-file FILE         Songs dataset file for NLP similarity (only used with --train)
  --min-support MIN_SUPPORT  Minimum support for frequent itemsets (only used with --train)
  --min-confidence MIN_CONFIDENCE  Minimum confidence for association rules (only used with --train)
```

### Python API

#### Training a model

```python
from song_recommender import SongRecommender

# Create a new recommender
recommender = SongRecommender()

# Load and prepare data (using separate files for playlists and songs)
recommender.load_and_prepare_data(
    playlist_file="data/2023_spotify_ds1.csv",
    songs_file="data/2023_spotify_songs.csv"
)

# Optional: Set parameters
recommender.min_support = 0.005  # Default is 0.01
recommender.min_confidence = 0.2  # Default is 0.2

# Mine frequent itemsets
recommender.mine_frequent_itemsets()

# Save the model
recommender.save_model("models/song_recommender.pkl")
```

#### Making recommendations

```python
from song_recommender import SongRecommender

# Load an existing model
recommender = SongRecommender.load_model("models/song_recommender.pkl")

# Get recommendations
input_songs = ["Ride Wit Me", "Sweet Emotion"]
recommendations = recommender.recommend_songs(input_songs, 5)
print(recommendations)
```

#### Running the test script

```
python test_recommender.py
```

## How It Works

### Frequent Itemset Mining

The system uses the Apriori algorithm to identify frequent itemsets and association rules between songs. Each playlist is treated as a "basket" of items (songs), and the algorithm finds patterns of songs that frequently appear together. For better performance, the system limits the itemset size to 2 items.

### NLP-based Similarity

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) with character-level n-grams to find songs with similar names. This helps recommend songs that may have similar titles but weren't necessarily in the same playlists.

When multiple input songs are provided, the system calculates similarity scores for each input song and combines them to find songs that are similar to ANY of the input songs, while giving fair consideration to all input songs.

### Balanced Recommendation Approach

The final recommendations are a balanced combination of both methods:

1. The system allocates approximately 50% of the recommendation slots to each method (association rules and similarity).
2. If one method doesn't have enough recommendations, the remaining slots are allocated to the other method.
3. This ensures a diverse set of recommendations that balance popularity patterns (from association rules) with song title similarity.

## Parameters

- `min_support`: Minimum support threshold for frequent itemsets (default: 0.01)
- `min_confidence`: Minimum confidence threshold for association rules (default: 0.2)
- `num_recommendations`: Maximum number of recommendations to return (max: 10)

# Song Recommendation System - Docker and Kubernetes Deployment

This README provides instructions for building, testing, and deploying the Song Recommendation System using Docker and Kubernetes.

## Project Structure

- `docker/` - Contains Docker-related files
  - `build_containers.sh` - Script to build Docker images
  - `test_containers.sh` - Script to test Docker containers locally
  - `Dockerfile.ml` - Dockerfile for the ML component
  - `Dockerfile.frontend` - Dockerfile for the Frontend component
- `kubernetes/` - Contains Kubernetes configuration files
  - `apply_kubernetes.sh` - Script to apply Kubernetes configurations
  - `ml-job.yaml` - Kubernetes Job for the ML component
  - `frontend-deployment.yaml` - Kubernetes Deployment for the Frontend
  - `frontend-service.yaml` - Kubernetes Service for the Frontend
  - `pvc.yaml` - PersistentVolumeClaim for shared storage

## Prerequisites

- Docker installed and configured
- Docker Hub account (default namespace: yuandongzhang)
- Kubernetes cluster access configured with kubectl
- Dataset files in the `data/` directory:
  - `2023_spotify_songs.csv`
  - `2023_spotify_ds2.csv`

## Building and Testing Docker Containers

### 1. Build the Docker images

```bash
cd hw2/docker
./build_containers.sh
```

This script builds two Docker images:
- `yuandongzhang/song-recommender-ml:1.0.0` - ML component for training the model
- `yuandongzhang/song-recommender-frontend:1.0.0` - Frontend component for serving recommendations

### 2. Test the containers locally

```bash
cd hw2/docker
./test_containers.sh
```

This script:
1. Creates test directories for shared volume and data
2. Copies dataset files to the test data directory
3. Runs the ML container to train the model
4. Verifies the model was created successfully
5. Runs the Frontend container
6. Tests the API with sample song inputs
7. Stops the Frontend container
8. Provides commands to push images to Docker Hub

### 3. Push images to Docker Hub

```bash
docker login
docker push yuandongzhang/song-recommender-ml:1.0.0
docker push yuandongzhang/song-recommender-frontend:1.0.0
```

## Deploying to Kubernetes

### 1. Apply Kubernetes configurations

```bash
cd hw2/kubernetes
./apply_kubernetes.sh
```

This script:
1. Replaces environment variables in YAML files
2. Creates a PersistentVolumeClaim for shared storage
3. Runs the ML Job to train the model
4. Deploys the Frontend Deployment and Service

### 2. Verify deployment

```bash
kubectl get pods -n yuandong
kubectl get services -n yuandong
kubectl get jobs -n yuandong
```

### 3. Access the application

The Frontend Service is exposed as a ClusterIP service on port 30502. You can access it using:

```bash
kubectl port-forward service/song-recommender-frontend-service 5000:30502 -n yuandong
```

Then open your browser to http://localhost:5000

## Customization

You can customize the deployment by modifying the environment variables:

- `DOCKER_REPO` - Docker Hub namespace (default: yuandongzhang)
- `IMAGE_TAG` - Image tag (default: 1.0.0)

For ML training parameters:
- `MIN_SUPPORT` - Minimum support for association rules (default: 0.002)
- `MIN_CONFIDENCE` - Minimum confidence for association rules (default: 0.01)
- `MAX_PLAYLISTS` - Maximum number of playlists to process (default: 1000)
- `MAX_SONGS_PER_PLAYLIST` - Maximum songs per playlist (default: 30)

## Troubleshooting

If you encounter issues:

1. Check the ML Job logs:
```bash
kubectl logs job/song-recommender-ml-job -n yuandong
```

2. Check the Frontend pod logs:
```bash
kubectl logs deployment/song-recommender-frontend -n yuandong
```

3. Verify the model was created in the shared volume:
```bash
kubectl exec deployment/song-recommender-frontend -n yuandong -- ls -la /shared/models
``` 