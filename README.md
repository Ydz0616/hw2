# Song Recommendation System

## Overview
This project implements a hybrid song recommendation system that combines association rule mining and NLP-based similarity matching to suggest music based on user preferences. The system is deployed using Docker containers orchestrated by Kubernetes, with continuous deployment managed through Argo CD.

## Features
- **Hybrid Recommendation Algorithm**: Combines Apriori algorithm for playlist pattern mining and TF-IDF for song title similarity
- **Interactive UI**: Clean, responsive interface with tabbed display for different recommendation types
- **Scalable Architecture**: Kubernetes-based deployment with configurable scaling
- **Flexible Dataset Selection**: Support for switching between different training datasets (ds1/ds2)
- **CI/CD Pipeline**: Automated deployment through Argo CD following GitOps principles

## Architecture
- **ML Component**: Trains the recommendation model using configurable parameters
- **Backend API**: Flask-based REST API that serves recommendations
- **Frontend**: User interface for song input and recommendation display

## Project Structure
```
/hw2
├── data/                  # Dataset files
│   ├── 2023_spotify_ds1.csv
│   ├── 2023_spotify_ds2.csv
│   └── 2023_spotify_songs.csv
├── docker/                # Docker configuration
│   ├── frontend/          # Frontend service
│   │   ├── static/        # CSS and JavaScript files
│   │   ├── templates/     # HTML templates
│   │   ├── app.py         # Flask application
│   │   └── Dockerfile
│   └── ml/                # ML training service
│       ├── train_model.py # Model training script
│       └── Dockerfile
├── kubernetes/            # Kubernetes manifests
│   ├── frontend-deployment.yaml
│   ├── ml-job-base.yaml
│   ├── ml-job-ds2.yaml
│   ├── pvc.yaml
│   └── kustomization.yaml
└── model_local_test/      # Local testing scripts
    └── song_recommender.py
```

## Deployment
The system is containerized using Docker and deployed on Kubernetes:
- ML training runs as Kubernetes Jobs
- Frontend service runs as a Deployment with multiple replicas
- Shared PersistentVolumeClaim for model storage
- Ingress for external access

### Prerequisites
- Docker
- Kubernetes cluster
- Argo CD installed on the cluster

### Deployment Steps
1. Build and push Docker images:
   ```bash
   docker build -t yuandongzhang/song-recommender-ml:latest ./docker/ml
   docker build -t yuandongzhang/song-recommender-frontend:latest ./docker/frontend
   docker push yuandongzhang/song-recommender-ml:latest
   docker push yuandongzhang/song-recommender-frontend:latest
   ```

2. Deploy to Kubernetes:
   ```bash
   kubectl apply -k kubernetes/
   ```

3. Train the model:
   ```bash
   # For dataset 1
   kubectl apply -f kubernetes/ml-job-base.yaml
   
   # For dataset 2
   kubectl apply -f kubernetes/ml-job-ds2.yaml
   ```

4. Access the application:

   Please contact yuandong.zhang@duke.edu for access requests
   
## Configuration
- **ML Parameters**: 
  - `MIN_SUPPORT`: Minimum support threshold for Apriori algorithm (default: 0.05)
  - `MIN_CONFIDENCE`: Minimum confidence threshold for association rules (default: 0.1)
  - `DATASET_VERSION`: Dataset version to use (ds1 or ds2)

- **Frontend Scaling**:
  - Adjust `replicas` in `kubernetes/frontend-deployment.yaml` to scale the frontend service

## Algorithm
The recommendation system uses two complementary approaches:

1. **Association Rule Mining**: Identifies patterns in playlist data using the Apriori algorithm to find songs that frequently appear together.

2. **NLP-based Similarity**: Uses TF-IDF vectorization with character n-grams to find songs with similar titles using cosine similarity.

The final recommendations combine both approaches to provide diverse and relevant suggestions.

## Testing
The system has been successfully tested for:
- Horizontal scaling capabilities
- Dataset switching flexibility
- Container version updates

## Technologies Used
- **Backend**: Python, Flask, MLxtend, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Infrastructure**: Docker, Kubernetes, Argo CD
- **Data Processing**: Pandas, NumPy

## Future Improvements
- Add user authentication and personalized recommendations
- Implement A/B testing for algorithm improvements
- Add monitoring and alerting for system performance
- Expand NLP capabilities for better similarity matching 