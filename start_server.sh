#!/bin/bash
# Script to start the Flask server

# Make sure the model exists
if [ ! -f "models/song_recommender.pkl" ]; then
    echo "Model file not found. Training the model first..."
    python test_recommender.py
fi

# Set environment variables
export FLASK_APP=app.py

# Start the server
echo "Starting Flask server on port 52010..."
echo "Web interface available at http://localhost:52010/"
flask run --host=0.0.0.0 --port=52010 