#!/usr/bin/env python3

"""
Model training script for the ML container.
This script will train the model and save it to the shared volume.
"""

import os
import sys
import time
import logging
import hashlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import the song_recommender module
sys.path.append("/app")

# Import our song recommender class
from song_recommender import SongRecommender

# Constants
DATA_DIR = os.environ.get('DATA_DIR', '/data')
SHARED_DIR = os.environ.get('SHARED_DIR', '/shared')
MODEL_DIR = os.path.join(SHARED_DIR, 'models')
# Get dataset version from environment variable (default to ds2 if not specified)
DATASET_VERSION = os.environ.get('DATASET_VERSION', 'ds2')
PLAYLIST_FILE = os.environ.get('PLAYLIST_FILE', os.path.join(DATA_DIR, f'2023_spotify_{DATASET_VERSION}.csv'))
SONGS_FILE = os.environ.get('SONGS_FILE', os.path.join(DATA_DIR, '2023_spotify_songs.csv'))
MODEL_FILE = os.environ.get('MODEL_FILE', os.path.join(MODEL_DIR, 'song_recommender.pkl'))
INFO_FILE = os.environ.get('INFO_FILE', os.path.join(MODEL_DIR, 'model_info.json'))

# Training parameters
MIN_SUPPORT = float(os.environ.get('MIN_SUPPORT', '0.05'))
MIN_CONFIDENCE = float(os.environ.get('MIN_CONFIDENCE', '0.1'))

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def save_model_info(model_file, training_parameters):
    """Save model metadata to a JSON file."""
    if not os.path.exists(model_file):
        logger.error(f"Model file {model_file} does not exist")
        return
    
    # Get file stats
    file_size = os.path.getsize(model_file)
    file_hash = calculate_file_hash(model_file)
    modified_date = datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Create model info
    model_info = {
        "version": "1.0.0",
        "model_file": os.path.basename(model_file),
        "model_size": file_size,
        "model_hash": file_hash,
        "modified_date": modified_date,
        "dataset_version": DATASET_VERSION,
        "training_parameters": training_parameters,
        "training_time": time.time()
    }
    
    # Save to file
    os.makedirs(os.path.dirname(INFO_FILE), exist_ok=True)
    with open(INFO_FILE, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Model info saved to {INFO_FILE}")
    return model_info

def main():
    """Main function to train the model and save it."""
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info("Starting model training...")
    logger.info(f"Dataset Version: {DATASET_VERSION}")
    logger.info(f"Parameters: MIN_SUPPORT: {MIN_SUPPORT}, MIN_CONFIDENCE: {MIN_CONFIDENCE}")
    logger.info(f"Files: PLAYLIST_FILE: {PLAYLIST_FILE}, SONGS_FILE: {SONGS_FILE}")
    
    # Check if data files exist
    if not os.path.exists(PLAYLIST_FILE) or not os.path.exists(SONGS_FILE):
        logger.error(f"Data files not found")
        return 1
    
    try:
        # Create and train recommender
        recommender = SongRecommender()
        
        # Load data
        logger.info("Loading and preparing data...")
        recommender.load_and_prepare_data(
            playlist_file=PLAYLIST_FILE,
            songs_file=SONGS_FILE
        )
        
        # Set parameters
        recommender.min_support = MIN_SUPPORT
        recommender.min_confidence = MIN_CONFIDENCE
        
        # Mine frequent itemsets using Apriori
        logger.info("Mining frequent itemsets using Apriori algorithm...")
        recommender.mine_frequent_itemsets()
        
        # Save the model
        logger.info(f"Saving model to {MODEL_FILE}...")
        recommender.save_model(MODEL_FILE)
        
        # Save model info
        training_parameters = {
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "algorithm": "apriori",
            "dataset_version": DATASET_VERSION
        }
        save_model_info(MODEL_FILE, training_parameters)
        
        logger.info("Training completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 