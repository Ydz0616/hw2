#!/usr/bin/env python3

"""
Model training script for the ML container.
This script will train the model and save it to the shared volume.
"""

import os
import sys
import time
import pickle
import logging
import hashlib
import json
import csv
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
PLAYLIST_FILE = os.environ.get('PLAYLIST_FILE', os.path.join(DATA_DIR, '2023_spotify_ds2.csv'))
SONGS_FILE = os.environ.get('SONGS_FILE', os.path.join(DATA_DIR, '2023_spotify_songs.csv'))
MODEL_FILE = os.environ.get('MODEL_FILE', os.path.join(MODEL_DIR, 'song_recommender.pkl'))
RULES_FILE = os.environ.get('RULES_FILE', os.path.join(MODEL_DIR, 'association_rules.csv'))
INFO_FILE = os.environ.get('INFO_FILE', os.path.join(MODEL_DIR, 'model_info.json'))

# Training parameters
MIN_SUPPORT = float(os.environ.get('MIN_SUPPORT', '0.2'))
MIN_CONFIDENCE = float(os.environ.get('MIN_CONFIDENCE', '0.2'))

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
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
        "training_parameters": training_parameters,
        "training_time": time.time()
    }
    
    # Save to file
    os.makedirs(os.path.dirname(INFO_FILE), exist_ok=True)
    with open(INFO_FILE, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Model info saved to {INFO_FILE}")
    return model_info

def save_rules_to_file(recommender, output_file=RULES_FILE):
    """Save all generated association rules to a CSV file for easier viewing."""
    if not recommender.association_rules:
        logger.info("No rules to save.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format rules for CSV - adapted for FP-Growth rule format
    rules_data = []
    for antecedent, consequent, confidence in recommender.association_rules:
        # Find support for this itemset if available
        support = 0.0
        for itemset, supp in recommender.freq_itemsets:
            if set(itemset) == set(list(antecedent) + list(consequent)):
                support = supp
                break
        
        # Calculate lift (if possible)
        # Lift = confidence / support(consequent)
        # Skip lift calculation as we don't have easy access to consequent support
        lift = "N/A"
        
        rules_data.append({
            'Antecedent': str(list(antecedent)),
            'Consequent': str(list(consequent)),
            'Support': support,
            'Confidence': confidence,
            'Lift': lift
        })
    
    # Sort by confidence for easier viewing
    rules_data.sort(key=lambda x: x['Confidence'], reverse=True)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        fields = ['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rules_data)
    
    logger.info(f"Saved {len(rules_data)} rules to {output_file}")

def train_model(playlist_file=PLAYLIST_FILE, songs_file=SONGS_FILE, model_file=MODEL_FILE, 
                min_support=0.05, min_confidence=0.1):
    """Function to train the model with given parameters. Can be called directly via API."""
    try:
        # Create a new recommender
        recommender = SongRecommender()
        min_support = 0.05
        min_confidence = 0.1
        # Load and prepare data
        logger.info("Loading and preparing data...")
        recommender.load_and_prepare_data(
            playlist_file=playlist_file,
            songs_file=songs_file
        )
        
        # Set parameters
        recommender.min_support = min_support
        recommender.min_confidence = min_confidence
        
        # Mine frequent itemsets
        logger.info("Mining frequent itemsets using FP-Growth...")
        recommender.mine_frequent_itemsets()
        
        # Save the model
        logger.info(f"Saving model to {model_file}...")
        recommender.save_model(model_file)
        
        # Save rules to file
        save_rules_to_file(recommender)
        
        # Save model info
        training_parameters = {
            "min_support": min_support,
            "min_confidence": min_confidence,
            "algorithm": "fp-growth"  # Mark that we're using FP-Growth
        }
        model_info = save_model_info(model_file, training_parameters)
        
        logger.info("Training completed successfully!")
        return {
            "status": "success",
            "model_info": model_info
        }
    
    except Exception as e:
        error_msg = f"Error during training: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg
        }

def main():
    """Main function to train the model and save it."""
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info("Starting model training...")
    logger.info(f"Parameters:")
    logger.info(f"  - MIN_SUPPORT: {MIN_SUPPORT}")
    logger.info(f"  - MIN_CONFIDENCE: {MIN_CONFIDENCE}")
    logger.info(f"  - PLAYLIST_FILE: {PLAYLIST_FILE}")
    logger.info(f"  - SONGS_FILE: {SONGS_FILE}")
    logger.info(f"  - MODEL_FILE: {MODEL_FILE}")
    
    # Check if data files exist
    if not os.path.exists(PLAYLIST_FILE):
        logger.error(f"Playlist file not found: {PLAYLIST_FILE}")
        return 1
    
    if not os.path.exists(SONGS_FILE):
        logger.error(f"Songs file not found: {SONGS_FILE}")
        return 1
    
    result = train_model()
    return 0 if result["status"] == "success" else 1


# API endpoint to initiate training - can be used if exposing this as a service
def api_train_model(request_data=None):
    """API wrapper for train_model function."""
    if request_data is None:
        request_data = {}
    
    # Extract parameters from request with defaults
    playlist_file = request_data.get('playlist_file', PLAYLIST_FILE)
    songs_file = request_data.get('songs_file', SONGS_FILE)
    model_file = request_data.get('model_file', MODEL_FILE)
    min_support = float(request_data.get('min_support', MIN_SUPPORT))
    min_confidence = float(request_data.get('min_confidence', MIN_CONFIDENCE))
    
    # Validate parameters
    if min_support <= 0 or min_support > 1:
        return {"status": "error", "message": "min_support must be between 0 and 1"}
    if min_confidence <= 0 or min_confidence > 1:
        return {"status": "error", "message": "min_confidence must be between 0 and 1"}
    
    # Call the training function
    return train_model(
        playlist_file=playlist_file,
        songs_file=songs_file,
        model_file=model_file,
        min_support=min_support,
        min_confidence=min_confidence
    )


if __name__ == "__main__":
    sys.exit(main())