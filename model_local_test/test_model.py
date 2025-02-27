#!/usr/bin/env python3

"""
Test script for the song recommender model.
This script trains a model using local data and tests recommendations.
"""

import os
import sys
import time
import argparse
from song_recommender import SongRecommender

# Paths for testing
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
PLAYLIST_FILE = os.path.join(DATA_DIR, '2023_spotify_ds1.csv')
SONGS_FILE = os.path.join(DATA_DIR, '2023_spotify_songs.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'test_recommender.pkl')

def test_train_model(min_support= 0.05 , min_confidence= 0.1 ):
    """Train a model with the provided parameters and test it."""
    print(f"Starting model training test...")
    min_support = 0.05
    min_confidence = 0.1
    print(f"Using data directory: {DATA_DIR}")
    print(f"Model will be saved to: {MODEL_FILE}")

    print (min_support, min_confidence)
    # Check if data files exist
    if not os.path.exists(PLAYLIST_FILE):
        print(f"ERROR: Playlist file not found: {PLAYLIST_FILE}")
        return False
    
    if not os.path.exists(SONGS_FILE):
        print(f"ERROR: Songs file not found: {SONGS_FILE}")
        return False
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Create a new recommender
        print("\n1. Creating recommender instance...")
        recommender = SongRecommender()
        
        # Set parameters
        recommender.min_support = min_support
        recommender.min_confidence = min_confidence
        print(f"   - Set min_support = {min_support}")
        print(f"   - Set min_confidence = {min_confidence}")
        
        # Load and prepare data
        print("\n2. Loading and preparing data...")
        recommender.load_and_prepare_data(
            playlist_file=PLAYLIST_FILE,
            songs_file=SONGS_FILE
        )
        print(f"   - Loaded {len(recommender.playlist_songs)} playlists")
        print(f"   - Loaded {len(recommender.song_names)} unique songs")
        
        # Mine frequent itemsets
        print("\n3. Mining frequent itemsets with FP-Growth...")
        recommender.mine_frequent_itemsets()
        print(f"   - Found {len(recommender.freq_itemsets)} frequent itemsets")
        print(f"   - Generated {len(recommender.association_rules)} association rules")
        
        # Save the model
        print("\n4. Saving model...")
        recommender.save_model(MODEL_FILE)
        print(f"   - Model saved to {MODEL_FILE}")
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Return the trained model for further testing
        return recommender
    
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_recommendations(recommender=None, input_songs=None):
    """Test the recommendation functionality with the given model and input songs."""
    if recommender is None:
        # Try to load the model
        if not os.path.exists(MODEL_FILE):
            print(f"ERROR: Model file not found: {MODEL_FILE}")
            return False
        
        try:
            print("\n5. Loading saved model...")
            recommender = SongRecommender.load_model(MODEL_FILE)
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            return False
    
    # If no input songs provided, use some default test cases
    if input_songs is None or len(input_songs) == 0:
        print("\n6. No input songs provided, trying to find some sample songs...")
        
        # Get some popular songs from the model
        if len(recommender.song_playlists) > 0:
            # Find songs that appear in multiple playlists
            popular_songs = sorted(
                [(song, len(playlists)) for song, playlists in recommender.song_playlists.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            if popular_songs:
                # Take 2 random songs from the top 10 popular songs
                import random
                input_songs = [song for song, _ in random.sample(popular_songs, min(2, len(popular_songs)))]
                print(f"   - Using popular songs as input: {input_songs}")
        
        # If we still don't have input songs, use some defaults that might be in the dataset
        if not input_songs:
            input_songs = ["Shape of You", "Blinding Lights"]
            print(f"   - Using default songs as input: {input_songs}")
    
    # Get recommendations
    print(f"\n7. Getting recommendations for: {input_songs}")
    try:
        # Test both recommendation methods
        print("\n   A. Testing rule-based recommendations...")
        rule_recs = recommender._recommend_from_rules(input_songs, 5)
        print(f"      Found {len(rule_recs)} rule-based recommendations")
        
        print("\n   B. Testing similarity-based recommendations...")
        sim_recs = recommender._find_similar_song_names(input_songs, 5)
        print(f"      Found {len(sim_recs)} similarity-based recommendations")
        
        print("\n   C. Testing combined recommendations...")
        combined_recs = recommender.recommend_songs(input_songs, 5)
        print(f"      Final recommendations: {combined_recs}")
        
        return True
        
    except Exception as e:
        print(f"ERROR during recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description='Test the song recommender model')
    
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--recommend', action='store_true', help='Test recommendations')
    parser.add_argument('--songs', nargs='+', help='Input songs for recommendation test')
    parser.add_argument('--min-support', type=float, default=0.02, help='Minimum support ratio')
    parser.add_argument('--min-confidence', type=float, default=0.01, help='Minimum confidence')
    
    args = parser.parse_args()
    
    # If no arguments, run both train and recommend
    if not (args.train or args.recommend):
        args.train = True
        args.recommend = True
    
    recommender = None
    
    # Train if requested
    if args.train:
        print("=" * 80)
        print("TESTING MODEL TRAINING")
        print("=" * 80)
        recommender = test_train_model(
            min_support=args.min_support,
            min_confidence=args.min_confidence
        )
        if recommender is None:
            print("Training failed, cannot proceed with recommendation test")
            return 1
    
    # Test recommendations if requested
    if args.recommend:
        print("\n" + "=" * 80)
        print("TESTING RECOMMENDATIONS")
        print("=" * 80)
        success = test_recommendations(recommender, args.songs)
        if not success:
            print("Recommendation test failed")
            return 1
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())