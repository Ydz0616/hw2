#!/usr/bin/env python3
"""
Command-line interface for song recommendations.
"""

import argparse
import os
import csv
from song_recommender import SongRecommender

def save_rules_to_file(recommender, output_file='models/association_rules.csv'):
    """
    Save all generated association rules to a CSV file for easier viewing.
    
    Args:
        recommender: Trained SongRecommender instance
        output_file: Path to save the rules
    """
    if recommender.association_rules is None or recommender.association_rules.empty:
        print("No rules to save.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format rules for CSV
    rules_data = []
    for _, rule in recommender.association_rules.iterrows():
        antecedent = list(rule['antecedents'])
        consequent = list(rule['consequents'])
        support = rule['support']
        confidence = rule['confidence']
        lift = rule['lift']
        
        rules_data.append({
            'Antecedent': str(antecedent),
            'Consequent': str(consequent),
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
    
    print(f"Saved {len(rules_data)} rules to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Get song recommendations.')
    parser.add_argument('--songs', nargs='+', help='Input song names (one or more)')
    parser.add_argument('--count', type=int, default=5, help='Number of recommendations (max 10)')
    parser.add_argument('--model', default='models/song_recommender.pkl', 
                        help='Path to the model file')
    parser.add_argument('--train', action='store_true', 
                        help='Train a new model before making recommendations')
    parser.add_argument('--playlist-file', default='data/2023_spotify_ds2.csv',
                        help='Playlist dataset file for training (only used with --train)')
    parser.add_argument('--songs-file', default='data/2023_spotify_songs.csv',
                        help='Songs dataset file for NLP similarity (only used with --train)')
    parser.add_argument('--min-support', type=float, default=0.005,
                        help='Minimum support for frequent itemsets (only used with --train)')
    parser.add_argument('--min-confidence', type=float, default=0.05,
                        help='Minimum confidence for association rules (only used with --train)')
    parser.add_argument('--max-playlists', type=int, default=2000,
                        help='Maximum number of playlists to use (only used with --train)')
    parser.add_argument('--max-songs-per-playlist', type=int, default=30,
                        help='Maximum number of songs per playlist (only used with --train)')
    parser.add_argument('--save-rules', action='store_true',
                        help='Save association rules to a CSV file for easier viewing')
    parser.add_argument('--rules-file', default='models/association_rules.csv',
                        help='Path to save the association rules (only used with --save-rules)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.train and not args.songs:
        parser.error("--songs is required unless --train is specified")
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    
    # Cap the number of recommendations at 10
    count = min(args.count, 10)
    
    if args.train:
        print(f"Training new model...")
        recommender = SongRecommender()
        
        print(f"Loading data from:")
        print(f"  - Playlist file: {args.playlist_file}")
        print(f"  - Songs file: {args.songs_file}")
        
        recommender.load_and_prepare_data(
            playlist_file=args.playlist_file,
            songs_file=args.songs_file
        )
        
        # Set parameters
        recommender.min_support = args.min_support
        recommender.min_confidence = args.min_confidence
        
        print(f"Parameters:")
        print(f"  - min_support: {args.min_support}")
        print(f"  - min_confidence: {args.min_confidence}")
        print(f"  - max_playlists: {args.max_playlists}")
        print(f"  - max_songs_per_playlist: {args.max_songs_per_playlist}")
        
        # Mine frequent itemsets with optimized parameters
        print("Mining frequent itemsets...")
        recommender.mine_frequent_itemsets(
            max_playlists=args.max_playlists,
            max_songs_per_playlist=args.max_songs_per_playlist
        )
        
        # Save rules to file if requested
        if args.save_rules:
            save_rules_to_file(recommender, args.rules_file)
        
        # Save the model
        print(f"Saving model to {args.model}...")
        recommender.save_model(args.model)
        
        # If songs argument is not provided, we're done
        if not args.songs:
            print(f"Model trained and saved to {args.model}.")
            return
    else:
        if not os.path.exists(args.model):
            print(f"Model file {args.model} not found. Please train a model first.")
            return
        
        print(f"Loading model from {args.model}...")
        recommender = SongRecommender.load_model(args.model)
        
        # Save rules to file if requested
        if args.save_rules:
            save_rules_to_file(recommender, args.rules_file)
            if not args.songs:
                return
    
    # Get recommendations
    print(f"\nGetting recommendations for: {args.songs}")
    recommendations = recommender.recommend_songs(args.songs, count)
    
    # Print recommendations
    print("\nRecommended songs:")
    for i, song in enumerate(recommendations, 1):
        print(f"{i}. {song}")


if __name__ == "__main__":
    main() 