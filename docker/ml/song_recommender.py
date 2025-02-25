import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random


class SongRecommender:
    """
    A song recommendation system that combines frequent itemset mining and NLP-based similarity.
    """
    
    def __init__(self):
        # For frequent itemset mining
        self.playlist_songs = defaultdict(list)
        self.song_playlists = defaultdict(list)
        self.association_rules = None
        self.min_support = 0.01  # Default value, can be adjusted
        self.min_confidence = 0.2  # Default value, can be adjusted
        
        # For NLP-based similarity
        self.song_names = []
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.tfidf_matrix = None
    
    def load_and_prepare_data(self, playlist_file: str, songs_file: str = None) -> None:
        """
        Load the dataset and prepare data structures for recommendation.
        
        Args:
            playlist_file: Path to the CSV file with playlist data
            songs_file: Path to the CSV file with unique songs for NLP (optional)
        """
        # Load the playlist dataset
        playlist_df = pd.read_csv(playlist_file)
        
        # Extract relevant columns
        playlists_data = playlist_df[['pid', 'track_name']].drop_duplicates()
        
        # Create mappings
        for _, row in playlists_data.iterrows():
            pid = row['pid']
            track_name = row['track_name']
            
            self.playlist_songs[pid].append(track_name)
            self.song_playlists[track_name].append(pid)
        
        # For NLP, if a separate songs file is provided, use it
        if songs_file:
            try:
                songs_df = pd.read_csv(songs_file)
                self.song_names = songs_df['track_name'].tolist()
                print(f"Loaded {len(self.song_names)} unique songs for NLP similarity.")
            except Exception as e:
                print(f"Error loading songs file: {e}")
                # Fallback to using song names from playlist data
                self.song_names = list(self.song_playlists.keys())
        else:
            # Use song names from playlist data
            self.song_names = list(self.song_playlists.keys())
        
        # Train the TF-IDF vectorizer
        self._train_tfidf()
    
    def _train_tfidf(self) -> None:
        """Train the TF-IDF vectorizer on all song names."""
        if not self.song_names:
            return
        
        # Convert to strings in case there are any non-string entries
        self.song_names = [str(name) for name in self.song_names]
        
        print(f"Training TF-IDF on {len(self.song_names)} songs...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.song_names)
    
    def _find_similar_song_names(self, input_songs: List[str], top_n: int = 5) -> List[str]:
        """
        Find songs with similar names using TF-IDF and cosine similarity.
        
        Args:
            input_songs: List of input song names
            top_n: Number of similar songs to return
            
        Returns:
            List of similar song names
        """
        if self.tfidf_matrix is None or len(self.song_names) == 0:
            return []
        
        # Dictionary to track combined similarity scores across all input songs
        similarity_scores = {}
        
        # Process each input song
        for song_idx, song in enumerate(input_songs):
            # Transform input song to TF-IDF vector
            song_vec = self.tfidf_vectorizer.transform([str(song)])
            
            # Calculate similarity with all songs - but process in batches to save memory
            batch_size = 1000
            n_batches = (len(self.song_names) + batch_size - 1) // batch_size
            
            print(f"Processing similarity for input song: {song}")
            
            # Process each batch
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.song_names))
                
                # Get similarity for this batch
                batch_similarities = cosine_similarity(
                    song_vec, 
                    self.tfidf_matrix[start_idx:end_idx]
                ).flatten()
                
                # Update similarity scores for each song
                for j, similarity in enumerate(batch_similarities):
                    song_idx = start_idx + j
                    song_name = self.song_names[song_idx]
                    
                    # Skip the input song itself
                    if song_name in input_songs:
                        continue
                    
                    # Add to the similarity score dictionary (accumulate scores)
                    if song_name not in similarity_scores:
                        similarity_scores[song_name] = []
                    
                    similarity_scores[song_name].append(similarity)
        
        # Calculate the aggregated similarity score for each song
        # Using a weighted average of similarity scores - higher scores get more weight
        weighted_scores = {}
        for song_name, scores in similarity_scores.items():
            # Only keep songs that have similarity with all input songs
            if len(scores) == len(input_songs):
                # Calculate a weighted average - square the scores to give higher weight to higher similarities
                squared_scores = [score * score for score in scores]
                weighted_scores[song_name] = sum(squared_scores) / sum(1 for _ in squared_scores)
        
        # Sort by weighted similarity (descending) and filter out input songs
        sorted_similarities = sorted(
            weighted_scores.items(),
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"Found {len(sorted_similarities)} similar songs")
        
        # Return top_n songs with highest similarity scores
        return [song for song, _ in sorted_similarities[:top_n]]
    
    def mine_frequent_itemsets(self, max_playlists=1000, max_songs_per_playlist=None) -> None:
        """
        Apply frequent itemset mining to discover association rules among songs.
        
        Args:
            max_playlists: Maximum number of playlists to use (to limit memory usage)
            max_songs_per_playlist: Maximum number of songs to consider per playlist (None for all)
        """
        print(f"Processing playlists for frequent itemset mining...")
        
        # Take a sample of playlists if we have more than max_playlists
        playlist_ids = list(self.playlist_songs.keys())
        
        # Skip if there are no playlists
        if not playlist_ids:
            print("No playlists found. Cannot mine frequent itemsets.")
            return
        
        # If there are too many playlists, sample from them
        if len(playlist_ids) > max_playlists:
            print(f"Sampling {max_playlists} playlists from {len(playlist_ids)} total")
            sample_pids = random.sample(playlist_ids, max_playlists)
        else:
            sample_pids = playlist_ids
        
        # Create transactions list manually (more memory efficient than using TransactionEncoder)
        transactions = []
        
        # Limit the number of songs per playlist to save memory
        for pid in sample_pids:
            songs = self.playlist_songs[pid]
            
            # If we need to limit songs per playlist
            if max_songs_per_playlist and len(songs) > max_songs_per_playlist:
                songs = random.sample(songs, max_songs_per_playlist)
            
            transactions.append(songs)
        
        print(f"Created {len(transactions)} transactions")
        
        # Print some transaction information
        transaction_lengths = [len(t) for t in transactions]
        avg_length = sum(transaction_lengths) / len(transaction_lengths) if transaction_lengths else 0
        print(f"Average transaction length: {avg_length:.2f} songs")
        print(f"Min transaction length: {min(transaction_lengths) if transaction_lengths else 0} songs")
        print(f"Max transaction length: {max(transaction_lengths) if transaction_lengths else 0} songs")
        
        # Calculate song frequencies for diagnostic purposes
        song_freq = {}
        for transaction in transactions:
            for song in transaction:
                if song in song_freq:
                    song_freq[song] += 1
                else:
                    song_freq[song] = 1
        
        # Get the top 5 most frequent songs
        top_songs = sorted(song_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 most frequent songs:")
        for song, count in top_songs:
            print(f"  - {song}: {count} occurrences ({count/len(transactions):.3f} support)")
        
        # Check if there are any songs that appear in multiple playlists
        multi_playlist_songs = {song: count for song, count in song_freq.items() if count > 1}
        print(f"\nFound {len(multi_playlist_songs)} songs that appear in at least 2 playlists")
        
        # Instead of using the TransactionEncoder, we'll create our own one-hot encoding
        # First, get all unique songs across all transactions
        all_songs = set()
        for transaction in transactions:
            all_songs.update(transaction)
        
        print(f"Found {len(all_songs)} unique songs across all transactions")
        
        # If we have too many songs, we can sample from them
        if len(all_songs) > 10000:
            print(f"Sampling 10000 songs from {len(all_songs)} total")
            all_songs = set(random.sample(list(all_songs), 10000))
        
        # Map songs to indices for faster lookup
        song_to_idx = {song: i for i, song in enumerate(all_songs)}
        
        # Create a binary DataFrame where rows are playlists and columns are songs
        # This is a memory-efficient approach
        playlist_song_matrix = pd.DataFrame(
            [[song in transaction for song in all_songs] for transaction in transactions],
            columns=list(all_songs)
        )
        
        # Convert to bool type explicitly to avoid warnings and improve performance
        playlist_song_matrix = playlist_song_matrix.astype(bool)
        
        print(f"Created binary matrix of shape {playlist_song_matrix.shape}")
        
        # Apply the Apriori algorithm with optimized parameters
        print(f"Applying Apriori algorithm with min_support={self.min_support}...")
        frequent_itemsets = apriori(
            playlist_song_matrix,
            min_support=self.min_support,
            use_colnames=True,
            max_len=3,  # Allow for itemsets of up to 3 items
            low_memory=True  # Use low memory mode
        )
        
        if len(frequent_itemsets) == 0:
            print("No frequent itemsets found. Try lowering the minimum support threshold.")
            return
        
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Count itemsets by size
        itemset_sizes = frequent_itemsets['itemsets'].apply(len)
        size_counts = itemset_sizes.value_counts().sort_index()
        print("\nItemset size distribution:")
        for size, count in size_counts.items():
            print(f"  - Size {size}: {count} itemsets")
        
        # Generate association rules with a reasonable limit
        print("Generating association rules...")
        
        # Limit the number of frequent itemsets for association rule generation if needed
        if len(frequent_itemsets) > 1000:
            print(f"Limiting to 1000 frequent itemsets for association rule generation (from {len(frequent_itemsets)})")
            # Sort by support value and take the top 1000
            frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(1000)
        
        # Use an even lower confidence threshold to generate more rules
        min_confidence = 0.01  # Further lower from 0.05 to 0.01
        print(f"Using minimum confidence threshold of {min_confidence}")
        
        self.association_rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # Check how many of the rules have a consequent with length 1
        if len(self.association_rules) > 0:
            single_consequent_rules = self.association_rules[self.association_rules['consequents'].map(len) == 1]
            print(f"Generated {len(self.association_rules)} association rules.")
            print(f"  - {len(single_consequent_rules)} rules have a single song as consequent")
            
            # Count rules by antecedent size
            antecedent_sizes = self.association_rules['antecedents'].apply(len)
            size_counts = antecedent_sizes.value_counts().sort_index()
            print("\nAntecedent size distribution:")
            for size, count in size_counts.items():
                print(f"  - Size {size}: {count} rules")
        else:
            print("No rules were generated. Consider using an even lower confidence threshold.")
        
        # Print all rules
        if len(self.association_rules) > 0:
            print("\nAssociation Rules:")
            print("=" * 80)
            print(f"{'Antecedent':<40} | {'Consequent':<40} | Confidence")
            print("-" * 80)
            
            # Sort by confidence for better readability
            sorted_rules = self.association_rules.sort_values(by='confidence', ascending=False)
            
            for idx, rule in sorted_rules.iterrows():
                antecedent = list(rule['antecedents'])
                consequent = list(rule['consequents'])
                confidence = rule['confidence']
                antecedent_str = str(antecedent)[:38]
                consequent_str = str(consequent)[:38]
                print(f"{antecedent_str:<40} | {consequent_str:<40} | {confidence:.3f}")
        else:
            print("No rules were generated. Consider lowering the confidence threshold further.")
    
    def _recommend_from_rules(self, input_songs: List[str], top_n: int = 5) -> List[str]:
        """
        Recommend songs based on association rules.
        
        Args:
            input_songs: List of input song names
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended song names
        """
        if self.association_rules is None or self.association_rules.empty:
            return []
        
        # Convert input songs to frozenset for rule matching
        input_set = frozenset(input_songs)
        
        # Store potential recommendations with their confidence
        recommendations = {}
        matched_rules = []
        
        print("\nMatching rules for input songs:")
        print("=" * 80)
        print(f"Input songs: {input_songs}")
        print("-" * 80)
        
        # Look for rules where the antecedent is a subset of input_songs
        for _, rule in self.association_rules.iterrows():
            antecedent = rule['antecedents']
            consequent = rule['consequents']
            confidence = rule['confidence']
            
            # If the antecedent is a subset of the input songs
            if antecedent.issubset(input_set):
                # For each song in the consequent
                for song in consequent:
                    # Skip if the song is already in the input
                    if song in input_songs:
                        continue
                    
                    # Add or update the recommendation with highest confidence
                    if song not in recommendations or confidence > recommendations[song]:
                        recommendations[song] = confidence
                        
                    # Record this matched rule
                    matched_rules.append((antecedent, song, confidence))
                    
        # Print matched rules
        if matched_rules:
            print(f"Found {len(matched_rules)} matching rules:")
            print(f"{'Antecedent':<40} | {'Consequent':<40} | Confidence")
            print("-" * 80)
            
            # Sort by confidence
            matched_rules.sort(key=lambda x: x[2], reverse=True)
            
            for antecedent, song, confidence in matched_rules:
                antecedent_str = str(list(antecedent))[:38]
                print(f"{antecedent_str:<40} | {song:<40} | {confidence:.3f}")
        else:
            print("No matching rules found for the input songs.")
            
        print("=" * 80)
        
        # Sort recommendations by confidence and return top-n
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop recommendations from rules:")
        if sorted_recommendations:
            for i, (song, confidence) in enumerate(sorted_recommendations[:top_n], 1):
                print(f"{i}. {song} (confidence: {confidence:.3f})")
        else:
            print("No recommendations found from rules.")
            
        return [song for song, _ in sorted_recommendations[:top_n]]
    
    def recommend_songs(self, input_songs: List[str], num_recommendations: int = 5) -> List[str]:
        """
        Generate song recommendations based on the input songs.
        Uses a balanced combination of association rules and name similarity.
        
        Args:
            input_songs: List of input song names
            num_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of recommended song names
        """
        if not input_songs:
            return []
        
        # Limit to a maximum of 10 recommendations
        num_recommendations = min(num_recommendations, 10)
        
        # Get recommendations from association rules
        rule_recommendations = self._recommend_from_rules(input_songs, num_recommendations * 2)
        
        # Get recommendations from name similarity
        similarity_recommendations = self._find_similar_song_names(input_songs, num_recommendations * 2)
        
        # Create balanced combined recommendations
        combined_recommendations = []
        
        # Check if we have recommendations from each method
        has_rule_recs = len(rule_recommendations) > 0
        has_similarity_recs = len(similarity_recommendations) > 0
        
        # Allocate slots based on availability
        if has_rule_recs and has_similarity_recs:
            # We have both types, aim for a 50/50 split
            rule_slots = min(int(num_recommendations * 0.5), len(rule_recommendations))
            similarity_slots = min(int(num_recommendations * 0.5), len(similarity_recommendations))
            
            # Adjust if the total is less than num_recommendations
            remaining_slots = num_recommendations - (rule_slots + similarity_slots)
            if remaining_slots > 0:
                # Allocate remaining slots based on which method has more recommendations
                if len(rule_recommendations) > rule_slots and len(similarity_recommendations) > similarity_slots:
                    # Both have excess, split evenly
                    rule_slots += remaining_slots // 2
                    similarity_slots += remaining_slots - (remaining_slots // 2)
                elif len(rule_recommendations) > rule_slots:
                    # Rules have excess
                    rule_slots += remaining_slots
                else:
                    # Similarity has excess (or neither has excess)
                    similarity_slots += remaining_slots
                    
        elif has_rule_recs:
            # Only rule-based recommendations available
            rule_slots = min(num_recommendations, len(rule_recommendations))
            similarity_slots = 0
        elif has_similarity_recs:
            # Only similarity-based recommendations available
            rule_slots = 0
            similarity_slots = min(num_recommendations, len(similarity_recommendations))
        else:
            # No recommendations from either method
            return []
        
        # Add rule-based recommendations
        for song in rule_recommendations:
            if song not in input_songs and song not in combined_recommendations:
                combined_recommendations.append(song)
                if len(combined_recommendations) >= rule_slots:
                    break
        
        # Add similarity-based recommendations
        for song in similarity_recommendations:
            if song not in input_songs and song not in combined_recommendations:
                combined_recommendations.append(song)
                if len(combined_recommendations) >= rule_slots + similarity_slots:
                    break
        
        # If we still don't have enough recommendations, add any remaining from either source
        if len(combined_recommendations) < num_recommendations:
            # Try to add more from rules first
            for song in rule_recommendations:
                if song not in input_songs and song not in combined_recommendations:
                    combined_recommendations.append(song)
                    if len(combined_recommendations) >= num_recommendations:
                        break
            
            # Then try to add more from similarity
            if len(combined_recommendations) < num_recommendations:
                for song in similarity_recommendations:
                    if song not in input_songs and song not in combined_recommendations:
                        combined_recommendations.append(song)
                        if len(combined_recommendations) >= num_recommendations:
                            break
        
        return combined_recommendations
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to a pickle file.
        
        Args:
            file_path: Path where the model will be saved
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {file_path}")
    
    @staticmethod
    def load_model(file_path: str) -> 'SongRecommender':
        """
        Load a trained model from a pickle file.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            Loaded SongRecommender instance
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {file_path}")
        return model


# Example usage
if __name__ == "__main__":
    # Create a new recommender
    recommender = SongRecommender()
    
    # Load and prepare data - using separate files for playlists and unique songs
    recommender.load_and_prepare_data(
        playlist_file="data/2023_spotify_ds1.csv",
        songs_file="data/2023_spotify_songs.csv"
    )
    
    # Mine frequent itemsets
    recommender.mine_frequent_itemsets(max_playlists=500, max_songs_per_playlist=20)
    
    # Example recommendation
    input_songs = ["Ride Wit Me", "Sweet Emotion"]
    recommendations = recommender.recommend_songs(input_songs, 5)
    
    print(f"Input songs: {input_songs}")
    print(f"Recommendations: {recommendations}")
    
    # Save the model
    recommender.save_model("models/song_recommender.pkl") 