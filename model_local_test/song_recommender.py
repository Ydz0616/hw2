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
import os
from multiprocessing import Pool, cpu_count


class SongRecommender:
    """
    A song recommendation system that combines Apriori association rules and NLP-based similarity.
    """
    
    def __init__(self):
        # For frequent itemset mining
        self.playlist_songs = defaultdict(list)
        self.song_playlists = defaultdict(list)
        self.freq_itemsets = []
        self.association_rules = []
        self.min_support = 0.01  # Default value, can be adjusted
        self.min_confidence = 0.01  # Default value, can be adjusted
        
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
    
    def mine_frequent_itemsets(self, use_sampling=False, sample_ratio=0.3) -> None:
        """
        Apply Apriori algorithm to discover association rules among songs.
        Uses ALL playlists by default, with optimizations for performance.
        
        Args:
            use_sampling: If True, use random sampling to improve performance (default: False)
            sample_ratio: Ratio of playlists to sample if sampling is enabled (default: 0.3)
        """
        print("Preparing data for Apriori mining...")
        playlist_ids = list(self.playlist_songs.keys())
        total_playlists = len(playlist_ids)
        
        # Optional sampling for faster processing during development/testing
        if use_sampling and total_playlists > 1000:
            sample_size = max(int(total_playlists * sample_ratio), 1000)
            print(f"Using sampling: {sample_size} playlists from {total_playlists} total (ratio: {sample_ratio})")
            playlist_ids = random.sample(playlist_ids, sample_size)
        else:
            print(f"Processing all {total_playlists} playlists...")
        
        # Create transactions more efficiently
        transactions = [self.playlist_songs[pid] for pid in playlist_ids]
        
        # For large datasets, process in batches
        if len(transactions) > 10000:
            print(f"Large dataset detected. Processing in batches...")
            self._process_large_dataset(transactions)
        else:
            # For smaller datasets, process all at once
            print(f"Running Apriori on {len(transactions)} playlists...")
            self._process_transactions(transactions)
        
        print(f"Found {len(self.freq_itemsets)} frequent itemsets and {len(self.association_rules)} rules.")
        
        # Print memory usage statistics
        import sys
        try:
            itemsets_size = sys.getsizeof(self.freq_itemsets) / (1024 * 1024)
            rules_size = sys.getsizeof(self.association_rules) / (1024 * 1024)
            print(f"Memory usage - Itemsets: {itemsets_size:.2f} MB, Rules: {rules_size:.2f} MB")
        except Exception as e:
            print(f"Error calculating memory usage: {e}")
    
    def _process_transactions(self, transactions):
        """Process a batch of transactions using Apriori algorithm."""
        try:
            # Convert transactions to one-hot encoded DataFrame
            te = TransactionEncoder()
            te_ary = te.fit_transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Apply Apriori algorithm
            print(f"Running Apriori with min_support={self.min_support}...")
            frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                print("No frequent itemsets found. Try lowering the support threshold.")
                self.freq_itemsets = []
                self.association_rules = []
                return
            
            # Generate association rules
            print(f"Generating association rules with min_confidence={self.min_confidence}...")
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                     min_threshold=self.min_confidence)
            
            if rules.empty:
                print("No rules generated. Try lowering the confidence threshold.")
                # Store the frequent itemsets but no rules
                self._convert_apriori_results(frequent_itemsets, pd.DataFrame())
            else:
                # Store both frequent itemsets and rules
                self._convert_apriori_results(frequent_itemsets, rules)
                
        except Exception as e:
            print(f"Error in Apriori processing: {e}")
            import traceback
            traceback.print_exc()
            self.freq_itemsets = []
            self.association_rules = []
    
    def _process_large_dataset(self, transactions):
        """Process a large dataset in batches using multiprocessing."""
        # Determine optimal batch size and number of processes
        num_processes = min(cpu_count(), 4)  # Limit to 4 processes to avoid memory issues
        batch_size = max(1000, len(transactions) // (num_processes * 2))
        num_batches = (len(transactions) + batch_size - 1) // batch_size
        
        print(f"Processing in {num_batches} batches with {num_processes} processes...")
        
        # Split transactions into batches
        batches = [transactions[i:i+batch_size] for i in range(0, len(transactions), batch_size)]
        
        # Process first batch to get all unique items
        print("Processing first batch to identify all unique items...")
        all_items = set()
        for transaction in batches[0]:
            all_items.update(transaction)
        
        # Add some items from other batches to ensure coverage
        for batch in batches[1:]:
            sample_size = min(len(batch), 10)
            for transaction in random.sample(batch, sample_size):
                all_items.update(transaction)
        
        print(f"Identified {len(all_items)} unique items across batches")
        
        # Process each batch sequentially (Apriori doesn't parallelize well due to memory constraints)
        all_frequent_itemsets = []
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")
            
            # Convert batch to one-hot encoded DataFrame
            te = TransactionEncoder()
            te_ary = te.fit_transform(batch)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Apply Apriori algorithm
            try:
                frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)
                if not frequent_itemsets.empty:
                    all_frequent_itemsets.append(frequent_itemsets)
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
        
        # Combine results
        if not all_frequent_itemsets:
            print("No frequent itemsets found across all batches.")
            self.freq_itemsets = []
            self.association_rules = []
            return
        
        # Combine all frequent itemsets
        combined_itemsets = pd.concat(all_frequent_itemsets).drop_duplicates()
        
        # Recalculate support values for the combined dataset
        # This is an approximation since we're processing in batches
        combined_itemsets['support'] = combined_itemsets['support'] / len(batches)
        
        # Generate association rules from the combined frequent itemsets
        print("Generating association rules from combined frequent itemsets...")
        try:
            rules = association_rules(combined_itemsets, metric="confidence", 
                                     min_threshold=self.min_confidence)
            self._convert_apriori_results(combined_itemsets, rules)
        except Exception as e:
            print(f"Error generating rules: {e}")
            self._convert_apriori_results(combined_itemsets, pd.DataFrame())
    
    def _convert_apriori_results(self, frequent_itemsets, rules):
        """Convert Apriori results to the format used by the recommender."""
        # Convert frequent itemsets to the format used by the recommender
        self.freq_itemsets = []
        for _, row in frequent_itemsets.iterrows():
            itemset = tuple(row['itemsets'])
            support = row['support']
            self.freq_itemsets.append((itemset, support))
        
        # Convert association rules to the format used by the recommender
        self.association_rules = []
        for _, row in rules.iterrows():
            antecedent = tuple(row['antecedents'])
            consequent = tuple(row['consequents'])
            confidence = row['confidence']
            self.association_rules.append((antecedent, consequent, confidence))
    
    def _recommend_from_rules(self, input_songs: List[str], top_n: int = 5) -> List[str]:
        """
        Recommend songs based on association rules.
        
        Args:
            input_songs: List of input song names
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended song names
        """
        if not self.association_rules:
            return []
        
        # Convert input songs to set for faster lookup
        input_set = set(input_songs)
        
        # Store potential recommendations with their confidence
        recommendations = {}
        matched_rules = []
        
        print("\nMatching rules for input songs:")
        print("=" * 80)
        print(f"Input songs: {input_songs}")
        print("-" * 80)
        
        # Look for rules where the antecedent is a subset of input_songs
        for antecedent, consequent, confidence in self.association_rules:
            # Convert items to sets for subset checking
            antecedent_set = set(antecedent)
            
            # If the antecedent is a subset of the input songs
            if antecedent_set.issubset(input_set):
                # For each song in the consequent that's not in the input
                for song in consequent:
                    if song not in input_songs:
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
            
            for antecedent, song, confidence in matched_rules[:10]:  # Show top 10 matches
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
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
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