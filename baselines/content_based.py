#!/usr/bin/env python3
"""
Content-Based Recommendation Baseline
Uses item metadata and text features for recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ContentBasedRecommender:
    """Content-based recommendation using TF-IDF and item features"""

    def __init__(self, interactions_df: pd.DataFrame, metadata_df: pd.DataFrame = None):
        """
        Initialize content-based recommender

        Args:
            interactions_df: DataFrame with user-item interactions
            metadata_df: Optional DataFrame with item metadata
        """
        self.interactions_df = interactions_df
        self.metadata_df = metadata_df
        self.tfidf_vectorizer = None
        self.item_features = None
        self.item_to_idx = {}
        self.user_profiles = {}

        # Add fallback mechanism
        self.item_similarities = None
        self.popular_items = None

        print(f"📊 Initializing Content-Based Recommender")
        print(f"   Interactions: {len(interactions_df)}")
        print(f"   Users: {interactions_df['reviewerID'].nunique()}")
        print(f"   Items: {interactions_df['asin'].nunique()}")
        if metadata_df is not None:
            print(f"   Metadata items: {len(metadata_df)}")

        self._build_content_features()
        self._build_user_profiles()
        self._build_fallback_systems()

    def _clean_text(self, text: str) -> str:
        """Clean text for feature extraction"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _build_content_features(self):
        """Build TF-IDF features from item content"""
        print("🔧 Building content features...")

        # Collect all unique items
        all_items = self.interactions_df['asin'].unique()
        content_texts = []
        valid_items = []

        for item_id in all_items:
            text_parts = []

            # Get text from interactions (reviews)
            item_reviews = self.interactions_df[self.interactions_df['asin'] == item_id]

            # Use review summaries and text
            if 'summary' in item_reviews.columns:
                summaries = item_reviews['summary'].dropna().astype(str)
                text_parts.extend([self._clean_text(s) for s in summaries])

            if 'reviewText' in item_reviews.columns:
                reviews = item_reviews['reviewText'].dropna().astype(str)
                # Use first 100 chars of each review to avoid overwhelming
                text_parts.extend([self._clean_text(r[:100]) for r in reviews])

            # Add metadata if available
            if self.metadata_df is not None and item_id in self.metadata_df['asin'].values:
                meta_row = self.metadata_df[self.metadata_df['asin'] == item_id].iloc[0]

                if 'title' in meta_row and pd.notna(meta_row['title']):
                    # Give title more weight by repeating it
                    title_text = self._clean_text(meta_row['title'])
                    text_parts.extend([title_text] * 3)

                if 'description' in meta_row and pd.notna(meta_row['description']):
                    desc_text = self._clean_text(str(meta_row['description'])[:200])
                    text_parts.append(desc_text)

                if 'category' in meta_row and pd.notna(meta_row['category']):
                    cat_text = self._clean_text(str(meta_row['category']))
                    text_parts.extend([cat_text] * 2)  # Category is important

            # Combine all text for this item
            combined_text = ' '.join(text_parts)

            if combined_text.strip():
                content_texts.append(combined_text)
                valid_items.append(item_id)

        print(f"   Built content for {len(valid_items)} items")

        if content_texts and len(content_texts) > 1:
            try:
                # Build TF-IDF matrix with more lenient parameters
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(content_texts) * 10),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=max(1, min(2, len(content_texts) // 20)),
                    max_df=0.9
                )

                self.item_features = self.tfidf_vectorizer.fit_transform(content_texts)
                self.item_to_idx = {item: idx for idx, item in enumerate(valid_items)}

                print(f"   TF-IDF features: {self.item_features.shape}")

                # Pre-compute item similarities for faster recommendations
                if self.item_features.shape[0] > 1:
                    print("   Computing item similarities...")
                    self.item_similarities = cosine_similarity(self.item_features)

            except Exception as e:
                print(f"   ⚠️ Error building TF-IDF: {e}")
                self.item_features = None
        else:
            print("   ⚠️ Insufficient content for TF-IDF")

    def _build_user_profiles(self):
        """Build user profiles from their interaction history"""
        print("👤 Building user profiles...")

        if self.item_features is None:
            print("   ⚠️ No item features available")
            return

        successful_profiles = 0

        for user_id in self.interactions_df['reviewerID'].unique():
            user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]

            # Get items user rated highly (>= 4.0)
            liked_items = user_items[user_items['overall'] >= 4.0]['asin'].tolist()

            # Get feature vectors for liked items
            liked_indices = [self.item_to_idx[item] for item in liked_items
                             if item in self.item_to_idx]

            if liked_indices:
                try:
                    # Average of liked item features
                    user_features = self.item_features[liked_indices]
                    user_profile = np.mean(user_features, axis=0)

                    # Convert to 1D array if needed
                    if hasattr(user_profile, 'A1'):
                        user_profile = user_profile.A1

                    self.user_profiles[user_id] = user_profile
                    successful_profiles += 1
                except Exception as e:
                    # Skip this user if there's an error
                    continue

        print(
            f"   Built profiles for {successful_profiles} users (out of {self.interactions_df['reviewerID'].nunique()})")

    def _build_fallback_systems(self):
        """Build fallback recommendation systems"""
        print("🔄 Building fallback systems...")

        # Build item popularity ranking
        item_stats = self.interactions_df.groupby('asin').agg({
            'overall': ['count', 'mean'],
            'reviewerID': 'nunique'
        }).round(3)

        item_stats.columns = ['interaction_count', 'avg_rating', 'unique_users']

        # Popularity score
        item_stats['score'] = (
                np.log1p(item_stats['interaction_count']) *
                item_stats['avg_rating'] *
                np.log1p(item_stats['unique_users'])
        )

        self.popular_items = item_stats.sort_values('score', ascending=False).reset_index()
        print(f"   Built popularity fallback with {len(self.popular_items)} items")

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> List[str]:
        """Generate recommendations for a user"""

        # Try content-based recommendation first
        if user_id in self.user_profiles and self.item_features is not None:
            try:
                return self._content_based_recommend(user_id, k, exclude_seen)
            except Exception as e:
                print(f"   ⚠️ Content-based failed for {user_id}: {e}")

        # Fallback to item-based collaborative filtering
        if self.item_similarities is not None:
            try:
                return self._item_based_recommend(user_id, k, exclude_seen)
            except Exception as e:
                print(f"   ⚠️ Item-based failed for {user_id}: {e}")

        # Final fallback to popularity
        return self._popularity_recommend(user_id, k, exclude_seen)

    def _content_based_recommend(self, user_id: str, k: int, exclude_seen: bool) -> List[str]:
        """Content-based recommendation using user profile"""
        user_profile = self.user_profiles[user_id]

        # Calculate similarities with all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]

        # Get top-k similar items
        item_indices = np.argsort(similarities)[::-1]

        recommendations = []
        seen_items = set()

        if exclude_seen:
            user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]
            seen_items = set(user_items['asin'].tolist())

        # Get recommendations
        for idx in item_indices:
            if len(recommendations) >= k:
                break

            item_id = list(self.item_to_idx.keys())[list(self.item_to_idx.values()).index(idx)]

            if item_id not in seen_items:
                recommendations.append(item_id)

        return recommendations

    def _item_based_recommend(self, user_id: str, k: int, exclude_seen: bool) -> List[str]:
        """Item-based collaborative filtering recommendation"""
        user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]

        # Get user's highly rated items
        liked_items = user_items[user_items['overall'] >= 4.0]['asin'].tolist()
        liked_indices = [self.item_to_idx[item] for item in liked_items if item in self.item_to_idx]

        if not liked_indices:
            return self._popularity_recommend(user_id, k, exclude_seen)

        # Find similar items to user's liked items
        item_scores = defaultdict(float)

        for liked_idx in liked_indices:
            # Get similarities for this liked item
            similarities = self.item_similarities[liked_idx]

            # Add weighted scores for similar items
            for item_idx, sim_score in enumerate(similarities):
                if item_idx != liked_idx and sim_score > 0.1:  # Threshold for similarity
                    item_id = list(self.item_to_idx.keys())[item_idx]
                    item_scores[item_id] += sim_score

        # Sort by score and get top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        seen_items = set()

        if exclude_seen:
            seen_items = set(user_items['asin'].tolist())

        for item_id, score in sorted_items:
            if len(recommendations) >= k:
                break

            if item_id not in seen_items:
                recommendations.append(item_id)

        return recommendations

    def _popularity_recommend(self, user_id: str, k: int, exclude_seen: bool) -> List[str]:
        """Popularity-based recommendation as final fallback"""
        seen_items = set()

        if exclude_seen and user_id:
            user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]
            seen_items = set(user_items['asin'].tolist())

        recommendations = []
        for _, row in self.popular_items.iterrows():
            if len(recommendations) >= k:
                break

            item_id = row['asin']
            if item_id not in seen_items:
                recommendations.append(item_id)

        return recommendations


class PopularityRecommender:
    """Popularity-based recommendation baseline"""

    def __init__(self, interactions_df: pd.DataFrame):
        """Initialize popularity recommender"""
        self.interactions_df = interactions_df
        self.popular_items = self._build_popularity_ranking()

        print(f"📈 Popularity Recommender initialized")
        print(f"   Most popular item has {self.popular_items.iloc[0]['score']:.2f} popularity score")

    def _build_popularity_ranking(self) -> pd.DataFrame:
        """Build popularity ranking of items"""
        # Calculate popularity score combining frequency and average rating
        item_stats = self.interactions_df.groupby('asin').agg({
            'overall': ['count', 'mean'],
            'reviewerID': 'nunique'
        }).round(3)

        item_stats.columns = ['interaction_count', 'avg_rating', 'unique_users']

        # Popularity score: log(interactions) * avg_rating * log(unique_users)
        item_stats['score'] = (
                np.log1p(item_stats['interaction_count']) *
                item_stats['avg_rating'] *
                np.log1p(item_stats['unique_users'])
        )

        return item_stats.sort_values('score', ascending=False).reset_index()

    def recommend(self, user_id: str = None, k: int = 10, exclude_seen: bool = True) -> List[str]:
        """Generate popular item recommendations"""

        seen_items = set()
        if exclude_seen and user_id:
            user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]
            seen_items = set(user_items['asin'].tolist())

        recommendations = []
        for _, row in self.popular_items.iterrows():
            if len(recommendations) >= k:
                break

            item_id = row['asin']
            if item_id not in seen_items:
                recommendations.append(item_id)

        return recommendations


class RandomRecommender:
    """Random recommendation baseline"""

    def __init__(self, interactions_df: pd.DataFrame):
        """Initialize random recommender"""
        # Fix: Store the interactions_df as an instance variable
        self.interactions_df = interactions_df
        self.all_items = interactions_df['asin'].unique().tolist()
        print(f"🎲 Random Recommender initialized with {len(self.all_items)} items")

    def recommend(self, user_id: str = None, k: int = 10, exclude_seen: bool = True) -> List[str]:
        """Generate random recommendations"""

        available_items = self.all_items.copy()

        if exclude_seen and user_id:
            # Now self.interactions_df is available
            user_items = self.interactions_df[self.interactions_df['reviewerID'] == user_id]
            seen_items = set(user_items['asin'].tolist())
            available_items = [item for item in available_items if item not in seen_items]

        if len(available_items) < k:
            return available_items

        return np.random.choice(available_items, size=k, replace=False).tolist()


def evaluate_baseline_methods(interactions_path: str, user_history_path: str, domain: str,
                              metadata_path: str = None, k: int = 3):
    """Evaluate all baseline methods"""

    print("🔬 COMPREHENSIVE BASELINE EVALUATION")
    print("=" * 60)

    # Load data
    print("📁 Loading data...")
    interactions_df = pd.read_csv(interactions_path)

    metadata_df = None
    if metadata_path and os.path.exists(metadata_path):
        try:
            metadata_df = pd.read_csv(metadata_path)
            print(f"✅ Loaded metadata: {len(metadata_df)} items")
        except Exception as e:
            print(f"⚠️ Could not load metadata: {e}")

    with open(user_history_path, 'r') as f:
        user_history = json.load(f)

    print(f"✅ Data loaded successfully")
    print(f"   Interactions: {len(interactions_df)}")
    print(f"   Users: {interactions_df['reviewerID'].nunique()}")
    print(f"   Items: {interactions_df['asin'].nunique()}")

    # Initialize all recommenders
    print("\n🤖 Initializing baseline methods...")

    recommenders = {
        'Content-Based': ContentBasedRecommender(interactions_df, metadata_df),
        'Popularity': PopularityRecommender(interactions_df),
        'Random': RandomRecommender(interactions_df)
    }

    # Get item titles for results
    asin2title = {}
    if metadata_df is not None:
        asin2title = {row['asin']: row['title'] for _, row in metadata_df.iterrows()
                      if pd.notnull(row.get('title'))}

    # Test users (users with sufficient data in the domain)
    test_users = []
    for user_id, domains in user_history.items():
        if (domain in domains and
                domains[domain].get('count', 0) >= 3 and
                len(domains[domain].get('liked', [])) >= 2):
            test_users.append(user_id)

    print(f"\n👥 Testing on {len(test_users)} users with sufficient {domain} data")

    # Evaluate each method
    all_results = {}

    for method_name, recommender in recommenders.items():
        print(f"\n🧪 Evaluating {method_name}...")

        method_results = []
        successful = 0

        for i, user_id in enumerate(test_users):
            if i % 50 == 0 and i > 0:
                print(f"   Progress: {i}/{len(test_users)} users")

            try:
                # Generate recommendations
                rec_items = recommender.recommend(user_id, k=k, exclude_seen=True)

                if rec_items:
                    # Format recommendations with titles
                    rec_titles = []
                    for item_id in rec_items:
                        title = asin2title.get(item_id, f"Item_{item_id}")
                        rec_titles.append(title)

                    # Create recommendation text
                    rec_text = "\n".join([
                        f"{i + 1}. {title} - {method_name} recommendation"
                        for i, title in enumerate(rec_titles)
                    ])

                    method_results.append({
                        "user_id": user_id,
                        "source_domain": domain,
                        "target_domain": domain,
                        "recommendations": rec_text,
                        "success": True,
                        "model_type": method_name,
                        "timestamp": pd.Timestamp.now().isoformat()
                    })
                    successful += 1

            except Exception as e:
                print(f"   ⚠️ Error for user {user_id}: {e}")

        print(f"   ✅ {method_name}: {successful}/{len(test_users)} successful recommendations")
        all_results[method_name] = method_results

    # Save results for each method
    print(f"\n💾 Saving results...")
    os.makedirs("results", exist_ok=True)

    saved_files = []
    for method_name, results in all_results.items():
        if results:
            filename = f"results/{method_name.lower().replace('-', '_')}_results.json"

            with open(filename, 'w') as f:
                json.dump({
                    "model_info": {
                        "type": method_name,
                        "domain": domain,
                        "k": k,
                        "users_tested": len(test_users),
                        "recommendations_generated": len(results)
                    },
                    "results": results
                }, f, indent=2)

            saved_files.append(filename)
            print(f"   💾 {method_name}: {filename}")

    return saved_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate baseline recommendation methods")
    parser.add_argument("--dataset", required=True, help="Path to interactions CSV")
    parser.add_argument("--user_history", required=True, help="Path to user history JSON")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument("--metadata", help="Path to metadata CSV (optional)")
    parser.add_argument("--k", type=int, default=3, help="Top-K recommendations")

    args = parser.parse_args()

    # Set metadata path if not provided
    if not args.metadata:
        base_dir = os.path.dirname(os.path.dirname(args.dataset))
        args.metadata = os.path.join(base_dir, "data", "processed", f"{args.domain}_metadata.csv")

    evaluate_baseline_methods(
        args.dataset,
        args.user_history,
        args.domain,
        args.metadata,
        args.k
    )