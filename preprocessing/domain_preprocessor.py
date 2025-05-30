import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging
from tqdm import tqdm
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossDomainPreprocessor:
    """
    Preprocesses data across multiple domains for cross-domain recommendation.

    This class handles the preprocessing of user interaction data across different domains,
    identifying overlapping users, filtering active users and items, and generating
    cross-domain user profiles for recommendation tasks.
    """

    def __init__(self, base_path: str, output_path: str, min_interactions: int = 5, min_domains: int = 2):
        """
        Initialize the cross-domain preprocessor.

        Args:
            base_path: Path to directory containing processed CSV files
            output_path: Path to save the filtered data and user history
            min_interactions: Minimum number of interactions per user/item to be considered active
            min_domains: Minimum number of domains a user must be active in (default: 2)
        """
        self.base_path = base_path
        self.output_path = output_path
        self.min_interactions = min_interactions
        self.min_domains = min_domains
        self.domains = ['Books', 'Movies_and_TV', 'CDs', 'Digital_Music']

        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Initialize data cache
        self.data_cache = {}
        self.metadata_cache = {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load review data for all domains.

        Returns:
            Dictionary mapping domain names to DataFrames containing review data
        """
        data = {}

        for domain in self.domains:
            file_path = os.path.join(self.base_path, f"{domain}_reviews.csv")
            if os.path.exists(file_path):
                try:
                    data[domain] = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(data[domain])} records for {domain}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")

        return data

    def load_metadata(self) -> Dict[str, pd.DataFrame]:
        """
        Load metadata for all domains.

        Returns:
            Dictionary mapping domain names to DataFrames containing metadata
        """
        metadata = {}

        for domain in self.domains:
            file_path = os.path.join(self.base_path, f"{domain}_metadata.csv")
            if os.path.exists(file_path):
                try:
                    metadata[domain] = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(metadata[domain])} metadata records for {domain}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"Metadata file not found: {file_path}")

        return metadata

    def filter_active_users_and_items(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter active users and items based on minimum interaction threshold.

        Args:
            data: Dictionary mapping domain names to DataFrames

        Returns:
            Dictionary mapping domain names to filtered DataFrames
        """
        filtered_data = {}

        for domain, df in data.items():
            if df.empty:
                logger.warning(f"Empty DataFrame for {domain}, skipping")
                continue

            logger.info(f"Filtering {domain} for active users and items...")

            # Filter active users
            user_counts = df['reviewerID'].value_counts()
            active_users = user_counts[user_counts >= self.min_interactions].index
            df_filtered = df[df['reviewerID'].isin(active_users)]

            # Filter active items
            item_counts = df_filtered['asin'].value_counts()
            active_items = item_counts[item_counts >= self.min_interactions].index
            df_filtered = df_filtered[df_filtered['asin'].isin(active_items)]

            filtered_data[domain] = df_filtered

            logger.info(f"Filtered {domain}: {len(df_filtered)} interactions, "
                        f"{len(active_users)} active users, {len(active_items)} active items")

        return filtered_data

    def find_overlapping_users(self, data: Dict[str, pd.DataFrame]) -> Set[str]:
        """
        Find users who are active in at least min_domains domains.

        Args:
            data: Dictionary mapping domain names to DataFrames

        Returns:
            Set of user IDs that have interactions in at least min_domains domains
        """
        if not data:
            logger.warning("No data provided to find overlapping users")
            return set()

        # Count domains per user
        user_domain_counts = defaultdict(int)

        for domain, df in data.items():
            if df.empty:
                continue

            for user_id in df['reviewerID'].unique():
                user_domain_counts[user_id] += 1

        # Filter users who appear in at least min_domains
        overlapping_users = {user for user, count in user_domain_counts.items()
                             if count >= self.min_domains}

        logger.info(f"Found {len(overlapping_users)} users active in at least {self.min_domains} domains")

        # Log domain distribution
        domain_distribution = defaultdict(int)
        for user, count in user_domain_counts.items():
            if user in overlapping_users:
                domain_distribution[count] += 1

        for count, num_users in sorted(domain_distribution.items()):
            logger.info(f"  Users active in {count} domains: {num_users}")

        return overlapping_users

    def save_filtered_data(self, data: Dict[str, pd.DataFrame], overlapping_users: Set[str]) -> None:
        """
        Save filtered data containing only overlapping users.

        Args:
            data: Dictionary mapping domain names to DataFrames
            overlapping_users: Set of user IDs to include
        """
        for domain, df in data.items():
            if df.empty:
                logger.warning(f"No data to save for {domain}")
                continue

            # Filter to only include overlapping users
            df_overlapping = df[df['reviewerID'].isin(overlapping_users)]

            # Skip if no users remain
            if df_overlapping.empty:
                logger.warning(f"No overlapping users found in {domain}, skipping")
                continue

            # Save to file
            save_path = os.path.join(self.output_path, f"{domain}_filtered.csv")
            df_overlapping.to_csv(save_path, index=False)

            logger.info(f"Saved {len(df_overlapping)} filtered records for {domain} to {save_path}")

            # Generate statistics
            user_count = df_overlapping['reviewerID'].nunique()
            item_count = df_overlapping['asin'].nunique()
            avg_rating = df_overlapping['overall'].mean()

            logger.info(f"{domain} statistics: {user_count} users, {item_count} items, "
                        f"average rating: {avg_rating:.2f}")

    def generate_user_history(self, data: Dict[str, pd.DataFrame], overlapping_users: Set[str]) -> None:
        """
        Generate user history JSON with preferences across domains.

        Args:
            data: Dictionary mapping domain names to DataFrames
            overlapping_users: Set of user IDs to include
        """
        user_history = {}
        metadata = self.load_metadata()

        # Process each domain
        for domain, df in data.items():
            if df.empty:
                logger.warning(f"No data for {domain}, skipping user history generation")
                continue

            logger.info(f"Generating user history for {domain}...")

            # Filter to overlapping users
            df = df[df['reviewerID'].isin(overlapping_users)]

            if df.empty:
                logger.warning(f"No overlapping users found in {domain}, skipping")
                continue

            # Try to load metadata for this domain
            domain_meta_df = None
            if domain in metadata and not metadata[domain].empty:
                domain_meta_df = metadata[domain]

            # Create user history for this domain
            for user_id, group in tqdm(df.groupby('reviewerID'),
                                       desc=f"Processing {domain} users"):
                if user_id not in user_history:
                    user_history[user_id] = {}

                # Process items with metadata if available
                processed_items = []
                for _, row in group.iterrows():
                    item_info = {
                        'asin': row['asin'],
                        'rating': row['overall']
                    }

                    # Add title from metadata if available
                    if domain_meta_df is not None:
                        meta_row = domain_meta_df[domain_meta_df['asin'] == row['asin']]
                        if not meta_row.empty and 'title' in meta_row.columns:
                            item_info['title'] = meta_row.iloc[0]['title']
                        else:
                            item_info['title'] = row['asin']  # Fallback to ASIN
                    else:
                        item_info['title'] = row['asin']  # Fallback to ASIN

                    processed_items.append(item_info)

                # Separate into liked and disliked
                liked = [item for item in processed_items if item['rating'] >= 4]
                disliked = [item for item in processed_items if item['rating'] < 4]

                user_history[user_id][domain] = {
                    'liked': liked,
                    'disliked': disliked,
                    'count': len(processed_items)
                }

        # Save user history to JSON file
        history_path = os.path.join(self.output_path, 'user_history.json')
        with open(history_path, 'w') as f:
            json.dump(user_history, f, indent=2)

        logger.info(f"Saved user history for {len(user_history)} users to {history_path}")

        # Generate domain overlap statistics
        self._generate_domain_overlap_stats(user_history)

    def _generate_domain_overlap_stats(self, user_history: Dict) -> None:
        """
        Generate statistics about domain overlap in user history.

        Args:
            user_history: Dictionary mapping user IDs to domain preferences
        """
        domain_counts = defaultdict(int)
        domain_pairs = defaultdict(int)

        for user_id, domains in user_history.items():
            # Count users per domain
            for domain in domains:
                if domains[domain]['count'] > 0:  # Only count if user has items
                    domain_counts[domain] += 1

            # Count domain pairs
            domains_with_items = [d for d in domains if domains[d]['count'] > 0]
            for i, domain1 in enumerate(domains_with_items):
                for domain2 in domains_with_items[i + 1:]:
                    pair = tuple(sorted([domain1, domain2]))
                    domain_pairs[pair] += 1

        # Save statistics
        stats = {
            "users_per_domain": dict(domain_counts),
            "domain_pair_overlaps": {f"{d1}_{d2}": count for (d1, d2), count in domain_pairs.items()}
        }

        stats_path = os.path.join(self.output_path, 'domain_overlap_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved domain overlap statistics to {stats_path}")

        # Log summary
        logger.info("Domain user counts:")
        for domain, count in domain_counts.items():
            logger.info(f"  {domain}: {count} users")

        logger.info("Domain pair overlaps:")
        for (d1, d2), count in sorted(domain_pairs.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {d1} - {d2}: {count} users")

    def generate_cold_warm_splits(self, data: Dict[str, pd.DataFrame], overlapping_users: Set[str]) -> None:
        """
        Generate cold and warm scenario splits based on interaction counts.

        Args:
            data: Dictionary mapping domain names to DataFrames
            overlapping_users: Set of user IDs to include
        """
        # Count total interactions per user across all domains
        user_interaction_counts = defaultdict(int)

        for domain, df in data.items():
            if df.empty:
                continue

            df_filtered = df[df['reviewerID'].isin(overlapping_users)]
            for user_id, count in df_filtered.groupby('reviewerID').size().items():
                user_interaction_counts[user_id] += count

        # Define cold and warm thresholds
        cold_threshold = 50  # Users with fewer than 50 interactions
        warm_threshold = 200  # Users with more than 200 interactions (lowered from 2000)

        # Identify cold and warm users
        cold_users = {user for user, count in user_interaction_counts.items() if count < cold_threshold}
        warm_users = {user for user, count in user_interaction_counts.items() if count > warm_threshold}

        # Save user sets
        cold_warm_path = os.path.join(self.output_path, 'cold_warm_users.json')
        with open(cold_warm_path, 'w') as f:
            json.dump({
                'cold_users': list(cold_users),
                'warm_users': list(warm_users),
                'user_interaction_counts': {user: count for user, count in user_interaction_counts.items()}
            }, f, indent=2)

        logger.info(f"Identified {len(cold_users)} cold users (< {cold_threshold} interactions)")
        logger.info(f"Identified {len(warm_users)} warm users (> {warm_threshold} interactions)")
        logger.info(f"Saved cold/warm user lists to {cold_warm_path}")

    def run(self):
        """
        Run the complete preprocessing pipeline.
        """
        logger.info("Starting cross-domain preprocessing pipeline")

        # Step 1: Load all domain data
        data = self.load_data()
        if not data:
            logger.error("No data found. Aborting preprocessing.")
            return

        # Step 2: Filter active users and items
        filtered_data = self.filter_active_users_and_items(data)

        # Check if we have any valid data after filtering
        valid_domains = [domain for domain, df in filtered_data.items() if not df.empty]
        if len(valid_domains) < self.min_domains:
            logger.error(f"Found valid data in only {len(valid_domains)} domains, "
                         f"which is less than required minimum of {self.min_domains}. "
                         f"Aborting preprocessing.")
            return

        logger.info(f"Continuing with valid domains: {', '.join(valid_domains)}")

        # Step 3: Find overlapping users across at least min_domains domains
        overlapping_users = self.find_overlapping_users(filtered_data)
        if not overlapping_users:
            logger.error("No overlapping users found. Aborting preprocessing.")
            return

        # Step 4: Save filtered data with only overlapping users
        self.save_filtered_data(filtered_data, overlapping_users)

        # Step 5: Generate user history JSON
        self.generate_user_history(filtered_data, overlapping_users)

        # Step 6: Generate cold/warm user splits
        self.generate_cold_warm_splits(filtered_data, overlapping_users)

        logger.info("Cross-domain preprocessing completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Preprocessor")
    parser.add_argument("--base_path", type=str, default="data/processed",
                        help="Path to processed CSV files")
    parser.add_argument("--output_path", type=str, default="data/splits",
                        help="Path to save filtered data and user history")
    parser.add_argument("--min_interactions", type=int, default=50,
                        help="Minimum interactions per user/item")
    parser.add_argument("--min_domains", type=int, default=2,
                        help="Minimum number of domains a user must be active in")

    args = parser.parse_args()

    processor = CrossDomainPreprocessor(
        base_path=args.base_path,
        output_path=args.output_path,
        min_interactions=args.min_interactions,
        min_domains=args.min_domains
    )

    processor.run()