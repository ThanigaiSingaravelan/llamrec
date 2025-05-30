import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from typing import Dict, List, Set, Optional, Union
import logging
import json
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Creates various data splits for recommendation model training and evaluation.

    This class provides functionality to create different types of dataset splits:
    1. Chronological train/test splits (based on time if available)
    2. Cold-start splits (simulating new users with limited history)
    3. K-fold cross-validation splits
    4. Cross-domain evaluation splits

    It handles the complexities of maintaining user history consistency across splits
    and creating appropriate evaluation scenarios.
    """

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 test_ratio: float = 0.2,
                 cold_threshold: int = 5,
                 n_splits: int = 5,
                 random_seed: int = 42):
        """
        Initialize the data splitter.

        Args:
            input_path: Path to filtered data
            output_path: Path to save split data
            test_ratio: Ratio of data to use for testing
            cold_threshold: Maximum number of interactions for cold-start training
            n_splits: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
        """
        self.input_path = input_path
        self.output_path = output_path
        self.test_ratio = test_ratio
        self.cold_threshold = cold_threshold
        self.n_splits = n_splits
        self.random_seed = random_seed

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

    def load_filtered_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load filtered data for all domains.

        Returns:
            Dictionary mapping domain names to DataFrames
        """
        data = {}

        for file in os.listdir(self.input_path):
            if file.endswith('_filtered.csv'):
                domain = file.replace('_filtered.csv', '')
                file_path = os.path.join(self.input_path, file)

                try:
                    df = pd.read_csv(file_path)
                    data[domain] = df
                    logger.info(f"Loaded {len(df)} records for {domain}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        return data

    def load_user_history(self) -> Optional[Dict]:
        """
        Load user history JSON if available.

        Returns:
            Dictionary containing user history or None if not found
        """
        history_path = os.path.join(self.input_path, 'user_history.json')

        if not os.path.exists(history_path):
            logger.warning(f"User history file not found: {history_path}")
            return None

        try:
            with open(history_path, 'r') as f:
                user_history = json.load(f)

            logger.info(f"Loaded user history for {len(user_history)} users")
            return user_history
        except Exception as e:
            logger.error(f"Error loading user history: {e}")
            return None

    def load_cold_warm_users(self) -> Dict[str, List[str]]:
        """
        Load cold and warm user lists if available.

        Returns:
            Dictionary with 'cold_users' and 'warm_users' lists
        """
        cold_warm_path = os.path.join(self.input_path, 'cold_warm_users.json')

        if not os.path.exists(cold_warm_path):
            logger.warning(f"Cold/warm users file not found: {cold_warm_path}")
            return {'cold_users': [], 'warm_users': []}

        try:
            with open(cold_warm_path, 'r') as f:
                user_sets = json.load(f)

            logger.info(
                f"Loaded {len(user_sets.get('cold_users', []))} cold users and {len(user_sets.get('warm_users', []))} warm users")
            return user_sets
        except Exception as e:
            logger.error(f"Error loading cold/warm users: {e}")
            return {'cold_users': [], 'warm_users': []}

    def split_train_test(self, df: pd.DataFrame, domain: str, use_time: bool = True) -> None:
        """
        Split data into train and test sets, optionally using time-based splitting.

        Args:
            df: DataFrame containing user-item interactions
            domain: Domain name
            use_time: Whether to use time-based splitting (if available)
        """
        logger.info(f"Creating train/test split for {domain}")

        has_time = 'unixReviewTime' in df.columns and df['unixReviewTime'].notna().all()
        time_based = use_time and has_time

        train_indices = []
        test_indices = []

        if time_based:
            logger.info(f"Using time-based splitting for {domain}")
            df = df.sort_values('unixReviewTime')

            # Group by users and take the last n% interactions as test
            for user_id, group in tqdm(df.groupby('reviewerID'), desc=f"Splitting {domain}"):
                n_test = max(1, int(len(group) * self.test_ratio))
                test_indices.extend(group.index[-n_test:])
                train_indices.extend(group.index[:-n_test])
        else:
            logger.info(f"Using random splitting for {domain}")

            # Group by users and randomly select test samples
            for user_id, group in tqdm(df.groupby('reviewerID'), desc=f"Splitting {domain}"):
                n_test = max(1, int(len(group) * self.test_ratio))

                # Use scikit-learn's train_test_split for proper randomization
                group_train, group_test = train_test_split(
                    group, test_size=n_test, random_state=self.random_seed
                )

                train_indices.extend(group_train.index)
                test_indices.extend(group_test.index)

        # Save train and test splits
        train_path = os.path.join(self.output_path, f"{domain}_train.csv")
        test_path = os.path.join(self.output_path, f"{domain}_test.csv")

        df.loc[train_indices].to_csv(train_path, index=False)
        df.loc[test_indices].to_csv(test_path, index=False)

        logger.info(f"Saved {len(train_indices)} train and {len(test_indices)} test records for {domain}")

        # Generate split statistics
        self._generate_split_stats(
            df.loc[train_indices],
            df.loc[test_indices],
            domain,
            "train_test"
        )

    def split_cold_start(self, df: pd.DataFrame, domain: str) -> None:
        """
        Create cold-start train/test splits.

        This simulates the scenario of new users with limited interaction history.
        Cold training set contains only the first few interactions for each user,
        while the test set contains the remaining interactions.

        Args:
            df: DataFrame containing user-item interactions
            domain: Domain name
        """
        logger.info(f"Creating cold-start split for {domain}")

        # Time-sort if possible
        if 'unixReviewTime' in df.columns and df['unixReviewTime'].notna().all():
            df = df.sort_values(['reviewerID', 'unixReviewTime'])

        cold_train_indices = []
        cold_test_indices = []
        skipped_users = 0

        for user_id, group in tqdm(df.groupby('reviewerID'), desc=f"Cold splitting {domain}"):
            if len(group) <= self.cold_threshold:
                # User doesn't have enough interactions for cold-start testing
                cold_train_indices.extend(group.index)
                skipped_users += 1
            else:
                # Take first cold_threshold interactions for training
                cold_train_indices.extend(group.index[:self.cold_threshold])
                # Take remaining interactions for testing
                cold_test_indices.extend(group.index[self.cold_threshold:])

        # Save cold-start splits
        cold_train_path = os.path.join(self.output_path, f"{domain}_cold_train.csv")
        cold_test_path = os.path.join(self.output_path, f"{domain}_cold_test.csv")

        df.loc[cold_train_indices].to_csv(cold_train_path, index=False)
        df.loc[cold_test_indices].to_csv(cold_test_path, index=False)

        logger.info(
            f"Saved {len(cold_train_indices)} cold-train and {len(cold_test_indices)} cold-test records for {domain}")
        logger.info(f"Skipped {skipped_users} users with ≤{self.cold_threshold} interactions")

        # Generate split statistics
        self._generate_split_stats(
            df.loc[cold_train_indices],
            df.loc[cold_test_indices],
            domain,
            "cold_start"
        )

    def split_kfold(self, df: pd.DataFrame, domain: str) -> None:
        """
        Create k-fold cross-validation splits.

        Args:
            df: DataFrame containing user-item interactions
            domain: Domain name
        """
        logger.info(f"Creating {self.n_splits}-fold cross-validation splits for {domain}")

        # Create KFold splitter
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)

        # Create user-stratified folds to ensure all users appear in train and validation
        user_groups = {}
        for user_id, group in df.groupby('reviewerID'):
            user_groups[user_id] = group

        all_users = list(user_groups.keys())

        for fold, (train_users, val_users) in enumerate(kf.split(all_users), start=1):
            train_indices = []
            val_indices = []

            # Collect indices for training and validation
            for user_idx in train_users:
                user_id = all_users[user_idx]
                train_indices.extend(user_groups[user_id].index)

            for user_idx in val_users:
                user_id = all_users[user_idx]
                val_indices.extend(user_groups[user_id].index)

            # Save fold splits
            fold_train_path = os.path.join(self.output_path, f"{domain}_fold{fold}_train.csv")
            fold_val_path = os.path.join(self.output_path, f"{domain}_fold{fold}_val.csv")

            df.loc[train_indices].to_csv(fold_train_path, index=False)
            df.loc[val_indices].to_csv(fold_val_path, index=False)

            logger.info(f"Saved fold {fold}: {len(train_indices)} train, {len(val_indices)} val records for {domain}")

    def create_cross_domain_splits(self, data: Dict[str, pd.DataFrame], user_history: Optional[Dict] = None) -> None:
        """
        Create cross-domain train/test splits for domain transfer evaluation.

        For each domain pair (source→target), create splits where:
        - Training data contains user interactions from the source domain
        - Test data contains user interactions from the target domain

        Args:
            data: Dictionary mapping domain names to DataFrames
            user_history: Optional user history dictionary
        """
        if len(data) < 2:
            logger.warning("Need at least 2 domains for cross-domain splits")
            return

        logger.info("Creating cross-domain splits")

        domains = list(data.keys())

        # Create directory for cross-domain splits
        cross_domain_dir = os.path.join(self.output_path, "cross_domain")
        os.makedirs(cross_domain_dir, exist_ok=True)

        # Process all domain pairs
        for i, source_domain in enumerate(domains):
            for j, target_domain in enumerate(domains):
                if i == j:
                    continue  # Skip same domain pairs

                logger.info(f"Creating {source_domain}→{target_domain} cross-domain split")

                # Get user overlap between domains
                source_users = set(data[source_domain]['reviewerID'].unique())
                target_users = set(data[target_domain]['reviewerID'].unique())
                common_users = source_users.intersection(target_users)

                logger.info(f"Found {len(common_users)} users with interactions in both domains")

                if len(common_users) == 0:
                    logger.warning(f"No common users between {source_domain} and {target_domain}, skipping")
                    continue

                # Create directory for this domain pair
                pair_dir = os.path.join(cross_domain_dir, f"{source_domain}_to_{target_domain}")
                os.makedirs(pair_dir, exist_ok=True)

                # Filter data to common users
                source_df = data[source_domain][data[source_domain]['reviewerID'].isin(common_users)]
                target_df = data[target_domain][data[target_domain]['reviewerID'].isin(common_users)]

                # Save source domain data as training set
                source_path = os.path.join(pair_dir, "source_train.csv")
                source_df.to_csv(source_path, index=False)

                # Split target domain data into train and test
                target_users_list = list(common_users)
                n_test_users = max(1, int(len(target_users_list) * self.test_ratio))

                np.random.shuffle(target_users_list)
                target_train_users = set(target_users_list[:-n_test_users])
                target_test_users = set(target_users_list[-n_test_users:])

                # Save target domain splits
                target_train_df = target_df[target_df['reviewerID'].isin(target_train_users)]
                target_test_df = target_df[target_df['reviewerID'].isin(target_test_users)]

                target_train_path = os.path.join(pair_dir, "target_train.csv")
                target_test_path = os.path.join(pair_dir, "target_test.csv")

                target_train_df.to_csv(target_train_path, index=False)
                target_test_df.to_csv(target_test_path, index=False)

                logger.info(f"Saved {len(source_df)} source training records")
                logger.info(
                    f"Saved {len(target_train_df)} target training and {len(target_test_df)} target test records")

                # Generate statistics for this cross-domain split
                self._generate_cross_domain_stats(
                    source_df, target_train_df, target_test_df,
                    source_domain, target_domain
                )

    def create_scenario_specific_splits(self, data: Dict[str, pd.DataFrame], user_sets: Dict[str, List[str]]) -> None:
        """
        Create splits for cold and warm user scenarios.

        Args:
            data: Dictionary mapping domain names to DataFrames
            user_sets: Dictionary with 'cold_users' and 'warm_users' lists
        """
        cold_users = set(user_sets.get('cold_users', []))
        warm_users = set(user_sets.get('warm_users', []))

        if not cold_users and not warm_users:
            logger.warning("No cold or warm users found, skipping scenario-specific splits")
            return

        logger.info(f"Creating scenario-specific splits for {len(cold_users)} cold and {len(warm_users)} warm users")

        # Create directory for scenario splits
        scenario_dir = os.path.join(self.output_path, "scenarios")
        os.makedirs(scenario_dir, exist_ok=True)

        for domain, df in data.items():
            # Create cold user split if we have cold users
            if cold_users:
                cold_df = df[df['reviewerID'].isin(cold_users)]
                if not cold_df.empty:
                    cold_path = os.path.join(scenario_dir, f"{domain}_cold_users.csv")
                    cold_df.to_csv(cold_path, index=False)
                    logger.info(f"Saved {len(cold_df)} cold user records for {domain}")

            # Create warm user split if we have warm users
            if warm_users:
                warm_df = df[df['reviewerID'].isin(warm_users)]
                if not warm_df.empty:
                    warm_path = os.path.join(scenario_dir, f"{domain}_warm_users.csv")
                    warm_df.to_csv(warm_path, index=False)
                    logger.info(f"Saved {len(warm_df)} warm user records for {domain}")

    def _generate_split_stats(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                              domain: str, split_type: str) -> None:
        """
        Generate and save statistics for a dataset split.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            domain: Domain name
            split_type: Type of split (e.g., "train_test", "cold_start")
        """
        stats = {
            "domain": domain,
            "split_type": split_type,
            "train": {
                "num_interactions": len(train_df),
                "num_users": train_df['reviewerID'].nunique(),
                "num_items": train_df['asin'].nunique(),
                "avg_rating": float(train_df['overall'].mean()),
                "rating_distribution": train_df['overall'].value_counts().to_dict(),
                "avg_interactions_per_user": len(train_df) / train_df['reviewerID'].nunique()
            },
            "test": {
                "num_interactions": len(test_df),
                "num_users": test_df['reviewerID'].nunique(),
                "num_items": test_df['asin'].nunique(),
                "avg_rating": float(test_df['overall'].mean()),
                "rating_distribution": test_df['overall'].value_counts().to_dict(),
                "avg_interactions_per_user": len(test_df) / test_df['reviewerID'].nunique() if len(test_df) > 0 else 0
            },
            "overlap": {
                "user_overlap": len(set(train_df['reviewerID'].unique()) & set(test_df['reviewerID'].unique())),
                "item_overlap": len(set(train_df['asin'].unique()) & set(test_df['asin'].unique()))
            }
        }

        # Save statistics
        stats_dir = os.path.join(self.output_path, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"{domain}_{split_type}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def _generate_cross_domain_stats(self, source_df: pd.DataFrame,
                                     target_train_df: pd.DataFrame,
                                     target_test_df: pd.DataFrame,
                                     source_domain: str,
                                     target_domain: str) -> None:
        """
        Generate and save statistics for a cross-domain split.

        Args:
            source_df: Source domain DataFrame
            target_train_df: Target domain training DataFrame
            target_test_df: Target domain test DataFrame
            source_domain: Source domain name
            target_domain: Target domain name
        """
        stats = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "source": {
                "num_interactions": len(source_df),
                "num_users": source_df['reviewerID'].nunique(),
                "num_items": source_df['asin'].nunique(),
                "avg_rating": float(source_df['overall'].mean()),
                "avg_interactions_per_user": len(source_df) / source_df['reviewerID'].nunique()
            },
            "target_train": {
                "num_interactions": len(target_train_df),
                "num_users": target_train_df['reviewerID'].nunique(),
                "num_items": target_train_df['asin'].nunique(),
                "avg_rating": float(target_train_df['overall'].mean()),
                "avg_interactions_per_user": len(target_train_df) / target_train_df['reviewerID'].nunique()
            },
            "target_test": {
                "num_interactions": len(target_test_df),
                "num_users": target_test_df['reviewerID'].nunique(),
                "num_items": target_test_df['asin'].nunique(),
                "avg_rating": float(target_test_df['overall'].mean()),
                "avg_interactions_per_user": len(target_test_df) / target_test_df['reviewerID'].nunique()
            },
            "overlap": {
                "user_overlap": len(set(source_df['reviewerID'].unique()) &
                                    set(target_train_df['reviewerID'].unique()) &
                                    set(target_test_df['reviewerID'].unique())),
                "source_target_train_user_overlap": len(set(source_df['reviewerID'].unique()) &
                                                        set(target_train_df['reviewerID'].unique())),
                "source_target_test_user_overlap": len(set(source_df['reviewerID'].unique()) &
                                                       set(target_test_df['reviewerID'].unique()))
            }
        }

        # Save statistics
        stats_dir = os.path.join(self.output_path, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"{source_domain}_to_{target_domain}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def run(self):
        """
        Run the complete data splitting pipeline.
        """
        logger.info("Starting data splitting pipeline")

        # Step 1: Load filtered data
        data = self.load_filtered_data()
        if not data:
            logger.error("No filtered data found. Aborting splitting.")
            return

        # Step 2: Load user history if available
        user_history = self.load_user_history()

        # Step 3: Load cold/warm user sets if available
        user_sets = self.load_cold_warm_users()

        # Step 4: Create various splits for each domain
        for domain, df in data.items():
            logger.info(f"Processing splits for {domain}")

            # Create standard train/test split
            self.split_train_test(df, domain)

            # Create cold-start split
            self.split_cold_start(df, domain)

            # Create k-fold cross-validation splits
            self.split_kfold(df, domain)

        # Step 5: Create cross-domain splits
        self.create_cross_domain_splits(data, user_history)

        # Step 6: Create scenario-specific splits
        self.create_scenario_specific_splits(data, user_sets)

        logger.info("Data splitting completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Splitter")
    parser.add_argument("--input_path", type=str, default="data/splits",
                        help="Path to filtered data")
    parser.add_argument("--output_path", type=str, default="data/splits",
                        help="Path to save split data")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Ratio of data to use for testing")
    parser.add_argument("--cold_threshold", type=int, default=5,
                        help="Maximum interactions for cold-start training")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    splitter = DataSplitter(
        input_path=args.input_path,
        output_path=args.output_path,
        test_ratio=args.test_ratio,
        cold_threshold=args.cold_threshold,
        n_splits=args.n_splits,
        random_seed=args.random_seed
    )

    splitter.run()