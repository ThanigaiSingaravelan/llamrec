# utils/performance_utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Set
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


class PerformanceOptimizer:
    """Performance optimization utilities for LLAMAREC."""

    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        return df

    @staticmethod
    @lru_cache(maxsize=10000)
    def cached_fuzzy_match(item1: str, item2: str, threshold: float = 0.8) -> bool:
        """Cached version of fuzzy matching for better performance."""
        words1 = frozenset(item1.lower().split())
        words2 = frozenset(item2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard_sim = intersection / union if union > 0 else 0.0
        return jaccard_sim >= threshold

    @staticmethod
    def parallel_user_processing(users: List[str],
                                 processing_func,
                                 n_workers: int = None,
                                 use_processes: bool = False) -> List:
        """Process users in parallel."""
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(users))

        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        with executor_class(max_workers=n_workers) as executor:
            results = list(executor.map(processing_func, users))

        return results

    @staticmethod
    def batch_ollama_requests(prompts: List[str],
                              ollama_caller,
                              batch_size: int = 5,
                              delay: float = 0.1) -> List:
        """Batch Ollama requests to avoid overwhelming the API."""
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []

            for prompt in batch:
                result = ollama_caller(prompt)
                batch_results.append(result)
                time.sleep(delay)  # Rate limiting

            results.extend(batch_results)

            # Progress indicator
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch)}/{len(prompts)} prompts")

        return results


# Enhanced CrossDomainPreprocessor with optimizations
class OptimizedCrossDomainPreprocessor:
    """Optimized version of CrossDomainPreprocessor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_domain_cache = {}

    def find_overlapping_users_optimized(self, data: Dict[str, pd.DataFrame]) -> Set[str]:
        """Optimized version using vectorized operations."""
        # Use pandas groupby for faster counting
        all_user_domains = []

        for domain, df in data.items():
            if df.empty:
                continue

            user_counts = df['reviewerID'].value_counts()
            active_users = user_counts[user_counts >= self.min_interactions].index

            for user_id in active_users:
                all_user_domains.append((user_id, domain))

        # Convert to DataFrame for vectorized operations
        user_domain_df = pd.DataFrame(all_user_domains, columns=['user_id', 'domain'])
        domain_counts = user_domain_df.groupby('user_id')['domain'].count()

        # Filter users with sufficient domains
        overlapping_users = set(
            domain_counts[domain_counts >= self.min_domains].index
        )

        return overlapping_users

    def parallel_user_history_generation(self, data: Dict[str, pd.DataFrame],
                                         overlapping_users: Set[str]) -> Dict:
        """Generate user history in parallel."""

        def process_user_batch(user_batch):
            batch_history = {}
            for user_id in user_batch:
                user_data = {}
                for domain, df in data.items():
                    user_df = df[df['reviewerID'] == user_id]
                    if not user_df.empty:
                        # Process user's items in this domain
                        liked = user_df[user_df['overall'] >= 4].to_dict('records')
                        disliked = user_df[user_df['overall'] < 4].to_dict('records')

                        user_data[domain] = {
                            'liked': liked,
                            'disliked': disliked,
                            'count': len(user_df)
                        }

                if user_data:
                    batch_history[user_id] = user_data

            return batch_history

        # Split users into batches for parallel processing
        user_list = list(overlapping_users)
        batch_size = max(1, len(user_list) // mp.cpu_count())
        user_batches = [user_list[i:i + batch_size]
                        for i in range(0, len(user_list), batch_size)]

        # Process batches in parallel
        with ProcessPoolExecutor() as executor:
            batch_results = list(executor.map(process_user_batch, user_batches))

        # Combine results
        user_history = {}
        for batch_result in batch_results:
            user_history.update(batch_result)

        return user_history


# Memory-efficient evaluation
class MemoryEfficientEvaluator:
    """Memory-efficient version of RecommendationEvaluator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_cache = {}

    def evaluate_recommendations_streaming(self, results_file: str,
                                           k: int = 3,
                                           chunk_size: int = 1000) -> Dict:
        """Evaluate recommendations in streaming fashion to save memory."""
        import json

        metrics_accumulator = {
            'precision_scores': [],
            'recall_scores': [],
            'ndcg_scores': [],
            'diversity_scores': [],
            'novelty_scores': []
        }

        # Process file in chunks
        with open(results_file, 'r') as f:
            data = json.load(f)
            results = data.get('results', [])

            for i in range(0, len(results), chunk_size):
                chunk = results[i:i + chunk_size]
                chunk_metrics = self._process_chunk(chunk, k)

                # Accumulate metrics
                for key, values in chunk_metrics.items():
                    metrics_accumulator[key].extend(values)

                # Clear cache periodically
                if i % (chunk_size * 10) == 0:
                    self.item_cache.clear()

        # Calculate final metrics
        return self._calculate_final_metrics(metrics_accumulator, k)

# Usage example:
# optimizer = PerformanceOptimizer()
# df_optimized = optimizer.optimize_dataframe_memory(df)
#
# # Parallel processing
# results = optimizer.parallel_user_processing(
#     users,
#     lambda user: process_single_user(user),
#     n_workers=4
# )