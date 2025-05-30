#!/usr/bin/env python3
"""
Quantitative Evaluation System for Cross-Domain Recommendations
Complete working version with proper item extraction and all features
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import re
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import warnings
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".llamarec.")))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Comprehensive evaluation system for cross-domain recommendations.
    Supports multiple metrics including ranking metrics, coverage, diversity, and novelty.
    """

    def __init__(self, user_history_path: str = "data/splits/user_history.json"):
        """
        Initialize evaluator with user history and item catalogs.

        Args:
            user_history_path: Path to user history JSON file
        """
        self.user_history = self.load_user_history(user_history_path)
        self.domain_catalogs = self.build_domain_catalogs()
        self.evaluation_results = {}

        logger.info(f"Built catalogs: {[(d, len(items)) for d, items in self.domain_catalogs.items()]}")
        logger.info(f"Initialized evaluator with {len(self.user_history)} users")

    def load_user_history(self, path: str) -> Dict:
        """Load user history from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user history: {e}")
            return {}

    def build_domain_catalogs(self) -> Dict[str, Set[str]]:
        """Build item catalogs for each domain from user history."""
        catalogs = defaultdict(set)

        for user_data in self.user_history.values():
            for domain, data in user_data.items():
                # Extract all items (liked + disliked)
                for item in data.get('liked', []):
                    item_id = item.get('asin') or item.get('title', str(item))
                    catalogs[domain].add(item_id)
                for item in data.get('disliked', []):
                    item_id = item.get('asin') or item.get('title', str(item))
                    catalogs[domain].add(item_id)

        return dict(catalogs)

    def extract_recommended_items(self, recommendation_text: str, k: int = 10) -> List[str]:
        """
        Extract item names from LLM recommendation text.
        Handles multiple formats including quoted titles with years and explanations.

        Args:
            recommendation_text: Raw text from LLM
            k: Maximum number of items to extract

        Returns:
            List of extracted item names
        """
        if not recommendation_text or not isinstance(recommendation_text, str):
            return []

        items = []

        # Debug: Log the input text for troubleshooting
        logger.debug(f"Extracting from text: {recommendation_text[:200]}...")

        # Pattern 1: "1. "ItemName" (year) - explanation" or "1. "ItemName" (year) [explanation]"
        pattern1 = r'^\d+\.\s*"([^"]+)"\s*(?:\([^)]*\))?\s*[-–—\[]'
        matches1 = re.findall(pattern1, recommendation_text, re.MULTILINE | re.IGNORECASE)
        items.extend([match.strip() for match in matches1])

        # Only try other patterns if we didn't get enough items from pattern1
        if len(items) < k:
            # Pattern 2: "1. ItemName - explanation" (without quotes)
            pattern2 = r'^\d+\.\s*([^-–—\[\n"]+?)\s*[-–—\[]'
            matches2 = re.findall(pattern2, recommendation_text, re.MULTILINE | re.IGNORECASE)
            items.extend([match.strip().strip('"\'*') for match in matches2 if
                          match.strip() not in [m.strip() for m in matches1]])

            # Pattern 3: "**ItemName** - explanation"
            pattern3 = r'\*\*([^*\n]+?)\*\*\s*[-–—\[]'
            matches3 = re.findall(pattern3, recommendation_text, re.IGNORECASE)
            items.extend([match.strip().strip('"\'') for match in matches3])

        # Debug: Log found items
        logger.debug(f"Raw extracted items: {items[:10]}")

        # Clean and deduplicate
        cleaned_items = []
        seen = set()

        for item in items[:k * 2]:  # Get more than needed, then filter
            # Clean the item name
            year_pattern = r'\s*\(\d{4}\).*$'
            item = re.sub(year_pattern, '', item)  # Remove year and everything after
            dash_pattern = r'\s*–.*$'
            item = re.sub(dash_pattern, '', item)  # Remove em-dash explanations
            hyphen_pattern = r'\s*-.*$'
            item = re.sub(hyphen_pattern, '', item)  # Remove dash explanations
            item = item.strip().strip('.,!?;:*"\'')

            # Skip unwanted items
            skip_words = ['recommendation', 'based on', 'similar to', 'matches', 'appeals',
                          'suggest', 'recommend', 'perfect', 'great', 'excellent', 'love',
                          'because', 'since', 'like', 'enjoy', 'fan', 'interest']

            if (len(item) < 2 or len(item) > 100 or
                    item.lower() in seen or
                    any(skip_word in item.lower() for skip_word in skip_words) or
                    item.lower().startswith(('the user', 'this user', 'based on', 'given'))):
                continue

            seen.add(item.lower())
            cleaned_items.append(item)

            if len(cleaned_items) >= k:
                break

        # Debug: Log final items
        logger.debug(f"Final extracted items: {cleaned_items}")

        return cleaned_items

    def calculate_precision_at_k(self, recommendations: List[str],
                                 relevant_items: List[str], k: int = 3) -> float:
        """
        Calculate Precision@K metric.

        Args:
            recommendations: List of recommended items
            relevant_items: List of items the user actually liked
            k: Number of top recommendations to consider

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        hits = 0
        for rec in top_k_recs:
            rec_clean = rec.lower().strip()
            # Exact match or fuzzy match for titles
            if (rec_clean in relevant_set or
                    any(self.fuzzy_match(rec_clean, rel_item) for rel_item in relevant_set)):
                hits += 1

        return hits / min(len(top_k_recs), k) if top_k_recs else 0.0

    def calculate_recall_at_k(self, recommendations: List[str],
                              relevant_items: List[str], k: int = 3) -> float:
        """
        Calculate Recall@K metric.

        Args:
            recommendations: List of recommended items
            relevant_items: List of items the user actually liked
            k: Number of top recommendations to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        hits = 0
        for rec in top_k_recs:
            rec_clean = rec.lower().strip()
            if (rec_clean in relevant_set or
                    any(self.fuzzy_match(rec_clean, rel_item) for rel_item in relevant_set)):
                hits += 1

        return hits / len(relevant_set) if relevant_set else 0.0

    def calculate_ndcg_at_k(self, recommendations: List[str],
                            relevant_items: List[str], k: int = 3) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).

        Args:
            recommendations: List of recommended items
            relevant_items: List of items the user actually liked
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        # Create relevance scores (1 for relevant, 0 for irrelevant)
        relevance_scores = []
        for rec in top_k_recs:
            rec_clean = rec.lower().strip()
            if (rec_clean in relevant_set or
                    any(self.fuzzy_match(rec_clean, rel_item) for rel_item in relevant_set)):
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.0)

        # Pad with zeros if needed
        while len(relevance_scores) < k:
            relevance_scores.append(0.0)

        # Calculate ideal relevance scores (all 1s up to min(k, len(relevant_items)))
        ideal_scores = [1.0] * min(k, len(relevant_set)) + [0.0] * max(0, k - len(relevant_set))

        return self.manual_ndcg(relevance_scores, ideal_scores)

    def manual_ndcg(self, relevance_scores: List[float], ideal_scores: List[float]) -> float:
        """Manual NDCG calculation as fallback."""

        def dcg(scores):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))

        dcg_score = dcg(relevance_scores)
        idcg_score = dcg(sorted(ideal_scores, reverse=True))

        return dcg_score / idcg_score if idcg_score > 0 else 0.0

    def fuzzy_match(self, item1: str, item2: str, threshold: float = 0.8) -> bool:
        """
        Check if two item names are similar enough to be considered a match.
        Uses Jaccard similarity on word sets.
        """
        words1 = set(item1.lower().split())
        words2 = set(item2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard_sim = intersection / union if union > 0 else 0.0
        return jaccard_sim >= threshold

    def calculate_coverage(self, all_recommendations: List[List[str]],
                           domain: str) -> float:
        """
        Calculate catalog coverage - fraction of catalog items recommended.

        Args:
            all_recommendations: List of recommendation lists from all tests
            domain: Target domain

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if domain not in self.domain_catalogs:
            return 0.0

        catalog = self.domain_catalogs[domain]
        recommended_items = set()

        for rec_list in all_recommendations:
            for item in rec_list:
                recommended_items.add(item.lower().strip())

        # Check how many catalog items were recommended
        covered_items = 0
        for catalog_item in catalog:
            catalog_clean = str(catalog_item).lower().strip()
            if (catalog_clean in recommended_items or
                    any(self.fuzzy_match(catalog_clean, rec_item)
                        for rec_item in recommended_items)):
                covered_items += 1

        return covered_items / len(catalog) if catalog else 0.0

    def calculate_diversity(self, recommendations: List[str]) -> float:
        """
        Calculate intra-list diversity using Jaccard distance.

        Args:
            recommendations: List of recommended items

        Returns:
            Average pairwise diversity (0.0 to 1.0)
        """
        if len(recommendations) < 2:
            return 0.0

        diversities = []

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                words1 = set(recommendations[i].lower().split())
                words2 = set(recommendations[j].lower().split())

                if not words1 or not words2:
                    continue

                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))

                # Jaccard distance = 1 - Jaccard similarity
                diversity = 1 - (intersection / union if union > 0 else 0)
                diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

    def calculate_novelty(self, recommendations: List[str],
                          user_history: Dict, domain: str) -> float:
        """
        Calculate novelty - how different recommendations are from user's history.

        Args:
            recommendations: List of recommended items
            user_history: User's interaction history
            domain: Target domain

        Returns:
            Average novelty score (0.0 to 1.0)
        """
        if not recommendations or domain not in user_history:
            return 0.0

        # Get user's historical items in target domain
        historical_items = []
        domain_data = user_history[domain]

        for item in domain_data.get('liked', []) + domain_data.get('disliked', []):
            item_name = item.get('title') or str(item)
            historical_items.append(item_name.lower().strip())

        if not historical_items:
            return 1.0  # All recommendations are novel if no history

        novelty_scores = []

        for rec in recommendations:
            rec_clean = rec.lower().strip()
            # Calculate minimum similarity to any historical item
            similarities = []

            for hist_item in historical_items:
                words_rec = set(rec_clean.split())
                words_hist = set(hist_item.split())

                if words_rec and words_hist:
                    intersection = len(words_rec.intersection(words_hist))
                    union = len(words_rec.union(words_hist))
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)

            # Novelty = 1 - max_similarity
            max_similarity = max(similarities) if similarities else 0
            novelty_scores.append(1 - max_similarity)

        return np.mean(novelty_scores) if novelty_scores else 1.0

    def evaluate_recommendations(self, results_file: str, k: int = 3) -> Dict:
        """
        Comprehensive evaluation of recommendation results.

        Args:
            results_file: Path to JSON file with recommendation results
            k: Number of top recommendations to consider

        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating recommendations from {results_file}")

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            return {}

        results = data.get('results', [])
        if not results:
            logger.warning("No results found in file")
            return {}

        # Initialize metrics collections
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        coverage_by_domain = defaultdict(list)
        diversity_scores = []
        novelty_scores = []

        # Process each result
        valid_results = 0

        for result in results:
            try:
                user_id = result.get('user_id')
                source_domain = result.get('source_domain')
                target_domain = result.get('target_domain')
                rec_text = result.get('recommendations', '')

                if not all([user_id, source_domain, target_domain, rec_text]):
                    continue

                # Extract recommended items
                recommended_items = self.extract_recommended_items(rec_text, k)
                if not recommended_items:
                    continue

                # Get user's actual preferences in target domain
                if user_id not in self.user_history:
                    continue

                user_data = self.user_history[user_id]
                if target_domain not in user_data:
                    continue

                target_liked = user_data[target_domain].get('liked', [])
                if not target_liked:
                    continue

                # Extract item names from user's liked items
                actual_items = []
                for item in target_liked:
                    item_name = item.get('title') or str(item)
                    actual_items.append(item_name)

                # Calculate metrics
                precision = self.calculate_precision_at_k(recommended_items, actual_items, k)
                recall = self.calculate_recall_at_k(recommended_items, actual_items, k)
                ndcg = self.calculate_ndcg_at_k(recommended_items, actual_items, k)
                diversity = self.calculate_diversity(recommended_items)
                novelty = self.calculate_novelty(recommended_items, user_data, target_domain)

                # Collect scores
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                diversity_scores.append(diversity)
                novelty_scores.append(novelty)
                coverage_by_domain[target_domain].append(recommended_items)

                valid_results += 1

            except Exception as e:
                logger.warning(f"Error processing result: {e}")
                continue

        if valid_results == 0:
            logger.warning("No valid results to evaluate")
            return {}

        # Calculate coverage for each domain
        coverage_scores = {}
        for domain, rec_lists in coverage_by_domain.items():
            coverage_scores[domain] = self.calculate_coverage(rec_lists, domain)

        # Compile final metrics
        evaluation_metrics = {
            'overall_metrics': {
                'precision_at_k': {
                    'mean': float(np.mean(precision_scores)),
                    'std': float(np.std(precision_scores)),
                    'median': float(np.median(precision_scores)),
                    'scores': [float(x) for x in precision_scores]
                },
                'recall_at_k': {
                    'mean': float(np.mean(recall_scores)),
                    'std': float(np.std(recall_scores)),
                    'median': float(np.median(recall_scores)),
                    'scores': [float(x) for x in recall_scores]
                },
                'ndcg_at_k': {
                    'mean': float(np.mean(ndcg_scores)),
                    'std': float(np.std(ndcg_scores)),
                    'median': float(np.median(ndcg_scores)),
                    'scores': [float(x) for x in ndcg_scores]
                },
                'diversity': {
                    'mean': float(np.mean(diversity_scores)),
                    'std': float(np.std(diversity_scores)),
                    'median': float(np.median(diversity_scores)),
                    'scores': [float(x) for x in diversity_scores]
                },
                'novelty': {
                    'mean': float(np.mean(novelty_scores)),
                    'std': float(np.std(novelty_scores)),
                    'median': float(np.median(novelty_scores)),
                    'scores': [float(x) for x in novelty_scores]
                }
            },
            'coverage_by_domain': {k: float(v) for k, v in coverage_scores.items()},
            'evaluation_info': {
                'k': int(k),
                'total_results': int(len(results)),
                'valid_results': int(valid_results),
                'evaluation_timestamp': datetime.now().isoformat(),
                'results_file': results_file
            }
        }

        logger.info(f"Evaluation complete: {valid_results} valid results processed")
        logger.info(f"Avg Precision@{k}: {np.mean(precision_scores):.3f}")
        logger.info(f"Avg Recall@{k}: {np.mean(recall_scores):.3f}")
        logger.info(f"Avg NDCG@{k}: {np.mean(ndcg_scores):.3f}")

        return evaluation_metrics

    def compare_systems(self, results_files: List[str], system_names: List[str],
                        k: int = 3) -> Dict:
        """
        Compare multiple recommendation systems statistically.

        Args:
            results_files: List of result file paths
            system_names: Names for each system
            k: Number of top recommendations to consider

        Returns:
            Comparison results with statistical tests
        """
        logger.info(f"Comparing {len(results_files)} systems")

        system_metrics = {}

        # Evaluate each system
        for file_path, system_name in zip(results_files, system_names):
            metrics = self.evaluate_recommendations(file_path, k)
            if metrics:
                system_metrics[system_name] = metrics

        if len(system_metrics) < 2:
            logger.warning("Need at least 2 systems for comparison")
            return {}

        # Statistical comparisons
        comparisons = {}
        metric_names = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity', 'novelty']

        systems = list(system_metrics.keys())

        for i, system1 in enumerate(systems):
            for system2 in systems[i + 1:]:
                comparison_key = f"{system1}_vs_{system2}"
                comparisons[comparison_key] = {}

                for metric in metric_names:
                    scores1 = system_metrics[system1]['overall_metrics'][metric]['scores']
                    scores2 = system_metrics[system2]['overall_metrics'][metric]['scores']

                    # T-test for statistical significance
                    try:
                        t_stat, p_value = ttest_ind(scores1, scores2)

                        comparisons[comparison_key][metric] = {
                            'system1_mean': float(np.mean(scores1)),
                            'system2_mean': float(np.mean(scores2)),
                            'difference': float(np.mean(scores1) - np.mean(scores2)),
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05)
                        }
                    except Exception as e:
                        logger.warning(f"Error in statistical test for {metric}: {e}")

        return {
            'system_metrics': system_metrics,
            'statistical_comparisons': comparisons,
            'comparison_info': {
                'systems_compared': system_names,
                'k': k,
                'comparison_timestamp': datetime.now().isoformat()
            }
        }

    def generate_evaluation_report(self, evaluation_results: Dict,
                                   output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            evaluation_results: Results from evaluate_recommendations()
            output_file: Optional file to save the report

        Returns:
            Report as string
        """
        if not evaluation_results:
            return "No evaluation results to report."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CROSS-DOMAIN RECOMMENDATION EVALUATION REPORT")
        report_lines.append("=" * 80)

        # Basic info
        info = evaluation_results.get('evaluation_info', {})
        report_lines.append(f"Evaluation Date: {info.get('evaluation_timestamp', 'Unknown')}")
        report_lines.append(f"Results File: {info.get('results_file', 'Unknown')}")
        report_lines.append(f"Total Results: {info.get('total_results', 0)}")
        report_lines.append(f"Valid Results: {info.get('valid_results', 0)}")
        report_lines.append(f"K (Top-K): {info.get('k', 3)}")
        report_lines.append("")

        # Overall metrics
        report_lines.append("OVERALL METRICS")
        report_lines.append("-" * 40)

        metrics = evaluation_results.get('overall_metrics', {})
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mean_val = metric_data['mean']
                std_val = metric_data['std']
                median_val = metric_data['median']

                report_lines.append(f"{metric_name.upper().replace('_', ' ')}:")
                report_lines.append(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
                report_lines.append(f"  Median: {median_val:.4f}")
                report_lines.append("")

        # Coverage by domain
        coverage = evaluation_results.get('coverage_by_domain', {})
        if coverage:
            report_lines.append("COVERAGE BY DOMAIN")
            report_lines.append("-" * 40)
            for domain, cov_score in coverage.items():
                report_lines.append(f"{domain}: {cov_score:.4f}")
            report_lines.append("")

        # Performance interpretation
        report_lines.append("PERFORMANCE INTERPRETATION")
        report_lines.append("-" * 40)

        precision_mean = metrics.get('precision_at_k', {}).get('mean', 0)
        recall_mean = metrics.get('recall_at_k', {}).get('mean', 0)
        ndcg_mean = metrics.get('ndcg_at_k', {}).get('mean', 0)

        if precision_mean > 0.1:
            report_lines.append(f"✓ Good precision: {precision_mean:.3f} (>10% relevant items in top-K)")
        else:
            report_lines.append(f"⚠ Low precision: {precision_mean:.3f} (<10% relevant items in top-K)")

        if recall_mean > 0.05:
            report_lines.append(f"✓ Reasonable recall: {recall_mean:.3f} (>5% of liked items found)")
        else:
            report_lines.append(f"⚠ Low recall: {recall_mean:.3f} (<5% of liked items found)")

        if ndcg_mean > 0.1:
            report_lines.append(f"✓ Good ranking quality: {ndcg_mean:.3f} (NDCG > 0.1)")
        else:
            report_lines.append(f"⚠ Poor ranking quality: {ndcg_mean:.3f} (NDCG < 0.1)")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")

        return report_text

    def create_visualization(self, evaluation_results: Dict,
                             output_dir: str = "evaluation_plots") -> None:
        """
        Create visualization plots for evaluation results.

        Args:
            evaluation_results: Results from evaluate_recommendations()
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualization.")
            return

        os.makedirs(output_dir, exist_ok=True)

        if not evaluation_results or 'overall_metrics' not in evaluation_results:
            logger.warning("No metrics to visualize")
            return

        metrics = evaluation_results['overall_metrics']

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Metrics distribution plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Cross-Domain Recommendation Metrics Distribution', fontsize=16)

        metric_names = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity', 'novelty']

        for i, metric_name in enumerate(metric_names):
            if metric_name in metrics and 'scores' in metrics[metric_name]:
                row, col = i // 3, i % 3
                scores = metrics[metric_name]['scores']

                axes[row, col].hist(scores, bins=20, alpha=0.7, edgecolor='black')
                axes[row, col].axvline(np.mean(scores), color='red', linestyle='--',
                                       label=f'Mean: {np.mean(scores):.3f}')
                axes[row, col].set_title(metric_name.replace('_', ' ').title())
                axes[row, col].set_xlabel('Score')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)

        # Remove empty subplot
        if len(metric_names) < 6:
            fig.delaxes(axes[1, 2])

        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Metrics summary bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_means = []
        metric_stds = []
        metric_labels = []

        for metric_name in metric_names:
            if metric_name in metrics and 'mean' in metrics[metric_name]:
                metric_means.append(metrics[metric_name]['mean'])
                metric_stds.append(metrics[metric_name]['std'])
                metric_labels.append(metric_name.replace('_', ' ').title())

        bars = ax.bar(metric_labels, metric_means, yerr=metric_stds,
                      capsize=5, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, mean_val in zip(bars, metric_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{mean_val:.3f}', ha='center', va='bottom')

        ax.set_title('Cross-Domain Recommendation Performance Summary')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Coverage by domain plot
        coverage = evaluation_results.get('coverage_by_domain', {})
        if coverage:
            fig, ax = plt.subplots(figsize=(10, 6))

            domains = list(coverage.keys())
            coverage_scores = list(coverage.values())

            bars = ax.bar(domains, coverage_scores, alpha=0.8, edgecolor='black')

            # Add value labels on bars
            for bar, score in zip(bars, coverage_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{score:.3f}', ha='center', va='bottom')

            ax.set_title('Catalog Coverage by Domain')
            ax.set_ylabel('Coverage Score')
            ax.set_xlabel('Domain')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/coverage_by_domain.png", dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Visualizations saved to {output_dir}/")


class BaselineComparator:
    """
    Baseline recommendation systems for comparison.
    """

    def __init__(self, user_history: Dict, domain_catalogs: Dict):
        self.user_history = user_history
        self.domain_catalogs = domain_catalogs

    def random_baseline(self, user_id: str, target_domain: str, k: int = 3) -> List[str]:
        """
        Random baseline: recommend random items from target domain.

        Args:
            user_id: User identifier
            target_domain: Target domain for recommendations
            k: Number of recommendations

        Returns:
            List of random recommendations
        """
        if target_domain not in self.domain_catalogs:
            return []

        catalog = list(self.domain_catalogs[target_domain])
        return np.random.choice(catalog, size=min(k, len(catalog)), replace=False).tolist()

    def popularity_baseline(self, user_id: str, target_domain: str, k: int = 3) -> List[str]:
        """
        Popularity baseline: recommend most popular items in target domain.

        Args:
            user_id: User identifier
            target_domain: Target domain for recommendations
            k: Number of recommendations

        Returns:
            List of popular recommendations
        """
        if target_domain not in self.domain_catalogs:
            return []

        # Count item frequencies across all users
        item_counts = defaultdict(int)

        for user_data in self.user_history.values():
            if target_domain in user_data:
                for item in user_data[target_domain].get('liked', []):
                    item_id = item.get('asin') or item.get('title', str(item))
                    item_counts[item_id] += 1

        # Sort by popularity and return top-k
        popular_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [item for item, count in popular_items[:k]]


def run_comprehensive_evaluation(results_files: List[str],
                                 system_names: List[str] = None,
                                 user_history_path: str = "data/splits/user_history.json",
                                 k: int = 3,
                                 output_dir: str = "evaluation_results") -> None:
    """
    Run comprehensive evaluation on multiple result files.

    Args:
        results_files: List of paths to result JSON files
        system_names: Names for each system (optional)
        user_history_path: Path to user history JSON
        k: Number of top recommendations to consider
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = RecommendationEvaluator(user_history_path)

    if system_names is None:
        system_names = [f"System_{i + 1}" for i in range(len(results_files))]

    logger.info(f"Starting comprehensive evaluation of {len(results_files)} systems")

    # Single system evaluations
    individual_results = {}

    for results_file, system_name in zip(results_files, system_names):
        logger.info(f"Evaluating {system_name}...")

        evaluation_results = evaluator.evaluate_recommendations(results_file, k)

        if evaluation_results:
            individual_results[system_name] = evaluation_results

            # Generate individual report
            report = evaluator.generate_evaluation_report(
                evaluation_results,
                f"{output_dir}/{system_name}_evaluation_report.txt"
            )

            # Create visualizations
            viz_dir = f"{output_dir}/{system_name}_plots"
            evaluator.create_visualization(evaluation_results, viz_dir)

            logger.info(f"Results for {system_name} saved to {output_dir}/")

    # Comparative evaluation
    if len(individual_results) > 1:
        logger.info("Running comparative evaluation...")

        comparison_results = evaluator.compare_systems(
            results_files, system_names, k
        )

        if comparison_results:
            # Save comparison results
            comparison_file = f"{output_dir}/system_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)

            # Generate comparison report
            generate_comparison_report(comparison_results, f"{output_dir}/comparison_report.txt")

            logger.info(f"Comparison results saved to {comparison_file}")

    # Generate baseline comparisons
    logger.info("Generating baseline comparisons...")
    generate_baseline_comparison(evaluator, individual_results, output_dir, k)

    logger.info(f"Comprehensive evaluation complete! Results saved to {output_dir}/")


def generate_comparison_report(comparison_results: Dict, output_file: str) -> None:
    """Generate a detailed comparison report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SYSTEM COMPARISON REPORT")
    report_lines.append("=" * 80)

    info = comparison_results.get('comparison_info', {})
    report_lines.append(f"Comparison Date: {info.get('comparison_timestamp', 'Unknown')}")
    report_lines.append(f"Systems: {', '.join(info.get('systems_compared', []))}")
    report_lines.append(f"K (Top-K): {info.get('k', 3)}")
    report_lines.append("")

    # System performance summary
    system_metrics = comparison_results.get('system_metrics', {})
    if system_metrics:
        report_lines.append("SYSTEM PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)

        for system_name, metrics in system_metrics.items():
            report_lines.append(f"\n{system_name.upper()}:")
            overall = metrics.get('overall_metrics', {})

            for metric_name, metric_data in overall.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    mean_val = metric_data['mean']
                    std_val = metric_data['std']
                    report_lines.append(f"  {metric_name.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        report_lines.append("")

    # Statistical comparisons
    comparisons = comparison_results.get('statistical_comparisons', {})
    if comparisons:
        report_lines.append("STATISTICAL COMPARISONS")
        report_lines.append("-" * 40)

        for comparison_key, comp_data in comparisons.items():
            report_lines.append(f"\n{comparison_key.replace('_vs_', ' vs. ').upper()}:")

            for metric, stats in comp_data.items():
                significance = "✓ SIGNIFICANT" if stats.get('significant', False) else "✗ Not significant"
                diff = stats.get('difference', 0)
                p_val = stats.get('p_value', 1.0)

                report_lines.append(f"  {metric.replace('_', ' ').title()}:")
                report_lines.append(f"    Difference: {diff:+.4f}")
                report_lines.append(f"    P-value: {p_val:.4f}")
                report_lines.append(f"    {significance}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Comparison report saved to {output_file}")


def generate_baseline_comparison(evaluator: RecommendationEvaluator,
                                 system_results: Dict,
                                 output_dir: str, k: int = 3) -> None:
    """Generate baseline comparison results."""

    baseline_comparator = BaselineComparator(
        evaluator.user_history,
        evaluator.domain_catalogs
    )

    # Generate baseline recommendations for sample users
    sample_users = list(evaluator.user_history.keys())[:20]  # Sample 20 users
    baseline_results = {
        'random': [],
        'popularity': []
    }

    for user_id in sample_users:
        user_data = evaluator.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]

        if len(domains) >= 2:
            source_domain = domains[0]
            target_domain = domains[1]

            # Generate baseline recommendations
            random_recs = baseline_comparator.random_baseline(user_id, target_domain, k)
            popular_recs = baseline_comparator.popularity_baseline(user_id, target_domain, k)

            # Get actual liked items for evaluation
            actual_items = []
            for item in user_data[target_domain].get('liked', []):
                actual_items.append(item.get('title', str(item)))

            if actual_items and random_recs:
                # Calculate metrics for each baseline
                baseline_results['random'].append({
                    'precision': evaluator.calculate_precision_at_k(random_recs, actual_items, k),
                    'recall': evaluator.calculate_recall_at_k(random_recs, actual_items, k),
                    'ndcg': evaluator.calculate_ndcg_at_k(random_recs, actual_items, k)
                })

            if actual_items and popular_recs:
                baseline_results['popularity'].append({
                    'precision': evaluator.calculate_precision_at_k(popular_recs, actual_items, k),
                    'recall': evaluator.calculate_recall_at_k(popular_recs, actual_items, k),
                    'ndcg': evaluator.calculate_ndcg_at_k(popular_recs, actual_items, k)
                })

    # Calculate baseline averages
    baseline_summary = {}
    for baseline_name, results in baseline_results.items():
        if results:
            baseline_summary[baseline_name] = {
                'precision_mean': np.mean([r['precision'] for r in results]),
                'recall_mean': np.mean([r['recall'] for r in results]),
                'ndcg_mean': np.mean([r['ndcg'] for r in results]),
                'num_evaluations': len(results)
            }

    # Generate baseline comparison report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BASELINE COMPARISON REPORT")
    report_lines.append("=" * 80)

    report_lines.append(f"Baseline Evaluation Date: {datetime.now().isoformat()}")
    report_lines.append(f"Sample Size: {len(sample_users)} users")
    report_lines.append(f"K (Top-K): {k}")
    report_lines.append("")

    # Baseline performance
    report_lines.append("BASELINE PERFORMANCE")
    report_lines.append("-" * 40)

    for baseline_name, summary in baseline_summary.items():
        report_lines.append(f"\n{baseline_name.upper().replace('_', ' ')}:")
        report_lines.append(f"  Precision@{k}: {summary['precision_mean']:.4f}")
        report_lines.append(f"  Recall@{k}: {summary['recall_mean']:.4f}")
        report_lines.append(f"  NDCG@{k}: {summary['ndcg_mean']:.4f}")
        report_lines.append(f"  Evaluations: {summary['num_evaluations']}")

    # Compare systems to baselines
    if system_results:
        report_lines.append(f"\nSYSTEM vs BASELINE COMPARISON")
        report_lines.append("-" * 40)

        for system_name, system_data in system_results.items():
            system_metrics = system_data.get('overall_metrics', {})

            system_precision = system_metrics.get('precision_at_k', {}).get('mean', 0)
            system_recall = system_metrics.get('recall_at_k', {}).get('mean', 0)
            system_ndcg = system_metrics.get('ndcg_at_k', {}).get('mean', 0)

            report_lines.append(f"\n{system_name.upper()}:")
            report_lines.append(f"  Precision@{k}: {system_precision:.4f}")
            report_lines.append(f"  Recall@{k}: {system_recall:.4f}")
            report_lines.append(f"  NDCG@{k}: {system_ndcg:.4f}")

            # Compare to best baseline
            best_baseline_precision = max([s['precision_mean'] for s in baseline_summary.values()] + [0])
            best_baseline_recall = max([s['recall_mean'] for s in baseline_summary.values()] + [0])
            best_baseline_ndcg = max([s['ndcg_mean'] for s in baseline_summary.values()] + [0])

            precision_improvement = ((system_precision - best_baseline_precision) /
                                     best_baseline_precision * 100 if best_baseline_precision > 0 else 0)
            recall_improvement = ((system_recall - best_baseline_recall) /
                                  best_baseline_recall * 100 if best_baseline_recall > 0 else 0)
            ndcg_improvement = ((system_ndcg - best_baseline_ndcg) /
                                best_baseline_ndcg * 100 if best_baseline_ndcg > 0 else 0)

            report_lines.append(f"  Improvement over best baseline:")
            report_lines.append(f"    Precision: {precision_improvement:+.1f}%")
            report_lines.append(f"    Recall: {recall_improvement:+.1f}%")
            report_lines.append(f"    NDCG: {ndcg_improvement:+.1f}%")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save baseline comparison
    baseline_file = f"{output_dir}/baseline_comparison.txt"
    with open(baseline_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    logger.info(f"Baseline comparison saved to {baseline_file}")


if __name__ == "__main__":
    # Example usage

    # Single system evaluation
    evaluator = RecommendationEvaluator("data/splits/user_history.json")

    results_file = "results/batch_test_results_20250525_060951.json"  # Your results file
    evaluation_results = evaluator.evaluate_recommendations(results_file, k=3)

    if evaluation_results:
        # Generate report
        report = evaluator.generate_evaluation_report(
            evaluation_results,
            "evaluation_report.txt"
        )
        print(report)

        # Create visualizations
        evaluator.create_visualization(evaluation_results, "evaluation_plots")

        # Save detailed results
        with open("detailed_evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)

    # Multi-system comparison example
    # results_files = [
    #     "results/system1_results.json",
    #     "results/system2_results.json"
    # ]
    # system_names = ["LLAMAREC", "Baseline_System"]
    #
    # run_comprehensive_evaluation(
    #     results_files,
    #     system_names,
    #     k=3,
    #     output_dir="comprehensive_evaluation"
    # )