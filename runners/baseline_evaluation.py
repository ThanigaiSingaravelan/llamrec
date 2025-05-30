#!/usr/bin/env python3
"""
Comprehensive Evaluation for Baseline Methods
Evaluates Content-Based, Popularity, and Random recommendation approaches
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class BaselineEvaluator:
    """Comprehensive evaluator for baseline recommendation methods"""

    def __init__(self, user_history_path: str = "data/splits/user_history.json"):
        """Initialize evaluator with user history"""
        self.user_history_path = user_history_path
        self.user_history = self.load_user_history()
        self.evaluation_results = {}

        print("🔬 Baseline Methods Evaluator Initialized")
        print(f"   User History: {len(self.user_history)} users loaded")

    def load_user_history(self) -> Dict:
        """Load user history for ground truth"""
        try:
            with open(self.user_history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading user history: {e}")
            return {}

    def load_results(self, results_files: List[str]) -> Dict:
        """Load results from JSON files"""
        all_results = {}

        for file_path in results_files:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract method name from file or data
                if 'model_info' in data and 'type' in data['model_info']:
                    method_name = data['model_info']['type']
                else:
                    method_name = os.path.basename(file_path).replace('.json', '').replace('_results', '')

                all_results[method_name] = data
                print(f"✅ Loaded {method_name}: {len(data.get('results', []))} recommendations")

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

        return all_results

    def extract_recommended_items(self, recommendation_text: str, k: int = 3) -> List[str]:
        """Extract item names from recommendation text"""
        if not recommendation_text:
            return []

        import re

        # Pattern to match numbered recommendations
        items = []
        lines = recommendation_text.split('\n')

        for line in lines:
            # Look for numbered lines like "1. ItemName - explanation"
            match = re.match(r'^\d+\.\s*(.+?)\s*[-–]', line.strip())
            if match:
                item_name = match.group(1).strip().strip('"\'')
                items.append(item_name)

        return items[:k]

    def calculate_precision_at_k(self, recommendations: List[str],
                                 relevant_items: List[str], k: int = 3) -> float:
        """Calculate Precision@K"""
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        hits = 0
        for rec in top_k_recs:
            rec_clean = rec.lower().strip()
            if rec_clean in relevant_set or any(self.fuzzy_match(rec_clean, rel) for rel in relevant_set):
                hits += 1

        return hits / min(len(top_k_recs), k) if top_k_recs else 0.0

    def calculate_recall_at_k(self, recommendations: List[str],
                              relevant_items: List[str], k: int = 3) -> float:
        """Calculate Recall@K"""
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        hits = 0
        for rec in top_k_recs:
            rec_clean = rec.lower().strip()
            if rec_clean in relevant_set or any(self.fuzzy_match(rec_clean, rel) for rel in relevant_set):
                hits += 1

        return hits / len(relevant_set) if relevant_set else 0.0

    def calculate_ndcg_at_k(self, recommendations: List[str],
                            relevant_items: List[str], k: int = 3) -> float:
        """Calculate NDCG@K"""
        if not recommendations or not relevant_items:
            return 0.0

        top_k_recs = recommendations[:k]
        relevant_set = set(item.lower().strip() for item in relevant_items)

        # Calculate DCG
        dcg = 0.0
        for i, rec in enumerate(top_k_recs):
            rec_clean = rec.lower().strip()
            if rec_clean in relevant_set or any(self.fuzzy_match(rec_clean, rel) for rel in relevant_set):
                dcg += 1 / np.log2(i + 2)

        # Calculate IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_diversity(self, recommendations: List[str]) -> float:
        """Calculate recommendation diversity using Jaccard distance"""
        if len(recommendations) < 2:
            return 0.0

        diversities = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                words1 = set(recommendations[i].lower().split())
                words2 = set(recommendations[j].lower().split())

                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    diversity = 1 - (intersection / union if union > 0 else 0)
                    diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

    def calculate_coverage(self, all_recommendations: List[List[str]],
                           domain_items: List[str]) -> float:
        """Calculate catalog coverage"""
        recommended_items = set()
        for rec_list in all_recommendations:
            for item in rec_list:
                recommended_items.add(item.lower().strip())

        domain_items_set = set(item.lower().strip() for item in domain_items)

        if not domain_items_set:
            return 0.0

        covered_items = 0
        for domain_item in domain_items_set:
            if domain_item in recommended_items:
                covered_items += 1

        return covered_items / len(domain_items_set)

    def fuzzy_match(self, item1: str, item2: str, threshold: float = 0.7) -> bool:
        """Simple fuzzy matching using Jaccard similarity"""
        words1 = set(item1.lower().split())
        words2 = set(item2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return (intersection / union) >= threshold if union > 0 else False

    def evaluate_method(self, method_name: str, method_data: Dict, k: int = 3) -> Dict:
        """Evaluate a single recommendation method"""
        print(f"\n🔍 Evaluating {method_name}...")

        results = method_data.get('results', [])
        if not results:
            print(f"   ⚠️ No results found for {method_name}")
            return {}

        # Metrics storage
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        diversity_scores = []
        coverage_recommendations = []

        valid_evaluations = 0

        for result in results:
            try:
                user_id = result.get('user_id')
                target_domain = result.get('target_domain', result.get('source_domain'))
                rec_text = result.get('recommendations', '')

                if not all([user_id, target_domain, rec_text]):
                    continue

                # Extract recommended items
                recommended_items = self.extract_recommended_items(rec_text, k)
                if not recommended_items:
                    continue

                # Get ground truth
                if user_id not in self.user_history:
                    continue

                user_data = self.user_history[user_id]
                if target_domain not in user_data:
                    continue

                liked_items = user_data[target_domain].get('liked', [])
                if not liked_items:
                    continue

                # Extract item names from liked items
                actual_items = []
                for item in liked_items:
                    if isinstance(item, dict):
                        item_name = item.get('title', str(item))
                    else:
                        item_name = str(item)
                    actual_items.append(item_name)

                # Calculate metrics
                precision = self.calculate_precision_at_k(recommended_items, actual_items, k)
                recall = self.calculate_recall_at_k(recommended_items, actual_items, k)
                ndcg = self.calculate_ndcg_at_k(recommended_items, actual_items, k)
                diversity = self.calculate_diversity(recommended_items)

                # Store scores
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                diversity_scores.append(diversity)
                coverage_recommendations.append(recommended_items)

                valid_evaluations += 1

            except Exception as e:
                continue

        if valid_evaluations == 0:
            print(f"   ❌ No valid evaluations for {method_name}")
            return {}

        # Calculate overall metrics
        evaluation_metrics = {
            'method_name': method_name,
            'valid_evaluations': valid_evaluations,
            'total_results': len(results),
            'success_rate': valid_evaluations / len(results),
            'precision_at_k': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores),
                'median': np.median(precision_scores),
                'scores': precision_scores
            },
            'recall_at_k': {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores),
                'median': np.median(recall_scores),
                'scores': recall_scores
            },
            'ndcg_at_k': {
                'mean': np.mean(ndcg_scores),
                'std': np.std(ndcg_scores),
                'median': np.median(ndcg_scores),
                'scores': ndcg_scores
            },
            'diversity': {
                'mean': np.mean(diversity_scores),
                'std': np.std(diversity_scores),
                'median': np.median(diversity_scores),
                'scores': diversity_scores
            }
        }

        print(f"   ✅ {method_name} Evaluation Complete:")
        print(f"      Valid Evaluations: {valid_evaluations}/{len(results)}")
        print(f"      Precision@{k}: {evaluation_metrics['precision_at_k']['mean']:.4f}")
        print(f"      Recall@{k}: {evaluation_metrics['recall_at_k']['mean']:.4f}")
        print(f"      NDCG@{k}: {evaluation_metrics['ndcg_at_k']['mean']:.4f}")
        print(f"      Diversity: {evaluation_metrics['diversity']['mean']:.4f}")

        return evaluation_metrics

    def compare_methods(self, all_results: Dict, k: int = 3) -> Dict:
        """Compare all baseline methods"""
        print(f"\n🏆 COMPREHENSIVE BASELINE COMPARISON")
        print("=" * 60)

        comparison_results = {}

        # Evaluate each method
        for method_name, method_data in all_results.items():
            evaluation = self.evaluate_method(method_name, method_data, k)
            if evaluation:
                comparison_results[method_name] = evaluation

        if len(comparison_results) < 2:
            print("⚠️ Need at least 2 methods for comparison")
            return comparison_results

        # Statistical comparison
        print(f"\n📊 STATISTICAL COMPARISON")
        print("-" * 40)

        # Create comparison table
        methods = list(comparison_results.keys())
        metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity']

        print(
            f"{'Method':<15} {'Precision@3':<12} {'Recall@3':<10} {'NDCG@3':<10} {'Diversity':<10} {'Success Rate':<12}")
        print("-" * 80)

        for method in methods:
            results = comparison_results[method]
            precision = results['precision_at_k']['mean']
            recall = results['recall_at_k']['mean']
            ndcg = results['ndcg_at_k']['mean']
            diversity = results['diversity']['mean']
            success_rate = results['success_rate']

            print(
                f"{method:<15} {precision:<12.4f} {recall:<10.4f} {ndcg:<10.4f} {diversity:<10.4f} {success_rate:<12.3f}")

        # Find best performing method for each metric
        print(f"\n🥇 BEST PERFORMERS BY METRIC:")
        print("-" * 40)

        for metric in metrics:
            best_method = max(comparison_results.items(),
                              key=lambda x: x[1][metric]['mean'])
            best_score = best_method[1][metric]['mean']
            print(f"{metric.replace('_', ' ').title()}: {best_method[0]} ({best_score:.4f})")

        return comparison_results

    def create_visualizations(self, comparison_results: Dict, output_dir: str = "baseline_evaluation"):
        """Create comprehensive visualizations"""
        os.makedirs(output_dir, exist_ok=True)

        if not comparison_results:
            print("⚠️ No results to visualize")
            return

        print(f"\n📈 Creating Visualizations...")

        # 1. Performance Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Methods Performance Comparison', fontsize=16, fontweight='bold')

        methods = list(comparison_results.keys())
        metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity']
        metric_titles = ['Precision@3', 'Recall@3', 'NDCG@3', 'Diversity']

        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            means = [comparison_results[method][metric]['mean'] for method in methods]
            stds = [comparison_results[method][metric]['std'] for method in methods]

            bars = ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.8,
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')

            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Score Distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Score Distributions by Method', fontsize=16, fontweight='bold')

        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            data_for_plot = []
            labels_for_plot = []

            for method in methods:
                scores = comparison_results[method][metric]['scores']
                if scores:  # Only add if we have scores
                    data_for_plot.append(scores)
                    labels_for_plot.append(method)

            if data_for_plot:
                box_plot = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)

                # Color the boxes
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for patch, color in zip(box_plot['boxes'], colors[:len(data_for_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Success Rate Comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        success_rates = [comparison_results[method]['success_rate'] for method in methods]
        valid_evals = [comparison_results[method]['valid_evaluations'] for method in methods]
        total_results = [comparison_results[method]['total_results'] for method in methods]

        bars = ax.bar(methods, success_rates, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

        # Add labels
        for bar, rate, valid, total in zip(bars, success_rates, valid_evals, total_results):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{rate:.1%}\n({valid}/{total})',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_title('Success Rate by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rates.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📁 Visualizations saved to {output_dir}/")

    def generate_report(self, comparison_results: Dict, output_file: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"baseline_evaluation/evaluation_report_{timestamp}.txt"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BASELINE METHODS COMPREHENSIVE EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Methods Evaluated: {len(comparison_results)}")
        report_lines.append("")

        if not comparison_results:
            report_lines.append("❌ No valid evaluation results found.")
            report_text = "\n".join(report_lines)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            return report_text

        # Method Performance Summary
        report_lines.append("METHOD PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Method':<15} {'Precision@3':<12} {'Recall@3':<10} {'NDCG@3':<10} {'Diversity':<10}")
        report_lines.append("-" * 70)

        for method_name, results in comparison_results.items():
            precision = results['precision_at_k']['mean']
            recall = results['recall_at_k']['mean']
            ndcg = results['ndcg_at_k']['mean']
            diversity = results['diversity']['mean']

            report_lines.append(
                f"{method_name:<15} {precision:<12.4f} {recall:<10.4f} {ndcg:<10.4f} {diversity:<10.4f}")

        report_lines.append("")

        # Detailed Analysis
        report_lines.append("DETAILED ANALYSIS")
        report_lines.append("-" * 40)

        for method_name, results in comparison_results.items():
            report_lines.append(f"\n{method_name.upper()}:")
            report_lines.append(
                f"  Success Rate: {results['success_rate']:.1%} ({results['valid_evaluations']}/{results['total_results']})")

            for metric in ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity']:
                mean_val = results[metric]['mean']
                std_val = results[metric]['std']
                median_val = results[metric]['median']

                report_lines.append(f"  {metric.replace('_', ' ').title()}:")
                report_lines.append(f"    Mean: {mean_val:.4f} ± {std_val:.4f}")
                report_lines.append(f"    Median: {median_val:.4f}")

        # Best Performers
        report_lines.append(f"\nBEST PERFORMERS")
        report_lines.append("-" * 40)

        metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'diversity']
        for metric in metrics:
            best_method = max(comparison_results.items(),
                              key=lambda x: x[1][metric]['mean'])
            best_score = best_method[1][metric]['mean']
            report_lines.append(f"🏆 {metric.replace('_', ' ').title()}: {best_method[0]} ({best_score:.4f})")

        # Insights and Recommendations
        report_lines.append(f"\nINSIGHTS AND RECOMMENDATIONS")
        report_lines.append("-" * 40)

        # Analyze results
        methods = list(comparison_results.keys())

        if 'Popularity' in comparison_results:
            pop_precision = comparison_results['Popularity']['precision_at_k']['mean']
            if pop_precision > 0.1:
                report_lines.append("✅ Popularity-based method shows strong baseline performance")
            else:
                report_lines.append("⚠️ Popularity-based method shows limited effectiveness")

        if 'Content-Based' in comparison_results:
            cb_precision = comparison_results['Content-Based']['precision_at_k']['mean']
            cb_success = comparison_results['Content-Based']['success_rate']
            if cb_success < 0.5:
                report_lines.append("⚠️ Content-based method has low success rate - may need more content features")
            if cb_precision > 0.05:
                report_lines.append("✅ Content-based method shows reasonable precision when successful")

        if 'Random' in comparison_results:
            random_precision = comparison_results['Random']['precision_at_k']['mean']
            report_lines.append(f"📊 Random baseline precision: {random_precision:.4f} (for comparison)")

        # Overall assessment
        best_overall = max(comparison_results.items(),
                           key=lambda x: x[1]['precision_at_k']['mean'])
        report_lines.append(f"\n🎯 OVERALL BEST METHOD: {best_overall[0]}")
        report_lines.append(f"   Precision@3: {best_overall[1]['precision_at_k']['mean']:.4f}")
        report_lines.append(f"   Success Rate: {best_overall[1]['success_rate']:.1%}")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"📋 Comprehensive report saved to: {output_file}")
        return report_text


def main():
    """Main function to run baseline evaluation"""
    print("🔬 BASELINE METHODS COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Initialize evaluator
    evaluator = BaselineEvaluator()

    # Define result files
    result_files = [
        "results/content_based_results.json",
        "results/popularity_results.json",
        "results/random_results.json"
    ]

    # Load results
    all_results = evaluator.load_results(result_files)

    if not all_results:
        print("❌ No result files found. Please run baseline evaluation first:")
        print("   python baselines/content_based.py --dataset [path] --user_history [path] --domain [domain]")
        return

    # Compare methods
    comparison_results = evaluator.compare_methods(all_results, k=3)

    if comparison_results:
        # Create visualizations
        evaluator.create_visualizations(comparison_results)

        # Generate report
        report = evaluator.generate_report(comparison_results)

        # Print summary
        print(f"\n🎯 EVALUATION SUMMARY")
        print("-" * 30)
        print(f"Methods Evaluated: {len(comparison_results)}")

        if comparison_results:
            best_method = max(comparison_results.items(),
                              key=lambda x: x[1]['precision_at_k']['mean'])
            print(f"Best Method: {best_method[0]} (Precision@3: {best_method[1]['precision_at_k']['mean']:.4f})")

        print(f"\n✅ Evaluation Complete!")
        print(f"📁 Results saved to: baseline_evaluation/")
        print(f"📊 Visualizations: baseline_evaluation/*.png")
        print(f"📋 Report: baseline_evaluation/evaluation_report_*.txt")

    else:
        print("❌ No valid evaluation results generated")


if __name__ == "__main__":
    main()