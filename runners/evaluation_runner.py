#!/usr/bin/env python3
"""
Evaluation Runner - Easy interface to run quantitative evaluations
Complete working version with all features
"""

import argparse
import os
import json
import glob
from datetime import datetime
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.quantitative_evaluator import RecommendationEvaluator, run_comprehensive_evaluation


def find_result_files(results_dir: str = "results") -> list:
    """Find all result JSON files in the results directory."""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []

    # Look for various result file patterns
    patterns = [
        "batch_test_results_*.json",
        "interactive_session_*.json",
        "ollama_llamarec_results_*.json",
        "*_results_*.json"
    ]

    result_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(results_dir, pattern))
        result_files.extend(files)

    # Remove duplicates and sort by modification time (newest first)
    result_files = list(set(result_files))
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return result_files


def interactive_evaluation():
    """Interactive evaluation interface."""
    print("Interactive Quantitative Evaluation")
    print("=" * 50)

    # Find result files
    result_files = find_result_files()

    if not result_files:
        print("No result files found in results/ directory")
        print("Please run some LLAMAREC tests first to generate results.")
        return

    print(f"Found {len(result_files)} result files:")
    for i, file_path in enumerate(result_files, 1):
        file_name = os.path.basename(file_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"  {i}. {file_name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")

    print("\nEvaluation Options:")
    print("1. Evaluate single result file")
    print("2. Compare multiple result files")
    print("3. Evaluate all result files")
    print("4. Quick evaluation of latest file")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        evaluate_single_file(result_files)
    elif choice == "2":
        compare_multiple_files(result_files)
    elif choice == "3":
        evaluate_all_files(result_files)
    elif choice == "4":
        quick_evaluation(result_files[0] if result_files else None)
    else:
        print("❌ Invalid choice. Please try again.")


def evaluate_single_file(result_files: list):
    """Evaluate a single result file."""
    print("\n📊 Single File Evaluation")
    print("-" * 30)

    # Select file
    while True:
        try:
            choice = int(input(f"Select file (1-{len(result_files)}): "))
            if 1 <= choice <= len(result_files):
                selected_file = result_files[choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(result_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

    # Get evaluation parameters
    k = int(input("Top-K for evaluation (default 3): ") or "3")
    output_name = input("Output name (default: auto-generated): ").strip()

    if not output_name:
        file_base = os.path.basename(selected_file).replace('.json', '')
        output_name = f"eval_{file_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🔄 Evaluating {os.path.basename(selected_file)}...")

    # Run evaluation
    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate_recommendations(selected_file, k=k)

    if results:
        # Create output directory
        output_dir = f"evaluation_results/{output_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Generate report
        report = evaluator.generate_evaluation_report(
            results,
            f"{output_dir}/evaluation_report.txt"
        )

        print(report)

        # Create visualizations
        evaluator.create_visualization(results, f"{output_dir}/plots")

        # Save detailed results
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Evaluation complete! Results saved to {output_dir}/")
    else:
        print("❌ Evaluation failed - no valid results found")


def compare_multiple_files(result_files: list):
    """Compare multiple result files."""
    print("\n🆚 Multi-System Comparison")
    print("-" * 30)

    # Select files
    selected_files = []
    system_names = []

    print("Select files to compare (enter numbers separated by spaces):")
    while True:
        try:
            choices = input(f"Files (1-{len(result_files)}): ").strip().split()
            file_indices = [int(c) - 1 for c in choices]

            if all(0 <= i < len(result_files) for i in file_indices) and len(file_indices) >= 2:
                selected_files = [result_files[i] for i in file_indices]
                break
            else:
                print("❌ Please select at least 2 valid files")
        except ValueError:
            print("❌ Please enter valid numbers separated by spaces")

    # Get system names
    print("\nEnter names for each system (or press Enter for auto-names):")
    for i, file_path in enumerate(selected_files):
        default_name = f"System_{i + 1}"
        name = input(f"Name for {os.path.basename(file_path)} (default: {default_name}): ").strip()
        system_names.append(name or default_name)

    # Get parameters
    k = int(input("Top-K for evaluation (default 3): ") or "3")

    output_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"evaluation_results/{output_name}"

    print(f"\n🔄 Comparing {len(selected_files)} systems...")

    # Run comparison
    run_comprehensive_evaluation(
        selected_files,
        system_names,
        k=k,
        output_dir=output_dir
    )

    print(f"\n✅ Comparison complete! Results saved to {output_dir}/")


def evaluate_all_files(result_files: list):
    """Evaluate all result files."""
    print(f"\n📈 Evaluating All {len(result_files)} Files")
    print("-" * 40)

    k = int(input("Top-K for evaluation (default 3): ") or "3")

    output_name = f"all_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"evaluation_results/{output_name}"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = RecommendationEvaluator()
    all_results = {}

    for i, result_file in enumerate(result_files, 1):
        file_name = os.path.basename(result_file).replace('.json', '')
        print(f"\n🔄 [{i}/{len(result_files)}] Evaluating {file_name}...")

        results = evaluator.evaluate_recommendations(result_file, k=k)

        if results:
            all_results[file_name] = results

            # Individual report
            evaluator.generate_evaluation_report(
                results,
                f"{output_dir}/{file_name}_report.txt"
            )

            print(f"✅ {file_name} evaluated successfully")
        else:
            print(f"❌ {file_name} evaluation failed")

    if len(all_results) > 1:
        # Generate comparison
        print(f"\n🆚 Generating comparison of {len(all_results)} systems...")

        valid_files = [f for f in result_files if os.path.basename(f).replace('.json', '') in all_results]
        valid_names = list(all_results.keys())

        run_comprehensive_evaluation(
            valid_files,
            valid_names,
            k=k,
            output_dir=output_dir
        )

    # Summary report
    generate_summary_report(all_results, f"{output_dir}/summary_report.txt", k)

    print(f"\n✅ All evaluations complete! Results saved to {output_dir}/")


def quick_evaluation(result_file: str):
    """Quick evaluation of a single file with minimal options."""
    if not result_file:
        print("❌ No result files available")
        return

    print(f"\n⚡ Quick Evaluation: {os.path.basename(result_file)}")
    print("-" * 50)

    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate_recommendations(result_file, k=3)

    if results:
        # Print quick summary
        metrics = results['overall_metrics']
        info = results['evaluation_info']

        print(f"📊 Results Summary (Top-3):")
        print(f"  Valid Results: {info['valid_results']}/{info['total_results']}")
        print(f"  Precision@3: {metrics['precision_at_k']['mean']:.4f} ± {metrics['precision_at_k']['std']:.4f}")
        print(f"  Recall@3: {metrics['recall_at_k']['mean']:.4f} ± {metrics['recall_at_k']['std']:.4f}")
        print(f"  NDCG@3: {metrics['ndcg_at_k']['mean']:.4f} ± {metrics['ndcg_at_k']['std']:.4f}")
        print(f"  Diversity: {metrics['diversity']['mean']:.4f}")
        print(f"  Novelty: {metrics['novelty']['mean']:.4f}")

        # Coverage info
        coverage = results.get('coverage_by_domain', {})
        if coverage:
            print(f"  Coverage by Domain:")
            for domain, score in coverage.items():
                print(f"    {domain}: {score:.4f}")

        # Performance interpretation
        print(f"\n💡 Quick Interpretation:")
        precision_mean = metrics['precision_at_k']['mean']
        if precision_mean > 0.1:
            print(f"  ✅ Good precision ({precision_mean:.3f}) - Recommendations are relevant")
        elif precision_mean > 0.05:
            print(f"  ⚠️  Moderate precision ({precision_mean:.3f}) - Some relevant recommendations")
        else:
            print(f"  ❌ Low precision ({precision_mean:.3f}) - Few relevant recommendations")

        ndcg_mean = metrics['ndcg_at_k']['mean']
        if ndcg_mean > 0.2:
            print(f"  ✅ Good ranking quality ({ndcg_mean:.3f})")
        elif ndcg_mean > 0.1:
            print(f"  ⚠️  Moderate ranking quality ({ndcg_mean:.3f})")
        else:
            print(f"  ❌ Poor ranking quality ({ndcg_mean:.3f})")

        # Save quick results
        output_dir = f"evaluation_results/quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/quick_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Quick results saved to {output_dir}/")
    else:
        print("❌ Quick evaluation failed")


def generate_summary_report(all_results: dict, output_file: str, k: int):
    """Generate a summary report across all evaluations."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE EVALUATION SUMMARY")
    report_lines.append("=" * 80)

    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Systems Evaluated: {len(all_results)}")
    report_lines.append(f"Top-K: {k}")
    report_lines.append("")

    if not all_results:
        report_lines.append("No valid evaluation results found.")
        with open(output_file, 'w') as f:
            f.write("\n".join(report_lines))
        return

    # Collect all metrics
    system_metrics = {}
    for system_name, results in all_results.items():
        metrics = results.get('overall_metrics', {})
        system_metrics[system_name] = {
            'precision': metrics.get('precision_at_k', {}).get('mean', 0),
            'recall': metrics.get('recall_at_k', {}).get('mean', 0),
            'ndcg': metrics.get('ndcg_at_k', {}).get('mean', 0),
            'diversity': metrics.get('diversity', {}).get('mean', 0),
            'novelty': metrics.get('novelty', {}).get('mean', 0),
            'valid_results': results.get('evaluation_info', {}).get('valid_results', 0)
        }

    # Rankings
    report_lines.append("SYSTEM RANKINGS")
    report_lines.append("-" * 40)

    metric_names = ['precision', 'recall', 'ndcg', 'diversity', 'novelty']

    for metric in metric_names:
        sorted_systems = sorted(system_metrics.items(),
                                key=lambda x: x[1][metric], reverse=True)

        report_lines.append(f"\n{metric.upper()}@{k} Ranking:")
        for rank, (system, metrics) in enumerate(sorted_systems, 1):
            score = metrics[metric]
            report_lines.append(f"  {rank}. {system}: {score:.4f}")

    # Overall performance table
    report_lines.append(f"\nOVERALL PERFORMANCE TABLE")
    report_lines.append("-" * 40)
    report_lines.append(
        f"{'System':<20} {'Prec@' + str(k):<8} {'Rec@' + str(k):<8} {'NDCG@' + str(k):<8} {'Diversity':<10} {'Novelty':<8} {'Valid':<6}")
    report_lines.append("-" * 80)

    for system_name, metrics in system_metrics.items():
        report_lines.append(
            f"{system_name[:19]:<20} "
            f"{metrics['precision']:<8.4f} "
            f"{metrics['recall']:<8.4f} "
            f"{metrics['ndcg']:<8.4f} "
            f"{metrics['diversity']:<10.4f} "
            f"{metrics['novelty']:<8.4f} "
            f"{metrics['valid_results']:<6d}"
        )

    # Best performing system
    best_system = max(system_metrics.items(), key=lambda x: x[1]['ndcg'])
    report_lines.append(f"\n🏆 BEST OVERALL SYSTEM (by NDCG@{k}): {best_system[0]}")
    report_lines.append(f"   NDCG@{k}: {best_system[1]['ndcg']:.4f}")
    report_lines.append(f"   Precision@{k}: {best_system[1]['precision']:.4f}")
    report_lines.append(f"   Recall@{k}: {best_system[1]['recall']:.4f}")

    # Recommendations
    report_lines.append(f"\n📋 RECOMMENDATIONS")
    report_lines.append("-" * 40)

    avg_precision = sum(m['precision'] for m in system_metrics.values()) / len(system_metrics)
    avg_ndcg = sum(m['ndcg'] for m in system_metrics.values()) / len(system_metrics)

    if avg_precision < 0.05:
        report_lines.append("• Consider improving recommendation relevance")
        report_lines.append("• Review prompt engineering and item extraction")

    if avg_ndcg < 0.1:
        report_lines.append("• Focus on improving ranking quality")
        report_lines.append("• Consider adding ranking-specific training")

    max_diversity = max(m['diversity'] for m in system_metrics.values())
    if max_diversity < 0.5:
        report_lines.append("• Consider adding diversity constraints")
        report_lines.append("• Explore different recommendation strategies")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))


def batch_evaluation_mode(args):
    """Batch evaluation mode for automation."""
    print("🤖 Batch Evaluation Mode")
    print("=" * 30)

    if args.files:
        result_files = args.files
    else:
        result_files = find_result_files(args.results_dir)

    if not result_files:
        print("❌ No result files found")
        return

    print(f"📁 Processing {len(result_files)} files")

    if len(result_files) == 1:
        # Single file evaluation
        evaluator = RecommendationEvaluator(args.user_history)
        results = evaluator.evaluate_recommendations(result_files[0], k=args.k)

        if results:
            output_dir = args.output or f"evaluation_results/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)

            # Generate report and visualizations
            evaluator.generate_evaluation_report(results, f"{output_dir}/evaluation_report.txt")
            evaluator.create_visualization(results, f"{output_dir}/plots")

            with open(f"{output_dir}/detailed_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            print(f"✅ Evaluation complete: {output_dir}/")
        else:
            print("❌ Evaluation failed")

    else:
        # Multi-file comparison
        system_names = args.names or [f"System_{i + 1}" for i in range(len(result_files))]
        output_dir = args.output or f"evaluation_results/batch_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_comprehensive_evaluation(
            result_files,
            system_names,
            user_history_path=args.user_history,
            k=args.k,
            output_dir=output_dir
        )

        print(f"✅ Comparison complete: {output_dir}/")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Quantitative Evaluation for Cross-Domain Recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python evaluation_runner.py

  # Batch evaluation of specific files
  python evaluation_runner.py --batch --files results/file1.json results/file2.json

  # Evaluate all files in results directory
  python evaluation_runner.py --batch --results-dir results/

  # Quick evaluation with custom parameters
  python evaluation_runner.py --batch --k 5 --output my_evaluation/
        """
    )

    parser.add_argument("--batch", action="store_true",
                        help="Run in batch mode (non-interactive)")

    parser.add_argument("--files", nargs="+",
                        help="Specific result files to evaluate")

    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result files (default: results)")

    parser.add_argument("--user-history", default="data/splits/user_history.json",
                        help="Path to user history JSON file")

    parser.add_argument("--k", type=int, default=3,
                        help="Top-K for evaluation metrics (default: 3)")

    parser.add_argument("--output",
                        help="Output directory for results")

    parser.add_argument("--names", nargs="+",
                        help="System names for comparison (when using --files)")

    args = parser.parse_args()

    # Check if user history exists
    if not os.path.exists(args.user_history):
        print(f"❌ User history file not found: {args.user_history}")
        print("Please run the data preprocessing pipeline first.")
        return

    if args.batch:
        batch_evaluation_mode(args)
    else:
        interactive_evaluation()


if __name__ == "__main__":
    main()