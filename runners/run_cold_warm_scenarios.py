#!/usr/bin/env python3
"""
Cold/Warm User Scenario Runner - Fixed Version
Specialized script for testing Books <-> Movies_and_TV scenarios only
"""

import os
import json
import pandas as pd
import requests
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the main tester
from testers.test_prompt_generator_cold_warm import PromptGeneratorTester


class ColdWarmScenarioRunner:
    """
    Specialized runner for cold-start and warm user scenarios.
    Fixed to only work with Books and Movies_and_TV domains.
    """

    def __init__(self, tester: PromptGeneratorTester):
        """Initialize with a PromptGeneratorTester instance."""
        self.tester = tester
        self.scenarios = []

    def create_cold_start_scenarios(self) -> List[Dict]:
        """
        Create specific cold-start scenarios for testing.
        Only Books <-> Movies_and_TV scenarios.
        """
        scenarios = [
            {
                "name": "New User - Books to Movies",
                "description": "Users with minimal book interactions recommending movies",
                "user_type": "cold",
                "source_domain": "Books",
                "target_domain": "Movies_and_TV",
                "max_source_interactions": 10,
                "min_source_interactions": 3
            },
            {
                "name": "New User - Movies to Books",
                "description": "Users with minimal movie interactions recommending books",
                "user_type": "cold",
                "source_domain": "Movies_and_TV",
                "target_domain": "Books",
                "max_source_interactions": 10,
                "min_source_interactions": 3
            }
        ]

        return scenarios

    def create_warm_user_scenarios(self) -> List[Dict]:
        """
        Create warm user scenarios for comparison.
        Only Books <-> Movies_and_TV scenarios.
        """
        scenarios = [
            {
                "name": "Expert Reader - Books to Movies",
                "description": "Heavy book readers getting movie recommendations",
                "user_type": "warm",
                "source_domain": "Books",
                "target_domain": "Movies_and_TV",
                "min_source_interactions": 50,
                "max_source_interactions": 1000
            },
            {
                "name": "Movie Expert - Movies to Books",
                "description": "Movie enthusiasts getting book recommendations",
                "user_type": "warm",
                "source_domain": "Movies_and_TV",
                "target_domain": "Books",
                "min_source_interactions": 50,
                "max_source_interactions": 1000
            }
        ]

        return scenarios

    def find_users_for_scenario(self, scenario: Dict, max_users: int = 15) -> List[str]:
        """
        Find users that match a specific scenario criteria.
        """
        # Get users of the right type
        if scenario['user_type'] == 'cold':
            candidate_users = set(self.tester.cold_warm_users.get('cold_users', []))
        else:
            candidate_users = set(self.tester.cold_warm_users.get('warm_users', []))

        if not candidate_users:
            return []

        # Find the right domain pair
        source_domain = scenario['source_domain']
        target_domain = scenario['target_domain']
        pair_name = f"{source_domain}_to_{target_domain}"

        if pair_name not in self.tester.cross_domain_data:
            print(f"Warning: Domain pair {pair_name} not found")
            return []

        # Get users who have interactions in both domains
        pair_data = self.tester.cross_domain_data[pair_name]
        source_users = set(pair_data['source_df']['reviewerID'].unique())
        target_users = set(pair_data['target_df']['reviewerID'].unique())

        eligible_users = candidate_users.intersection(source_users).intersection(target_users)

        # Filter by interaction count if specified
        user_counts = self.tester.cold_warm_users.get('user_interaction_counts', {})
        filtered_users = []

        for user_id in eligible_users:
            user_count = user_counts.get(user_id, 0)

            # Check against scenario criteria
            min_interactions = scenario.get('min_source_interactions', 0)
            max_interactions = scenario.get('max_source_interactions', float('inf'))

            if min_interactions <= user_count <= max_interactions:
                filtered_users.append(user_id)

        # Return up to max_users
        return filtered_users[:max_users]

    def run_scenario(self, scenario: Dict, max_users: int = 15, temperature: float = 0.7) -> Dict:
        """
        Run a specific scenario.
        """
        print(f"\nRunning Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 50)

        # Find users for this scenario
        users = self.find_users_for_scenario(scenario, max_users)

        if not users:
            print(f"No users found for scenario: {scenario['name']}")
            return {"error": "No eligible users found"}

        print(f"Found {len(users)} eligible users")

        # Generate prompts for these specific users
        source_domain = scenario['source_domain']
        target_domain = scenario['target_domain']
        pair_name = f"{source_domain}_to_{target_domain}"

        pair_data = self.tester.cross_domain_data[pair_name]
        source_df = pair_data['source_df']
        target_df = pair_data['target_df']

        # Filter dataframes to our users
        source_df_filtered = source_df[source_df['reviewerID'].isin(users)]
        target_df_filtered = target_df[target_df['reviewerID'].isin(users)]

        # Generate prompts using the existing function
        from prompts.prompt_generator import get_item_descriptions, generate_prompts

        source_map = get_item_descriptions(source_df_filtered)
        target_map = get_item_descriptions(target_df_filtered)

        prompts = generate_prompts(
            source_df_filtered, target_df_filtered,
            source_map, target_map,
            source_domain, target_domain,
            max_prompts=len(users)
        )

        # Add scenario metadata to prompts
        for prompt in prompts:
            prompt['scenario_name'] = scenario['name']
            prompt['user_type'] = scenario['user_type']
            prompt['pair_name'] = pair_name

            # Add user interaction count
            user_counts = self.tester.cold_warm_users.get('user_interaction_counts', {})
            prompt['user_interaction_count'] = user_counts.get(prompt['user_id'], 0)

        # Test with Ollama
        print(f"Testing {len(prompts)} prompts with Ollama...")
        results = self.tester.test_prompts_with_ollama(prompts, temperature, save_intermediate=False)

        # Analyze results
        scenario_analysis = self.analyze_scenario_results(results, scenario)

        return {
            "scenario": scenario,
            "results": results,
            "analysis": scenario_analysis,
            "user_count": len(users),
            "success_rate": float(sum(1 for r in results if r['success']) / len(results)) if results else 0.0
        }

    def analyze_scenario_results(self, results: List[Dict], scenario: Dict) -> Dict:
        """
        Analyze results for a specific scenario.
        """
        successful_results = [r for r in results if r['success']]

        if not successful_results:
            return {"error": "No successful results to analyze"}

        # Basic statistics
        response_times = [r['response_time'] for r in successful_results]
        token_counts = [r['tokens_estimated'] for r in successful_results]
        interaction_counts = [r['user_interaction_count'] for r in successful_results]

        # Quality analysis
        quality_scores = []
        for result in successful_results:
            quality = self.tester.estimate_response_quality(result['ollama_response'])
            quality_scores.append(quality)

        # Response format analysis
        format_analysis = self.analyze_response_formats(successful_results)

        analysis = {
            "basic_stats": {
                "successful_responses": len(successful_results),
                "total_responses": len(results),
                "success_rate": float(len(successful_results) / len(results)),
                "avg_response_time": float(sum(response_times) / len(response_times)),
                "avg_tokens": float(sum(token_counts) / len(token_counts)),
                "avg_user_interactions": float(sum(interaction_counts) / len(interaction_counts)),
                "avg_quality_score": float(sum(quality_scores) / len(quality_scores))
            },
            "quality_distribution": {
                "excellent": int(sum(1 for q in quality_scores if q >= 4.0)),
                "good": int(sum(1 for q in quality_scores if 3.0 <= q < 4.0)),
                "fair": int(sum(1 for q in quality_scores if 2.0 <= q < 3.0)),
                "poor": int(sum(1 for q in quality_scores if q < 2.0))
            },
            "format_analysis": format_analysis,
            "sample_responses": [r['ollama_response'][:200] + "..." for r in successful_results[:3]]
        }

        return analysis

    def analyze_response_formats(self, results: List[Dict]) -> Dict:
        """
        Analyze the format of generated responses.
        """
        format_stats = {
            "has_numbered_list": 0,
            "has_explanations": 0,
            "has_quoted_titles": 0,
            "has_bold_text": 0,
            "appropriate_length": 0,
            "mentions_domains": 0
        }

        for result in results:
            response = result['ollama_response'].lower()

            # Check for numbered list
            if any(f"{i}." in response for i in range(1, 6)):
                format_stats["has_numbered_list"] += 1

            # Check for explanations
            explanation_words = ["because", "since", "due to", "matches", "similar", "appeals", "like"]
            if any(word in response for word in explanation_words):
                format_stats["has_explanations"] += 1

            # Check for quoted titles
            if '"' in result['ollama_response']:
                format_stats["has_quoted_titles"] += 1

            # Check for bold text
            if "**" in result['ollama_response']:
                format_stats["has_bold_text"] += 1

            # Check length appropriateness
            word_count = len(result['ollama_response'].split())
            if 50 <= word_count <= 200:
                format_stats["appropriate_length"] += 1

            # Check for domain mentions
            domains = ["book", "movie", "film", "novel"]
            if any(domain in response for domain in domains):
                format_stats["mentions_domains"] += 1

        total = len(results)
        return {key: {"count": int(count), "percentage": float(count / total * 100)}
                for key, count in format_stats.items()}

    def run_comparative_study(self, max_users_per_scenario: int = 10, temperature: float = 0.7) -> Dict:
        """
        Run a comprehensive comparative study between cold and warm users.
        """
        print("\nCOMPREHENSIVE COLD vs WARM USER STUDY")
        print("=" * 60)

        # Get all scenarios
        cold_scenarios = self.create_cold_start_scenarios()
        warm_scenarios = self.create_warm_user_scenarios()

        all_results = []
        scenario_results = {}

        # Run cold scenarios
        print("\nRUNNING COLD USER SCENARIOS")
        print("-" * 40)

        for scenario in cold_scenarios:
            result = self.run_scenario(scenario, max_users_per_scenario, temperature)
            if "error" not in result:
                all_results.extend(result['results'])
                scenario_results[scenario['name']] = result

        # Run warm scenarios
        print("\nRUNNING WARM USER SCENARIOS")
        print("-" * 40)

        for scenario in warm_scenarios:
            result = self.run_scenario(scenario, max_users_per_scenario, temperature)
            if "error" not in result:
                all_results.extend(result['results'])
                scenario_results[scenario['name']] = result

        # Comprehensive analysis
        comprehensive_analysis = self.compare_cold_warm_scenarios(scenario_results)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/cold_warm_comparative_study_{timestamp}.json"

        study_data = {
            "study_info": {
                "timestamp": datetime.now().isoformat(),
                "model": self.tester.ollama_model,
                "temperature": float(temperature),
                "total_scenarios": len(scenario_results),
                "total_users_tested": len(set(r['user_id'] for r in all_results)),
                "total_prompts": len(all_results)
            },
            "scenario_results": scenario_results,
            "comparative_analysis": comprehensive_analysis,
            "all_results": all_results
        }

        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(study_data, f, indent=2, ensure_ascii=False)

        print(f"\nStudy results saved to: {results_file}")

        # Generate report
        self.generate_comparative_report(study_data)

        return study_data

    def compare_cold_warm_scenarios(self, scenario_results: Dict) -> Dict:
        """
        Compare cold vs warm scenario results.
        """
        cold_results = []
        warm_results = []

        for scenario_name, result in scenario_results.items():
            if result['scenario']['user_type'] == 'cold':
                cold_results.extend([r for r in result['results'] if r['success']])
            else:
                warm_results.extend([r for r in result['results'] if r['success']])

        if not cold_results or not warm_results:
            return {"error": "Insufficient data for comparison"}

        # Calculate metrics
        cold_metrics = self.calculate_group_metrics(cold_results)
        warm_metrics = self.calculate_group_metrics(warm_results)

        comparison = {
            "cold_metrics": cold_metrics,
            "warm_metrics": warm_metrics,
            "differences": {
                "response_time": float(warm_metrics['avg_response_time'] - cold_metrics['avg_response_time']),
                "tokens": float(warm_metrics['avg_tokens'] - cold_metrics['avg_tokens']),
                "quality": float(warm_metrics['avg_quality'] - cold_metrics['avg_quality']),
                "user_interactions": float(warm_metrics['avg_user_interactions'] - cold_metrics['avg_user_interactions'])
            },
            "sample_sizes": {
                "cold": len(cold_results),
                "warm": len(warm_results)
            }
        }

        # Statistical tests if scipy is available
        try:
            from scipy import stats
            cold_qualities = [self.tester.estimate_response_quality(r['ollama_response']) for r in cold_results]
            warm_qualities = [self.tester.estimate_response_quality(r['ollama_response']) for r in warm_results]

            t_stat, p_value = stats.ttest_ind(cold_qualities, warm_qualities)
            comparison['statistical_test'] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }
        except ImportError:
            comparison['statistical_test'] = {"error": "scipy not available"}

        return comparison

    def calculate_group_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for a group of results."""
        response_times = [r['response_time'] for r in results]
        token_counts = [r['tokens_estimated'] for r in results]
        interaction_counts = [r['user_interaction_count'] for r in results]
        quality_scores = [self.tester.estimate_response_quality(r['ollama_response']) for r in results]

        return {
            "count": len(results),
            "avg_response_time": float(sum(response_times) / len(response_times)),
            "avg_tokens": float(sum(token_counts) / len(token_counts)),
            "avg_user_interactions": float(sum(interaction_counts) / len(interaction_counts)),
            "avg_quality": float(sum(quality_scores) / len(quality_scores)),
            "quality_scores": [float(q) for q in quality_scores]
        }

    def generate_comparative_report(self, study_data: Dict):
        """Generate a detailed comparative study report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"results/cold_warm_study_report_{timestamp}.txt"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COLD vs WARM USER COMPARATIVE STUDY REPORT")
        report_lines.append("=" * 80)

        info = study_data['study_info']
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {info['model']}")
        report_lines.append(f"Temperature: {info['temperature']}")
        report_lines.append(f"Total Scenarios: {info['total_scenarios']}")
        report_lines.append(f"Total Users: {info['total_users_tested']}")
        report_lines.append(f"Total Prompts: {info['total_prompts']}")
        report_lines.append("")

        # Scenario results
        report_lines.append("SCENARIO RESULTS")
        report_lines.append("-" * 40)

        for scenario_name, result in study_data['scenario_results'].items():
            analysis = result['analysis']
            basic_stats = analysis['basic_stats']

            report_lines.append(f"\n{scenario_name.upper()}")
            report_lines.append(f"  User Type: {result['scenario']['user_type']}")
            report_lines.append(f"  Success Rate: {basic_stats['success_rate']:.1%}")
            report_lines.append(f"  Avg Quality: {basic_stats['avg_quality_score']:.2f}/5.0")
            report_lines.append(f"  Avg Response Time: {basic_stats['avg_response_time']:.2f}s")
            report_lines.append(f"  Users Tested: {basic_stats['successful_responses']}")

        # Comparative analysis
        comp_analysis = study_data['comparative_analysis']
        if 'error' not in comp_analysis:
            report_lines.append(f"\nCOMPARATIVE ANALYSIS")
            report_lines.append("-" * 40)

            cold_metrics = comp_analysis['cold_metrics']
            warm_metrics = comp_analysis['warm_metrics']
            differences = comp_analysis['differences']

            report_lines.append(f"\nCOLD USERS ({cold_metrics['count']} samples):")
            report_lines.append(f"  Avg Quality: {cold_metrics['avg_quality']:.2f}")
            report_lines.append(f"  Avg Response Time: {cold_metrics['avg_response_time']:.2f}s")
            report_lines.append(f"  Avg User Interactions: {cold_metrics['avg_user_interactions']:.1f}")

            report_lines.append(f"\nWARM USERS ({warm_metrics['count']} samples):")
            report_lines.append(f"  Avg Quality: {warm_metrics['avg_quality']:.2f}")
            report_lines.append(f"  Avg Response Time: {warm_metrics['avg_response_time']:.2f}s")
            report_lines.append(f"  Avg User Interactions: {warm_metrics['avg_user_interactions']:.1f}")

            report_lines.append(f"\nDIFFERENCES (Warm - Cold):")
            report_lines.append(f"  Quality: {differences['quality']:+.2f}")
            report_lines.append(f"  Response Time: {differences['response_time']:+.2f}s")
            report_lines.append(f"  User Experience: {differences['user_interactions']:+.1f} interactions")

            # Statistical significance
            if 'statistical_test' in comp_analysis and 'error' not in comp_analysis['statistical_test']:
                stat_test = comp_analysis['statistical_test']
                significance = "SIGNIFICANT" if stat_test['significant'] else "Not significant"
                report_lines.append(f"\nSTATISTICAL TEST:")
                report_lines.append(f"  T-statistic: {stat_test['t_statistic']:.3f}")
                report_lines.append(f"  P-value: {stat_test['p_value']:.4f}")
                report_lines.append(f"  Result: {significance}")

        # Key findings
        report_lines.append(f"\nKEY FINDINGS")
        report_lines.append("-" * 40)
        if 'error' not in comp_analysis:
            quality_diff = comp_analysis['differences']['quality']
            if quality_diff > 0.2:
                report_lines.append("- Warm users receive significantly higher quality recommendations")
            elif quality_diff < -0.2:
                report_lines.append("- Cold users surprisingly receive higher quality recommendations")
            else:
                report_lines.append("- Quality difference between cold and warm users is minimal")

            time_diff = comp_analysis['differences']['response_time']
            if time_diff > 1.0:
                report_lines.append("- Warm user prompts take notably longer to process")
            elif time_diff < -1.0:
                report_lines.append("- Cold user prompts are processed more slowly")
            else:
                report_lines.append("- Processing time is similar for both user types")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"Detailed report saved to: {report_file}")

        # Print summary to console
        print("\nSTUDY SUMMARY:")
        if 'error' not in comp_analysis:
            differences = comp_analysis['differences']
            print(f"  Quality Difference: {differences['quality']:+.2f} (Warm - Cold)")
            print(f"  Response Time Difference: {differences['response_time']:+.2f}s")
            print(f"  Cold Users Tested: {comp_analysis['sample_sizes']['cold']}")
            print(f"  Warm Users Tested: {comp_analysis['sample_sizes']['warm']}")


def main():
    """Main function to run cold/warm scenarios."""
    parser = argparse.ArgumentParser(description="Run Cold/Warm User Scenarios")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory containing data splits")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument("--mode", choices=["scenarios", "comparative", "single"], default="comparative",
                        help="Running mode")
    parser.add_argument("--scenario", help="Specific scenario to run (for single mode)")
    parser.add_argument("--max-users", type=int, default=10, help="Maximum users per scenario")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    # Initialize the main tester
    tester = PromptGeneratorTester(
        splits_dir=args.splits_dir,
        ollama_model=args.model
    )

    if not tester.user_history:
        print("Cannot proceed without user history")
        return

    # Initialize scenario runner
    scenario_runner = ColdWarmScenarioRunner(tester)

    if args.mode == "comparative":
        # Run full comparative study
        print("Running comprehensive comparative study...")
        study_results = scenario_runner.run_comparative_study(args.max_users, args.temperature)

    elif args.mode == "scenarios":
        # Run all scenarios individually
        print("Running individual scenarios...")

        cold_scenarios = scenario_runner.create_cold_start_scenarios()
        warm_scenarios = scenario_runner.create_warm_user_scenarios()

        all_scenarios = cold_scenarios + warm_scenarios
        for scenario in all_scenarios:
            result = scenario_runner.run_scenario(scenario, args.max_users, args.temperature)
            if "error" not in result:
                print(f"Completed {scenario['name']}: {result['success_rate']:.1%} success rate")
            else:
                print(f"Failed {scenario['name']}: {result['error']}")

    elif args.mode == "single":
        # Run a single scenario
        if not args.scenario:
            print("Please specify --scenario for single mode")
            return

        # Find the scenario
        all_scenarios = (scenario_runner.create_cold_start_scenarios() +
                         scenario_runner.create_warm_user_scenarios())

        target_scenario = None
        for scenario in all_scenarios:
            if scenario['name'].lower() == args.scenario.lower():
                target_scenario = scenario
                break

        if not target_scenario:
            print(f"Scenario '{args.scenario}' not found")
            print("Available scenarios:")
            for scenario in all_scenarios:
                print(f"  - {scenario['name']}")
            return

        result = scenario_runner.run_scenario(target_scenario, args.max_users, args.temperature)
        if "error" not in result:
            print(f"Scenario completed with {result['success_rate']:.1%} success rate")
        else:
            print(f"Scenario failed: {result['error']}")

    print("\nCold/Warm scenario testing completed!")


if __name__ == "__main__":
    main()