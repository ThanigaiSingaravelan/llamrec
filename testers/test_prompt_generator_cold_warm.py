#!/usr/bin/env python3
"""
Test Script for Prompt Generator with Cold/Warm User Analysis
Tests prompt generation and evaluates recommendations using Ollama Llama 3:8b
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.prompt_generator import (
    load_data, get_item_descriptions, get_top_items,
    get_output_items, generate_prompts, TEMPLATE
)

class PromptGeneratorTester:
    """
    Comprehensive tester for prompt generator focusing on cold/warm user scenarios.
    """

    def __init__(self,
                 splits_dir: str = "data/splits",
                 ollama_model: str = "llama3:8b",
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the tester.

        Args:
            splits_dir: Directory containing data splits
            ollama_model: Ollama model to use
            ollama_url: Ollama API URL
        """
        self.splits_dir = splits_dir
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        # Load data
        self.user_history = self.load_user_history()
        self.cold_warm_users = self.load_cold_warm_users()
        self.cross_domain_data = self.load_cross_domain_data()

        # Results storage
        self.test_results = []

        print("🧪 Prompt Generator Tester Initialized")
        print("=" * 50)
        self.test_ollama_connection()
        self.print_dataset_info()

    def test_ollama_connection(self) -> bool:
        """Test Ollama connection and model availability."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                if self.ollama_model in models:
                    print(f"✅ Connected to Ollama. Model {self.ollama_model} ready!")
                    return True
                else:
                    print(f"❌ Model {self.ollama_model} not found. Available: {models}")
                    return False
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure Ollama is running with: ollama serve")
            return False

    def load_user_history(self) -> Dict:
        """Load user history JSON."""
        history_path = os.path.join(self.splits_dir, "user_history.json")

        if not os.path.exists(history_path):
            print(f"❌ User history not found: {history_path}")
            return {}

        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"✅ Loaded user history: {len(history)} users")
            return history
        except Exception as e:
            print(f"❌ Error loading user history: {e}")
            return {}

    def load_cold_warm_users(self) -> Dict:
        """Load cold and warm user classifications."""
        cold_warm_path = os.path.join(self.splits_dir, "cold_warm_users.json")

        if not os.path.exists(cold_warm_path):
            print(f"❌ Cold/warm users not found: {cold_warm_path}")
            return {"cold_users": [], "warm_users": [], "user_interaction_counts": {}}

        try:
            with open(cold_warm_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(
                f"✅ Loaded cold/warm users: {len(data.get('cold_users', []))} cold, {len(data.get('warm_users', []))} warm")
            return data
        except Exception as e:
            print(f"❌ Error loading cold/warm users: {e}")
            return {"cold_users": [], "warm_users": [], "user_interaction_counts": {}}

    def load_cross_domain_data(self) -> Dict:
        """Load cross-domain split data."""
        cross_domain_dir = os.path.join(self.splits_dir, "cross_domain")

        if not os.path.exists(cross_domain_dir):
            print(f"❌ Cross-domain directory not found: {cross_domain_dir}")
            return {}

        data = {}
        for pair_dir in os.listdir(cross_domain_dir):
            if "_to_" not in pair_dir:
                continue

            pair_path = os.path.join(cross_domain_dir, pair_dir)
            if not os.path.isdir(pair_path):
                continue

            try:
                # Load source and target data
                source_path = os.path.join(pair_path, "source_train.csv")
                target_path = os.path.join(pair_path, "target_test.csv")

                if os.path.exists(source_path) and os.path.exists(target_path):
                    source_df = pd.read_csv(source_path)
                    target_df = pd.read_csv(target_path)

                    data[pair_dir] = {
                        'source_df': source_df,
                        'target_df': target_df,
                        'source_domain': pair_dir.split("_to_")[0],
                        'target_domain': pair_dir.split("_to_")[1]
                    }

            except Exception as e:
                print(f"⚠️ Error loading {pair_dir}: {e}")
                continue

        print(f"✅ Loaded cross-domain data: {len(data)} domain pairs")
        return data

    def print_dataset_info(self):
        """Print comprehensive dataset information."""
        print("\n📊 Dataset Information")
        print("-" * 30)

        # User statistics
        cold_users = set(self.cold_warm_users.get('cold_users', []))
        warm_users = set(self.cold_warm_users.get('warm_users', []))
        all_users = set(self.user_history.keys())

        print(f"Total Users: {len(all_users)}")
        print(f"Cold Users: {len(cold_users)} ({len(cold_users) / len(all_users) * 100:.1f}%)")
        print(f"Warm Users: {len(warm_users)} ({len(warm_users) / len(all_users) * 100:.1f}%)")
        print(f"Regular Users: {len(all_users - cold_users - warm_users)}")

        # Interaction statistics
        user_counts = self.cold_warm_users.get('user_interaction_counts', {})
        if user_counts:
            interactions = list(user_counts.values())
            print(f"\nInteraction Statistics:")
            print(f"  Min interactions: {min(interactions)}")
            print(f"  Max interactions: {max(interactions)}")
            print(f"  Avg interactions: {sum(interactions) / len(interactions):.1f}")

        # Domain pair information
        print(f"\nCross-Domain Pairs: {len(self.cross_domain_data)}")
        for pair_name, pair_data in list(self.cross_domain_data.items())[:5]:
            source_users = len(pair_data['source_df']['reviewerID'].unique())
            target_users = len(pair_data['target_df']['reviewerID'].unique())
            print(f"  {pair_name}: {source_users} source users, {target_users} target users")

        if len(self.cross_domain_data) > 5:
            print(f"  ... and {len(self.cross_domain_data) - 5} more pairs")

    def generate_test_prompts(self,
                              user_type: str = "cold",
                              max_prompts: int = 50,
                              domain_pair: Optional[str] = None) -> List[Dict]:
        """
        Generate test prompts for specific user types.

        Args:
            user_type: "cold", "warm", or "random"
            max_prompts: Maximum number of prompts to generate
            domain_pair: Specific domain pair (e.g., "Books_to_Movies_and_TV")

        Returns:
            List of generated prompts
        """
        print(f"\n🔄 Generating {user_type} user prompts...")

        # Select users based on type
        if user_type == "cold":
            target_users = set(self.cold_warm_users.get('cold_users', []))
        elif user_type == "warm":
            target_users = set(self.cold_warm_users.get('warm_users', []))
        else:  # random
            target_users = set(self.user_history.keys())

        if not target_users:
            print(f"❌ No {user_type} users found")
            return []

        # Select domain pairs to test
        pairs_to_test = []
        if domain_pair and domain_pair in self.cross_domain_data:
            pairs_to_test = [domain_pair]
        else:
            pairs_to_test = list(self.cross_domain_data.keys())

        all_prompts = []

        for pair_name in pairs_to_test:
            pair_data = self.cross_domain_data[pair_name]
            source_df = pair_data['source_df']
            target_df = pair_data['target_df']
            source_domain = pair_data['source_domain']
            target_domain = pair_data['target_domain']

            # Filter to target user type
            source_users = set(source_df['reviewerID'].unique())
            target_users_in_pair = target_users.intersection(source_users)

            if not target_users_in_pair:
                continue

            # Limit users for this pair
            users_to_test = list(target_users_in_pair)[:max_prompts // len(pairs_to_test) + 1]

            print(f"  📝 {pair_name}: Testing {len(users_to_test)} {user_type} users")

            # Generate item descriptions
            source_map = get_item_descriptions(source_df)
            target_map = get_item_descriptions(target_df)

            # Generate prompts for these users
            pair_prompts = generate_prompts(
                source_df, target_df,
                source_map, target_map,
                source_domain, target_domain,
                max_prompts=len(users_to_test)
            )

            # Filter to our target users and add metadata
            for prompt in pair_prompts:
                if prompt['user_id'] in target_users_in_pair:
                    prompt['user_type'] = user_type
                    prompt['pair_name'] = pair_name

                    # Add interaction count
                    user_counts = self.cold_warm_users.get('user_interaction_counts', {})
                    prompt['user_interaction_count'] = user_counts.get(prompt['user_id'], 0)

                    all_prompts.append(prompt)

                    if len(all_prompts) >= max_prompts:
                        break

            if len(all_prompts) >= max_prompts:
                break

        print(f"✅ Generated {len(all_prompts)} {user_type} user prompts")
        return all_prompts

    def call_ollama(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> Dict:
        """
        Call Ollama API with error handling and timing.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with response and metadata
        """
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.1
                }
            }

            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()["response"].strip()
                return {
                    "success": True,
                    "response": result,
                    "response_time": end_time - start_time,
                    "tokens_estimated": len(result.split()),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "response_time": end_time - start_time,
                    "tokens_estimated": 0,
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "tokens_estimated": 0,
                "error": str(e)
            }

    def test_prompts_with_ollama(self,
                                 prompts: List[Dict],
                                 temperature: float = 0.7,
                                 save_intermediate: bool = True) -> List[Dict]:
        """
        Test generated prompts with Ollama and collect results.

        Args:
            prompts: List of prompt dictionaries
            temperature: Sampling temperature
            save_intermediate: Whether to save results incrementally

        Returns:
            List of test results
        """
        print(f"\n🚀 Testing {len(prompts)} prompts with Ollama...")

        results = []
        failed_count = 0

        for i, prompt_data in enumerate(tqdm(prompts, desc="Testing prompts")):
            # Call Ollama
            ollama_result = self.call_ollama(
                prompt_data['input'],
                max_tokens=600,
                temperature=temperature
            )

            # Compile result
            result = {
                "prompt_id": i,
                "user_id": prompt_data['user_id'],
                "user_type": prompt_data['user_type'],
                "user_interaction_count": prompt_data['user_interaction_count'],
                "source_domain": prompt_data['source_domain'],
                "target_domain": prompt_data['target_domain'],
                "pair_name": prompt_data['pair_name'],
                "input_prompt": prompt_data['input'],
                "expected_output": prompt_data['output'],
                "ollama_response": ollama_result['response'],
                "success": ollama_result['success'],
                "response_time": ollama_result['response_time'],
                "tokens_estimated": ollama_result['tokens_estimated'],
                "error": ollama_result['error'],
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

            if not ollama_result['success']:
                failed_count += 1
                print(f"❌ Failed prompt {i + 1}: {ollama_result['error']}")

            # Save intermediate results every 10 prompts
            if save_intermediate and (i + 1) % 10 == 0:
                self.save_results(results, f"intermediate_results_{i + 1}.json")

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.1)

        print(f"✅ Completed testing: {len(results) - failed_count}/{len(results)} successful")
        if failed_count > 0:
            print(f"⚠️ {failed_count} prompts failed")

        return results

    def analyze_results_by_user_type(self, results: List[Dict]) -> Dict:
        """
        Analyze results by user type (cold vs warm).

        Args:
            results: List of test results

        Returns:
            Analysis dictionary
        """
        print("\n📊 Analyzing Results by User Type...")

        # Group results by user type
        by_user_type = defaultdict(list)
        for result in results:
            if result['success']:
                by_user_type[result['user_type']].append(result)

        analysis = {}

        for user_type, type_results in by_user_type.items():
            if not type_results:
                continue

            # Basic statistics
            response_times = [r['response_time'] for r in type_results]
            token_counts = [r['tokens_estimated'] for r in type_results]
            interaction_counts = [r['user_interaction_count'] for r in type_results]

            # Response quality analysis (simple heuristics)
            quality_scores = []
            for result in type_results:
                response = result['ollama_response']
                score = self.estimate_response_quality(response)
                quality_scores.append(score)

            analysis[user_type] = {
                "count": len(type_results),
                "avg_response_time": sum(response_times) / len(response_times),
                "avg_tokens": sum(token_counts) / len(token_counts),
                "avg_user_interactions": sum(interaction_counts) / len(interaction_counts),
                "avg_quality_score": sum(quality_scores) / len(quality_scores),
                "response_times": response_times,
                "quality_scores": quality_scores,
                "domain_pairs": list(set(r['pair_name'] for r in type_results))
            }

        # Print analysis
        for user_type, stats in analysis.items():
            print(f"\n🔍 {user_type.upper()} USERS ({stats['count']} users):")
            print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
            print(f"  Avg Tokens Generated: {stats['avg_tokens']:.1f}")
            print(f"  Avg User Interactions: {stats['avg_user_interactions']:.1f}")
            print(f"  Avg Quality Score: {stats['avg_quality_score']:.2f}/5.0")
            print(f"  Domain Pairs Tested: {len(stats['domain_pairs'])}")

        return analysis

    def estimate_response_quality(self, response: str) -> float:
        """
        Estimate response quality using simple heuristics.

        Args:
            response: Generated response text

        Returns:
            Quality score from 0.0 to 5.0
        """
        if not response or len(response.strip()) < 20:
            return 0.0

        score = 1.0  # Base score

        # Check for structured format (numbered list)
        if any(f"{i}." in response for i in range(1, 6)):
            score += 1.0

        # Check for explanations/reasoning
        explanation_words = ["because", "since", "due to", "matches", "similar", "appeals", "like"]
        if any(word in response.lower() for word in explanation_words):
            score += 1.0

        # Check for item titles (quoted or bolded text)
        if '"' in response or "**" in response:
            score += 0.5

        # Check for domain-specific terms
        domains = ["book", "movie", "music", "album", "film", "novel", "song", "cd"]
        if any(domain in response.lower() for domain in domains):
            score += 0.5

        # Penalize very short or very long responses
        word_count = len(response.split())
        if word_count < 30:
            score -= 0.5
        elif word_count > 200:
            score -= 0.3

        return min(5.0, max(0.0, score))

    def compare_cold_vs_warm_performance(self, analysis: Dict) -> Dict:
        """
        Compare cold vs warm user performance.

        Args:
            analysis: Analysis results from analyze_results_by_user_type

        Returns:
            Comparison results
        """
        if 'cold' not in analysis or 'warm' not in analysis:
            print("⚠️ Missing cold or warm user data for comparison")
            return {}

        cold_stats = analysis['cold']
        warm_stats = analysis['warm']

        comparison = {
            "response_time_difference": warm_stats['avg_response_time'] - cold_stats['avg_response_time'],
            "token_difference": warm_stats['avg_tokens'] - cold_stats['avg_tokens'],
            "quality_difference": warm_stats['avg_quality_score'] - cold_stats['avg_quality_score'],
            "cold_count": cold_stats['count'],
            "warm_count": warm_stats['count']
        }

        print("\n🆚 COLD vs WARM USER COMPARISON:")
        print("-" * 40)
        print(f"Response Time: Warm users take {comparison['response_time_difference']:+.2f}s more")
        print(f"Token Generation: Warm users get {comparison['token_difference']:+.1f} more tokens")
        print(f"Quality Score: Warm users score {comparison['quality_difference']:+.2f} points higher")
        print(f"Sample Sizes: {comparison['cold_count']} cold, {comparison['warm_count']} warm")

        # Statistical significance (simple)
        from scipy import stats
        if len(cold_stats['quality_scores']) > 5 and len(warm_stats['quality_scores']) > 5:
            t_stat, p_value = stats.ttest_ind(cold_stats['quality_scores'], warm_stats['quality_scores'])
            print(
                f"Quality Difference P-value: {p_value:.4f} {'(Significant)' if p_value < 0.05 else '(Not significant)'}")
            comparison['quality_p_value'] = p_value

        return comparison

    def save_results(self, results: List[Dict], filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"prompt_generator_test_results_{timestamp}.json"

        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)

        output_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "model": self.ollama_model,
                "total_prompts": len(results),
                "successful_prompts": sum(1 for r in results if r['success'])
            },
            "results": results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {filepath}")
        return filepath

    def generate_summary_report(self, results: List[Dict], analysis: Dict, comparison: Dict):
        """Generate a comprehensive summary report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"results/prompt_test_report_{timestamp}.txt"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PROMPT GENERATOR TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.ollama_model}")
        report_lines.append(f"Total Prompts Tested: {len(results)}")
        report_lines.append(f"Successful Responses: {sum(1 for r in results if r['success'])}")
        report_lines.append("")

        # User type analysis
        report_lines.append("USER TYPE ANALYSIS")
        report_lines.append("-" * 40)
        for user_type, stats in analysis.items():
            report_lines.append(f"\n{user_type.upper()} USERS:")
            report_lines.append(f"  Count: {stats['count']}")
            report_lines.append(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
            report_lines.append(f"  Avg Tokens: {stats['avg_tokens']:.1f}")
            report_lines.append(f"  Avg Quality Score: {stats['avg_quality_score']:.2f}/5.0")
            report_lines.append(f"  Domain Pairs: {len(stats['domain_pairs'])}")

        # Comparison results
        if comparison:
            report_lines.append(f"\nCOLD vs WARM COMPARISON")
            report_lines.append("-" * 40)
            report_lines.append(f"Response Time Difference: {comparison['response_time_difference']:+.2f}s")
            report_lines.append(f"Quality Score Difference: {comparison['quality_difference']:+.2f}")
            if 'quality_p_value' in comparison:
                significance = "Significant" if comparison['quality_p_value'] < 0.05 else "Not significant"
                report_lines.append(f"Statistical Significance: {significance} (p={comparison['quality_p_value']:.4f})")

        # Domain pair performance
        domain_performance = defaultdict(list)
        for result in results:
            if result['success']:
                quality = self.estimate_response_quality(result['ollama_response'])
                domain_performance[result['pair_name']].append(quality)

        report_lines.append(f"\nDOMAIN PAIR PERFORMANCE")
        report_lines.append("-" * 40)
        for pair, qualities in sorted(domain_performance.items()):
            avg_quality = sum(qualities) / len(qualities)
            report_lines.append(f"{pair}: {avg_quality:.2f} avg quality ({len(qualities)} tests)")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"📋 Summary report saved to: {report_file}")
        return report_file


def main():
    """Main function to run the prompt generator tests."""
    parser = argparse.ArgumentParser(description="Test Prompt Generator with Cold/Warm Users")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory containing data splits")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model to use")
    parser.add_argument("--user-type", choices=["cold", "warm", "both", "random"], default="both",
                        help="Type of users to test")
    parser.add_argument("--max-prompts", type=int, default=30, help="Maximum prompts per user type")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--domain-pair", help="Specific domain pair to test (e.g., Books_to_Movies_and_TV)")

    args = parser.parse_args()

    # Initialize tester
    tester = PromptGeneratorTester(
        splits_dir=args.splits_dir,
        ollama_model=args.model
    )

    if not tester.user_history:
        print("❌ Cannot proceed without user history")
        return

    # Generate and test prompts
    all_results = []

    if args.user_type in ["cold", "both"]:
        print("\n🥶 Testing COLD users...")
        cold_prompts = tester.generate_test_prompts("cold", args.max_prompts, args.domain_pair)
        if cold_prompts:
            cold_results = tester.test_prompts_with_ollama(cold_prompts, args.temperature)
            all_results.extend(cold_results)

    if args.user_type in ["warm", "both"]:
        print("\n🔥 Testing WARM users...")
        warm_prompts = tester.generate_test_prompts("warm", args.max_prompts, args.domain_pair)
        if warm_prompts:
            warm_results = tester.test_prompts_with_ollama(warm_prompts, args.temperature)
            all_results.extend(warm_results)

    if args.user_type == "random":
        print("\n🎲 Testing RANDOM users...")
        random_prompts = tester.generate_test_prompts("random", args.max_prompts, args.domain_pair)
        if random_prompts:
            random_results = tester.test_prompts_with_ollama(random_prompts, args.temperature)
            all_results.extend(random_results)

    if not all_results:
        print("❌ No results to analyze")
        return

    # Analyze results
    analysis = tester.analyze_results_by_user_type(all_results)

    # Compare cold vs warm if both were tested
    comparison = {}
    if args.user_type == "both":
        comparison = tester.compare_cold_vs_warm_performance(analysis)

    # Save results and generate report
    tester.save_results(all_results)
    tester.generate_summary_report(all_results, analysis, comparison)

    print("\n✅ Testing completed successfully!")
    print(f"📊 Total tests: {len(all_results)}")
    print(f"🎯 Success rate: {sum(1 for r in all_results if r['success']) / len(all_results) * 100:.1f}%")


if __name__ == "__main__":
    main()