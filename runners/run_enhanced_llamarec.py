#!/usr/bin/env python3
"""
LLAMAREC with Ollama Llama3 - Fixed Version
Enhanced Cross-Domain Recommendations without LoRA dependencies
"""

import json
import os
import requests
import random
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedLlamaRecEngine:
    """
    Enhanced LLAMAREC engine using available Llama3 models.
    Works with existing Ollama models without requiring LoRA adapters.
    """

    def __init__(self,
                 user_history_path: str = "data/splits/user_history.json",
                 base_model: str = "llama3:70b-instruct",
                 additional_models: List[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialize the enhanced LLAMAREC engine.
        """
        self.user_history_path = user_history_path
        self.base_model = base_model
        self.additional_models = additional_models or []
        self.ollama_url = ollama_url

        # Load user data
        self.user_history = self.load_user_history()
        self.cold_warm_users = self.load_cold_warm_users()

        # Model configurations for different available models
        self.model_configs = self._initialize_model_configs()

        # Results storage
        self.evaluation_results = {}
        self.comparative_results = {}

        print("Enhanced LLAMAREC Engine Initialized")
        print("=" * 60)
        self._print_initialization_info()
        self._detect_available_models()

    def _initialize_model_configs(self) -> Dict[str, Dict]:
        """Initialize configurations for different available models."""
        configs = {
            "base_model": {
                "model_name": self.base_model,
                "description": f"Base {self.base_model} model",
                "temperature": 0.7,
                "max_tokens": 512,
                "template_type": "standard"
            }
        }

        # Add any additional models if specified
        for i, model in enumerate(self.additional_models, 1):
            configs[f"model_{i}"] = {
                "model_name": model,
                "description": f"Alternative model: {model}",
                "temperature": 0.7,
                "max_tokens": 512,
                "template_type": "standard"
            }

        return configs

    def _print_initialization_info(self):
        """Print initialization information."""
        print(f"User History: {len(self.user_history)} users loaded")
        print(f"Base Model: {self.base_model}")

        if self.additional_models:
            print(f"Additional Models: {len(self.additional_models)}")
            for model in self.additional_models:
                print(f"   • {model}")

        if self.cold_warm_users:
            cold_count = len(self.cold_warm_users.get('cold_users', []))
            warm_count = len(self.cold_warm_users.get('warm_users', []))
            print(f"User Types: {cold_count} cold, {warm_count} warm")

    def _detect_available_models(self) -> bool:
        """Detect and verify available models in Ollama."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]

                print(f"Connected to Ollama at {self.ollama_url}")
                print(f"Available models: {len(model_names)}")

                # Check base model availability
                if self.base_model in model_names:
                    print(f"Base model {self.base_model} available")
                else:
                    print(f"WARNING: Base model {self.base_model} not found")
                    print(f"Available models: {model_names[:5]}...")

                    # Try to use the first available llama model
                    llama_models = [m for m in model_names if 'llama' in m.lower()]
                    if llama_models:
                        self.base_model = llama_models[0]
                        self.model_configs["base_model"]["model_name"] = self.base_model
                        print(f"Using {self.base_model} instead")
                    else:
                        print("No Llama models found!")
                        return False

                # Filter additional models to only include available ones
                available_additional = [m for m in self.additional_models if m in model_names]
                if available_additional != self.additional_models:
                    print(f"Found {len(available_additional)} of {len(self.additional_models)} additional models")
                    self.additional_models = available_additional
                    # Update model configs
                    self.model_configs = self._initialize_model_configs()

                return True
            else:
                print(f"Ollama API returned status code: {response.status_code}")
                return False

        except Exception as e:
            print(f"Cannot connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")
            return False

    def load_user_history(self) -> Dict:
        """Load user history with error handling."""
        if not os.path.exists(self.user_history_path):
            logger.error(f"User history not found: {self.user_history_path}")
            return {}

        try:
            with open(self.user_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            logger.info(f"Loaded user history: {len(history)} users")
            return history
        except Exception as e:
            logger.error(f"Error loading user history: {e}")
            return {}

    def load_cold_warm_users(self) -> Dict:
        """Load cold/warm user classifications."""
        splits_dir = os.path.dirname(self.user_history_path)
        cold_warm_path = os.path.join(splits_dir, "cold_warm_users.json")

        if not os.path.exists(cold_warm_path):
            logger.warning(f"Cold/warm users not found: {cold_warm_path}")
            return {"cold_users": [], "warm_users": [], "user_interaction_counts": {}}

        try:
            with open(cold_warm_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded cold/warm classifications")
            return data
        except Exception as e:
            logger.error(f"Error loading cold/warm users: {e}")
            return {"cold_users": [], "warm_users": [], "user_interaction_counts": {}}

    def get_prompt_template(self, template_type: str = "standard") -> str:
        """Get prompt template for recommendations."""
        templates = {
            "standard": """Based on the user's preferences in {source_domain}, generate personalized recommendations for items from {target_domain}.

User's highly-rated items from {source_domain}: {user_items}
Target domain: {target_domain}

Task: Recommend the top 3 {target_domain} items the user is most likely to enjoy, based on their preferences above. Provide a brief explanation for each recommendation that connects it to the user's interests.

Output Format:
1. Item Title – Explanation
2. Item Title – Explanation  
3. Item Title – Explanation

Recommendations:""",

            "detailed": """You are an expert cross-domain recommendation system. Analyze the user's preferences step-by-step.

Step 1: Analyze patterns in user's {source_domain} preferences: {user_items}
Step 2: Identify transferable preferences to {target_domain}
Step 3: Recommend 3 {target_domain} items that match these patterns

Provide detailed reasoning for each recommendation.

Recommendations:""",

            "few_shot": """Here are examples of cross-domain recommendations:

Example: User likes classic literature → Recommend classic films
User likes: "The Great Gatsby", "1984", "To Kill a Mockingbird" 
Recommendations:
1. "The Shawshank Redemption" – Classic narrative themes
2. "Casablanca" – Timeless storytelling
3. "12 Angry Men" – Social commentary

Now generate recommendations:
User's {source_domain} preferences: {user_items}
Target domain: {target_domain}

Recommendations:"""
        }

        return templates.get(template_type, templates["standard"])

    def call_ollama_with_config(self, prompt: str, model_config: Dict) -> Dict:
        """Call Ollama with specific model configuration."""
        try:
            payload = {
                "model": model_config["model_name"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model_config.get("temperature", 0.7),
                    "top_p": 0.9,
                    "num_predict": model_config.get("max_tokens", 512),
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
                    "model_used": model_config["model_name"],
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "response_time": end_time - start_time,
                    "tokens_estimated": 0,
                    "model_used": model_config["model_name"],
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "tokens_estimated": 0,
                "model_used": model_config.get("model_name", "unknown"),
                "error": str(e)
            }

    def format_user_items(self, items: List, max_items: int = 5) -> str:
        """Format user items for prompts."""
        if not items:
            return "[]"

        items = items[:max_items]
        formatted = []

        for item in items:
            if isinstance(item, dict):
                title = item.get('title', item.get('asin', str(item)))
            else:
                title = str(item)

            # Clean and truncate title
            title = title.replace('"', "'").strip()
            if len(title) > 80:
                title = title[:77] + "..."
            formatted.append(f'"{title}"')

        return f"[{', '.join(formatted)}]"

    def generate_cross_domain_recommendations(self,
                                              user_id: str,
                                              source_domain: str,
                                              target_domain: str,
                                              model_config: Dict,
                                              template_type: str = "standard",
                                              num_recs: int = 3) -> Dict:
        """Generate cross-domain recommendations using specified model configuration."""

        if user_id not in self.user_history:
            return {"error": f"User {user_id} not found", "model_used": model_config["model_name"]}

        user_data = self.user_history[user_id]

        if source_domain not in user_data or target_domain not in user_data:
            return {"error": f"Domains not available for user {user_id}", "model_used": model_config["model_name"]}

        source_items = user_data[source_domain].get('liked', [])
        if not source_items:
            return {"error": f"No liked items in {source_domain} for user {user_id}",
                    "model_used": model_config["model_name"]}

        # Format prompt based on template type
        template = self.get_prompt_template(template_type)
        formatted_items = self.format_user_items(source_items)

        prompt = template.format(
            source_domain=source_domain,
            target_domain=target_domain,
            user_items=formatted_items
        )

        # Generate recommendation
        result = self.call_ollama_with_config(prompt, model_config)

        # Prepare response
        response = {
            "user_id": user_id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "model_config": model_config["model_name"],
            "template_type": template_type,
            "source_items": formatted_items,
            "recommendations": result.get("response", ""),
            "success": result.get("success", False),
            "response_time": result.get("response_time", 0),
            "tokens_estimated": result.get("tokens_estimated", 0),
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }

        return response

    def run_simple_comparison_study(self,
                                    max_users: int = 30,
                                    domains_to_test: List[str] = None,
                                    template_types: List[str] = None) -> Dict:
        """Run simplified comparison study with available models."""

        print(f"\nModel Comparison Study")
        print("=" * 50)

        # Default configurations
        if domains_to_test is None:
            domains_to_test = [
                ("Books", "Movies_and_TV"),
                ("Movies_and_TV", "Books")
            ]

        if template_types is None:
            template_types = ["standard", "detailed"]

        # Get sample users
        all_users = list(self.user_history.keys())
        sample_users = all_users[:max_users]

        print(f"Testing {len(sample_users)} users")
        print(f"Domain pairs: {len(domains_to_test)}")
        print(f"Model configs: {len(self.model_configs)}")
        print(f"Template types: {len(template_types)}")

        study_results = {
            "study_info": {
                "timestamp": datetime.now().isoformat(),
                "max_users": max_users,
                "domains_tested": domains_to_test,
                "template_types": template_types,
                "model_configs": list(self.model_configs.keys())
            },
            "results": [],
            "performance_summary": {}
        }

        total_tests = len(sample_users) * len(domains_to_test) * len(self.model_configs) * len(template_types)
        completed_tests = 0

        print(f"Starting {total_tests} total tests...")

        with tqdm(total=total_tests, desc="Running comparison") as pbar:
            for user_id in sample_users:
                user_data = self.user_history[user_id]
                available_domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]

                if len(available_domains) < 2:
                    pbar.update(len(domains_to_test) * len(self.model_configs) * len(template_types))
                    continue

                for source_domain, target_domain in domains_to_test:
                    if source_domain not in available_domains or target_domain not in available_domains:
                        pbar.update(len(self.model_configs) * len(template_types))
                        continue

                    for config_name, model_config in self.model_configs.items():
                        for template_type in template_types:
                            try:
                                result = self.generate_cross_domain_recommendations(
                                    user_id, source_domain, target_domain,
                                    model_config, template_type
                                )

                                # Add study metadata
                                result["config_name"] = config_name
                                result[
                                    "study_id"] = f"{user_id}_{source_domain}_{target_domain}_{config_name}_{template_type}"

                                # Calculate quality metrics if successful
                                if result["success"]:
                                    result["quality_score"] = self._estimate_response_quality(result["recommendations"])
                                    result["user_type"] = self._classify_user_type(user_id)

                                study_results["results"].append(result)
                                completed_tests += 1

                            except Exception as e:
                                logger.error(f"Error in test {completed_tests}: {e}")

                            pbar.update(1)
                            time.sleep(0.1)  # Rate limiting

        # Generate performance summary
        study_results["performance_summary"] = self._analyze_performance(study_results["results"])

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/model_comparison_study_{timestamp}.json"

        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(study_results, f, indent=2, ensure_ascii=False)

        print(f"\nStudy results saved to: {results_file}")
        print(f"Completed {completed_tests} tests successfully")

        # Generate summary report
        self._generate_comparison_report(study_results, f"results/model_comparison_report_{timestamp}.txt")

        return study_results

    def _analyze_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance across different configurations."""
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {"error": "No successful results to analyze"}

        # Group by different factors
        by_config = defaultdict(list)
        by_template = defaultdict(list)
        by_domain_pair = defaultdict(list)
        by_user_type = defaultdict(list)

        for result in successful_results:
            config_name = result.get("config_name", "unknown")
            template_type = result.get("template_type", "unknown")
            domain_pair = f"{result.get('source_domain', '')}→{result.get('target_domain', '')}"
            user_type = result.get("user_type", "unknown")

            quality = result.get("quality_score", 0)
            response_time = result.get("response_time", 0)

            by_config[config_name].append({"quality": quality, "time": response_time})
            by_template[template_type].append({"quality": quality, "time": response_time})
            by_domain_pair[domain_pair].append({"quality": quality, "time": response_time})
            by_user_type[user_type].append({"quality": quality, "time": response_time})

        def calculate_metrics(data_list):
            qualities = [d["quality"] for d in data_list]
            times = [d["time"] for d in data_list]
            return {
                "count": len(data_list),
                "avg_quality": float(np.mean(qualities)) if qualities else 0,
                "std_quality": float(np.std(qualities)) if qualities else 0,
                "avg_response_time": float(np.mean(times)) if times else 0,
                "std_response_time": float(np.std(times)) if times else 0
            }

        performance_summary = {
            "overall": {
                "total_successful": len(successful_results),
                "avg_quality": float(np.mean([r.get("quality_score", 0) for r in successful_results])),
                "avg_response_time": float(np.mean([r.get("response_time", 0) for r in successful_results]))
            },
            "by_model_config": {config: calculate_metrics(data) for config, data in by_config.items()},
            "by_template": {template: calculate_metrics(data) for template, data in by_template.items()},
            "by_domain_pair": {pair: calculate_metrics(data) for pair, data in by_domain_pair.items()},
            "by_user_type": {user_type: calculate_metrics(data) for user_type, data in by_user_type.items()}
        }

        # Identify best performing configurations
        if performance_summary["by_model_config"]:
            best_config = max(performance_summary["by_model_config"].items(),
                              key=lambda x: x[1]["avg_quality"])
            performance_summary["best_config"] = {
                "name": best_config[0],
                "quality": best_config[1]["avg_quality"]
            }

        if performance_summary["by_template"]:
            best_template = max(performance_summary["by_template"].items(),
                                key=lambda x: x[1]["avg_quality"])
            performance_summary["best_template"] = {
                "name": best_template[0],
                "quality": best_template[1]["avg_quality"]
            }

        return performance_summary

    def _estimate_response_quality(self, response: str) -> float:
        """Estimate response quality using heuristics."""
        if not response or len(response.strip()) < 20:
            return 0.0

        score = 1.0  # Base score

        # Check for structured format
        if any(f"{i}." in response for i in range(1, 6)):
            score += 1.0

        # Check for explanations
        explanation_words = ["because", "since", "due to", "matches", "similar", "appeals", "like", "based on"]
        if any(word in response.lower() for word in explanation_words):
            score += 1.0

        # Check for item titles
        if '"' in response or "**" in response:
            score += 0.5

        # Check for domain mentions
        domains = ["book", "movie", "music", "album", "film", "novel", "song", "cd"]
        if any(domain in response.lower() for domain in domains):
            score += 0.5

        # Length penalty/bonus
        word_count = len(response.split())
        if 30 <= word_count <= 150:
            score += 0.5
        elif word_count < 20 or word_count > 200:
            score -= 0.5

        return min(5.0, max(0.0, score))

    def _classify_user_type(self, user_id: str) -> str:
        """Classify user as cold, warm, or regular."""
        cold_users = set(self.cold_warm_users.get('cold_users', []))
        warm_users = set(self.cold_warm_users.get('warm_users', []))

        if user_id in cold_users:
            return "cold"
        elif user_id in warm_users:
            return "warm"
        else:
            return "regular"

    def _generate_comparison_report(self, study_results: Dict, output_file: str):
        """Generate comprehensive comparison report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ENHANCED LLAMAREC MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)

        info = study_results["study_info"]
        summary = study_results["performance_summary"]

        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Study Timestamp: {info['timestamp']}")
        report_lines.append(f"Users Tested: {info['max_users']}")
        report_lines.append(f"Domain Pairs: {len(info['domains_tested'])}")
        report_lines.append(f"Model Configurations: {len(info['model_configs'])}")
        report_lines.append(f"Template Types: {len(info['template_types'])}")
        report_lines.append("")

        # Overall performance
        if "error" not in summary:
            overall = summary["overall"]
            report_lines.append("OVERALL PERFORMANCE")
            report_lines.append("-" * 40)
            report_lines.append(f"Successful Tests: {overall['total_successful']}")
            report_lines.append(f"Average Quality Score: {overall['avg_quality']:.3f}/5.0")
            report_lines.append(f"Average Response Time: {overall['avg_response_time']:.2f}s")
            report_lines.append("")

            # Model configuration comparison
            report_lines.append("MODEL CONFIGURATION PERFORMANCE")
            report_lines.append("-" * 40)

            config_performances = []
            for config, metrics in summary["by_model_config"].items():
                config_performances.append((config, metrics["avg_quality"], metrics["count"]))

            # Sort by quality
            config_performances.sort(key=lambda x: x[1], reverse=True)

            for rank, (config, quality, count) in enumerate(config_performances, 1):
                report_lines.append(f"{rank}. {config}")
                report_lines.append(f"   Quality: {quality:.3f}/5.0 ({count} tests)")
                report_lines.append(f"   Response Time: {summary['by_model_config'][config]['avg_response_time']:.2f}s")
                report_lines.append("")

            # Template comparison
            report_lines.append("TEMPLATE TYPE PERFORMANCE")
            report_lines.append("-" * 40)

            for template, metrics in summary["by_template"].items():
                report_lines.append(f"{template.upper()}:")
                report_lines.append(f"  Quality: {metrics['avg_quality']:.3f}/5.0")
                report_lines.append(f"  Response Time: {metrics['avg_response_time']:.2f}s")
                report_lines.append(f"  Test Count: {metrics['count']}")
                report_lines.append("")

            # Best performers
            if "best_config" in summary and "best_template" in summary:
                report_lines.append("TOP PERFORMERS")
                report_lines.append("-" * 40)
                report_lines.append(
                    f"Best Model Config: {summary['best_config']['name']} ({summary['best_config']['quality']:.3f})")
                report_lines.append(
                    f"Best Template: {summary['best_template']['name']} ({summary['best_template']['quality']:.3f})")
                report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"Detailed report saved to: {output_file}")

    def run_interactive_demo(self):
        """Interactive demo for testing different configurations."""
        print(f"\nInteractive Demo")
        print("=" * 40)

        if not self.user_history:
            print("No user history available")
            return

        # Show available configurations
        print("Available Model Configurations:")
        for i, (config_name, config) in enumerate(self.model_configs.items(), 1):
            print(f"  {i}. {config_name} - {config['description']}")

        # Show sample users
        sample_users = list(self.user_history.keys())[:10]
        print(f"\nSample Users: {', '.join(sample_users)}")

        while True:
            print(f"\n{'=' * 50}")
            print("Interactive Testing Menu")
            print("1. Test specific user with model")
            print("2. Compare all models for one user")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == '1':
                self._interactive_single_test()
            elif choice == '2':
                self._interactive_model_comparison()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def _interactive_single_test(self):
        """Interactive single user test."""
        user_id = input("Enter user ID: ").strip()

        if user_id not in self.user_history:
            print(f"User {user_id} not found")
            return

        # Show available domains
        user_data = self.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]
        print(f"Available domains: {', '.join(domains)}")

        source_domain = input("Source domain: ").strip()
        target_domain = input("Target domain: ").strip()

        if source_domain not in domains or target_domain not in domains:
            print("Invalid domain(s)")
            return

        # Show model configurations
        print("\nModel Configurations:")
        config_names = list(self.model_configs.keys())
        for i, config_name in enumerate(config_names, 1):
            print(f"  {i}. {config_name}")

        try:
            config_choice = int(input("Choose configuration (number): ")) - 1
            if config_choice < 0 or config_choice >= len(config_names):
                print("Invalid choice")
                return

            selected_config_name = config_names[config_choice]
            selected_config = self.model_configs[selected_config_name]

        except ValueError:
            print("Invalid input")
            return

        template_type = input("Template type (standard/detailed/few_shot, default: standard): ").strip()
        if not template_type:
            template_type = "standard"

        print(f"\nGenerating recommendations...")
        print(f"User: {user_id}")
        print(f"Transfer: {source_domain} → {target_domain}")
        print(f"Model: {selected_config_name}")
        print(f"Template: {template_type}")

        result = self.generate_cross_domain_recommendations(
            user_id, source_domain, target_domain,
            selected_config, template_type
        )

        if result.get("success", False):
            print(f"\nRecommendations:")
            print("-" * 40)
            print(result["recommendations"])
            print(f"\nResponse Time: {result['response_time']:.2f}s")
            print(f"Quality Score: {self._estimate_response_quality(result['recommendations']):.2f}/5.0")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    def _interactive_model_comparison(self):
        """Compare all model configurations for one user."""
        user_id = input("Enter user ID: ").strip()

        if user_id not in self.user_history:
            print(f"User {user_id} not found")
            return

        user_data = self.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]

        if len(domains) < 2:
            print("User needs at least 2 domains for comparison")
            return

        print(f"Available domains: {', '.join(domains)}")
        source_domain = input("Source domain: ").strip()
        target_domain = input("Target domain: ").strip()

        if source_domain not in domains or target_domain not in domains:
            print("Invalid domain(s)")
            return

        print(f"\nComparing all model configurations...")
        print(f"User: {user_id} | {source_domain} → {target_domain}")
        print("=" * 60)

        comparison_results = []

        for config_name, config in self.model_configs.items():
            print(f"\nTesting {config_name}...")

            result = self.generate_cross_domain_recommendations(
                user_id, source_domain, target_domain, config
            )

            if result.get("success", False):
                quality = self._estimate_response_quality(result["recommendations"])
                comparison_results.append({
                    "config": config_name,
                    "quality": quality,
                    "time": result["response_time"],
                    "recommendations": result["recommendations"]
                })

                print(f"Quality: {quality:.2f}/5.0 | Time: {result['response_time']:.2f}s")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")

        # Show comparison summary
        if comparison_results:
            print(f"\nCOMPARISON SUMMARY")
            print("-" * 40)

            # Sort by quality
            comparison_results.sort(key=lambda x: x["quality"], reverse=True)

            for i, result in enumerate(comparison_results, 1):
                marker = "BEST" if i == 1 else f"#{i}"
                print(f"{marker} {result['config']}: {result['quality']:.2f}/5.0")

            # Show best result
            best_result = comparison_results[0]
            print(f"\nBest Result - {best_result['config']}:")
            print("-" * 40)
            print(best_result['recommendations'])


def main():
    """Main function to run enhanced LLAMAREC."""
    parser = argparse.ArgumentParser(
        description="Enhanced LLAMAREC for Cross-Domain Recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive demo
  python run_ollama_llama3_lora.py --mode interactive

  # Run comparison study
  python run_ollama_llama3_lora.py --mode comparison --max-users 30

  # Use additional models
  python run_ollama_llama3_lora.py --additional-models llama3:70b llama2:13b
        """
    )

    parser.add_argument("--user-history", default="data/splits/user_history.json",
                        help="Path to user history JSON file")

    parser.add_argument("--base-model", default="llama3:70b-instruct",
                        help="Base Llama3 model name")

    parser.add_argument("--additional-models", nargs="+",
                        help="List of additional model names to test")

    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama API URL")

    parser.add_argument("--mode", choices=["interactive", "comparison", "demo"],
                        default="interactive",
                        help="Running mode")

    parser.add_argument("--max-users", type=int, default=30,
                        help="Maximum users for comparison study")

    parser.add_argument("--templates", nargs="+",
                        choices=["standard", "detailed", "few_shot"],
                        default=["standard", "detailed"],
                        help="Prompt templates to test")

    parser.add_argument("--domains", nargs="+",
                        help="Domain pairs to test (format: source,target)")

    args = parser.parse_args()

    # Process domain pairs
    domain_pairs = None
    if args.domains:
        domain_pairs = []
        for domain_spec in args.domains:
            if ',' in domain_spec:
                source, target = domain_spec.split(',', 1)
                domain_pairs.append((source.strip(), target.strip()))

    # Initialize engine
    print("Initializing Enhanced LLAMAREC...")

    engine = EnhancedLlamaRecEngine(
        user_history_path=args.user_history,
        base_model=args.base_model,
        additional_models=args.additional_models or [],
        ollama_url=args.ollama_url
    )

    if not engine.user_history:
        print("Cannot proceed without user history")
        return

    # Run based on mode
    if args.mode == "interactive" or args.mode == "demo":
        print("\nStarting Interactive Demo...")
        engine.run_interactive_demo()

    elif args.mode == "comparison":
        print(f"\nRunning Model Comparison Study...")
        study_results = engine.run_simple_comparison_study(
            max_users=args.max_users,
            domains_to_test=domain_pairs,
            template_types=args.templates
        )

        # Basic analysis
        if study_results and "performance_summary" in study_results:
            summary = study_results["performance_summary"]
            if "error" not in summary:
                print(f"\nStudy Summary:")
                print(f"   Successful Tests: {summary['overall']['total_successful']}")
                print(f"   Average Quality: {summary['overall']['avg_quality']:.3f}/5.0")

                if "best_config" in summary:
                    best = summary["best_config"]
                    print(f"   Best Config: {best['name']} ({best['quality']:.3f})")

                if "best_template" in summary:
                    best_template = summary["best_template"]
                    print(f"   Best Template: {best_template['name']} ({best_template['quality']:.3f})")

    print("\nEnhanced LLAMAREC session completed!")


if __name__ == "__main__":
    main()