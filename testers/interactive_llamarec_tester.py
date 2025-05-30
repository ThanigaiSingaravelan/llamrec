#!/usr/bin/env python3
"""
Interactive LLAMAREC Tester
Enhanced interactive testing interface for LLAMAREC with detailed analysis and comparison features.
"""

import json
import os
import requests
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd


class InteractiveLlamaRecTester:
    """Enhanced interactive testing interface for LLAMAREC"""

    def __init__(self, user_history_path="data/splits/user_history.json",
                 ollama_model="llama3:8b", ollama_url="http://localhost:11434"):
        self.user_history_path = user_history_path
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.user_history = self.load_user_history()
        self.session_results = []

        print("🎯 Interactive LLAMAREC Tester")
        print("=" * 50)
        self.test_connection()
        self.display_dataset_stats()

    def test_connection(self):
        """Test Ollama connection and model availability"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                if self.ollama_model in models:
                    print(f"✅ Connected to Ollama. Model {self.ollama_model} ready!")
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
        return True

    def load_user_history(self) -> Dict:
        """Load user history with detailed statistics"""
        if not os.path.exists(self.user_history_path):
            print(f"❌ User history not found: {self.user_history_path}")
            return {}

        try:
            with open(self.user_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            print(f"✅ Loaded {len(history)} users")
            return history
        except Exception as e:
            print(f"❌ Error loading user history: {e}")
            return {}

    def display_dataset_stats(self):
        """Display comprehensive dataset statistics"""
        if not self.user_history:
            return

        print("\n📊 Dataset Statistics:")
        print("-" * 30)

        # Domain statistics
        domain_stats = defaultdict(int)
        user_domain_counts = defaultdict(int)
        total_interactions = 0

        for user_id, domains in self.user_history.items():
            active_domains = 0
            for domain, data in domains.items():
                if data.get('count', 0) > 0:
                    domain_stats[domain] += 1
                    active_domains += 1
                    total_interactions += data.get('count', 0)
            user_domain_counts[active_domains] += 1

        print("Domain Coverage:")
        for domain, count in sorted(domain_stats.items()):
            print(f"  {domain}: {count} users")

        print(f"\nMulti-domain Users:")
        for domain_count, user_count in sorted(user_domain_counts.items()):
            print(f"  {domain_count} domains: {user_count} users")

        print(f"\nTotal Interactions: {total_interactions:,}")

    def call_ollama(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call Ollama API with customizable parameters"""
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
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()["response"].strip()
                response_time = end_time - start_time
                print(f"⏱️ Response time: {response_time:.2f}s")
                return result
            else:
                print(f"❌ API error: {response.status_code}")
                return ""

        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return ""

    def format_items(self, items: List, max_items: int = 5) -> str:
        """Format items for prompt with better handling"""
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
            if len(title) > 100:
                title = title[:97] + "..."
            formatted.append(f'"{title}"')

        return f"[{', '.join(formatted)}]"

    def get_user_profile(self, user_id: str) -> Dict:
        """Get detailed user profile"""
        if user_id not in self.user_history:
            return {}

        user_data = self.user_history[user_id]
        profile = {
            "user_id": user_id,
            "domains": {},
            "total_interactions": 0,
            "domain_count": 0
        }

        for domain, data in user_data.items():
            if data.get('count', 0) > 0:
                profile["domains"][domain] = {
                    "liked_count": len(data.get('liked', [])),
                    "disliked_count": len(data.get('disliked', [])),
                    "total_count": data.get('count', 0),
                    "sample_liked": [item.get('title', str(item)) for item in data.get('liked', [])[:3]]
                }
                profile["total_interactions"] += data.get('count', 0)
                profile["domain_count"] += 1

        return profile

    def display_user_profile(self, user_id: str):
        """Display detailed user profile"""
        profile = self.get_user_profile(user_id)
        if not profile:
            print(f"❌ User {user_id} not found")
            return

        print(f"\n👤 User Profile: {user_id}")
        print("=" * 40)
        print(f"Total Interactions: {profile['total_interactions']}")
        print(f"Active Domains: {profile['domain_count']}")

        for domain, stats in profile["domains"].items():
            print(f"\n📚 {domain}:")
            print(f"  Liked: {stats['liked_count']}, Disliked: {stats['disliked_count']}")
            if stats['sample_liked']:
                print(f"  Sample Liked Items:")
                for item in stats['sample_liked']:
                    print(f"    • {item[:80]}{'...' if len(item) > 80 else ''}")

    def generate_recommendation(self, user_id: str, source_domain: str,
                                target_domain: str, num_recs: int = 3,
                                temperature: float = 0.7) -> Dict:
        """Generate recommendation with detailed tracking"""

        if user_id not in self.user_history:
            return {"error": f"User {user_id} not found"}

        user_data = self.user_history[user_id]

        if source_domain not in user_data or target_domain not in user_data:
            return {"error": f"Required domains not available for user {user_id}"}

        source_items = user_data[source_domain].get('liked', [])
        if not source_items:
            return {"error": f"No liked items in {source_domain} for user {user_id}"}

        formatted_items = self.format_items(source_items)

        # Enhanced prompt with reasoning
        prompt = f"""You are an expert cross-domain recommendation system. Analyze user preferences to make accurate recommendations.

**User Analysis:**
- Source Domain: {source_domain}
- User's Highly-Rated Items: {formatted_items}
- Target Domain: {target_domain}

**Task:** Recommend {num_recs} items from {target_domain} that this user would most likely enjoy.

**Analysis Process:**
1. Identify patterns in the user's {source_domain} preferences
2. Consider how these patterns translate to {target_domain}
3. Recommend items that match these cross-domain patterns

**Output Format:**
1. **[Item Title]** - [Specific reason based on user's preferences]
2. **[Item Title]** - [Specific reason based on user's preferences]
3. **[Item Title]** - [Specific reason based on user's preferences]

**Recommendations:**"""

        start_time = time.time()
        result = self.call_ollama(prompt, max_tokens=600, temperature=temperature)
        end_time = time.time()

        recommendation_data = {
            "user_id": user_id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "source_items": formatted_items,
            "recommendations": result,
            "timestamp": datetime.now().isoformat(),
            "response_time": end_time - start_time,
            "temperature": temperature,
            "num_requested": num_recs
        }

        self.session_results.append(recommendation_data)
        return recommendation_data

    def evaluate_item_preference(self, user_id: str, source_domain: str,
                                 target_domain: str, target_item: str,
                                 temperature: float = 0.3) -> Dict:
        """Evaluate specific item preference"""

        if user_id not in self.user_history:
            return {"error": f"User {user_id} not found"}

        user_data = self.user_history[user_id]
        source_items = user_data.get(source_domain, {}).get('liked', [])

        if not source_items:
            return {"error": f"No liked items in {source_domain} for user {user_id}"}

        formatted_items = self.format_items(source_items)

        prompt = f"""Analyze whether this user would enjoy a specific item based on their preferences.

**User's Preferences in {source_domain}:**
{formatted_items}

**Target Item to Evaluate:**
"{target_item}" from {target_domain}

**Analysis Steps:**
1. What patterns do you see in their {source_domain} preferences?
2. What aspects of "{target_item}" align with these patterns?
3. What aspects might not align?

**Prediction:** [Yes/No]
**Confidence:** [High/Medium/Low]
**Reasoning:** [Detailed explanation]

**Analysis:**"""

        result = self.call_ollama(prompt, max_tokens=400, temperature=temperature)

        evaluation_data = {
            "user_id": user_id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "target_item": target_item,
            "evaluation": result,
            "timestamp": datetime.now().isoformat(),
            "temperature": temperature
        }

        return evaluation_data

    def compare_recommendations(self, user_id: str, target_domain: str,
                                source_domains: List[str]) -> Dict:
        """Compare recommendations from different source domains"""

        print(f"\n Comparing Recommendations for User {user_id}")
        print(f"Target Domain: {target_domain}")
        print("=" * 50)

        comparisons = {}

        for source_domain in source_domains:
            if source_domain == target_domain:
                continue

            print(f"\n {source_domain} → {target_domain}")
            print("-" * 30)

            rec_data = self.generate_recommendation(user_id, source_domain, target_domain)

            if "error" in rec_data:
                print(f" {rec_data['error']}")
                continue

            print(f" Recommendations:\n{rec_data['recommendations']}")
            comparisons[source_domain] = rec_data

        return comparisons

    def batch_test_users(self, num_users: int = 5, save_results: bool = True):
        """Test multiple users systematically"""

        print(f"\n Batch Testing {num_users} Users")
        print("=" * 40)

        # Get users with good cross-domain coverage
        eligible_users = []
        for user_id, domains in self.user_history.items():
            active_domains = sum(1 for d in domains.values() if d.get('count', 0) > 0)
            if active_domains >= 2:
                eligible_users.append((user_id, active_domains))

        # Sort by domain coverage and take top users
        eligible_users.sort(key=lambda x: x[1], reverse=True)
        test_users = [user_id for user_id, _ in eligible_users[:num_users]]

        batch_results = []

        for i, user_id in enumerate(test_users, 1):
            print(f"\n👤 Testing User {i}/{len(test_users)}: {user_id}")

            user_data = self.user_history[user_id]
            domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]

            # Test first available domain pair
            if len(domains) >= 2:
                source_domain = domains[0]
                target_domain = domains[1]

                rec_data = self.generate_recommendation(user_id, source_domain, target_domain)
                if "error" not in rec_data:
                    batch_results.append(rec_data)
                    print(f" Generated {source_domain} → {target_domain} recommendations")
                else:
                    print(f" {rec_data['error']}")

        if save_results and batch_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/batch_test_results_{timestamp}.json"
            os.makedirs("results", exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_info": {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.ollama_model,
                        "num_users_tested": len(test_users),
                        "successful_tests": len(batch_results)
                    },
                    "results": batch_results
                }, f, indent=2, ensure_ascii=False)

            print(f"\n Results saved to: {filename}")

        return batch_results

    def save_session_results(self):
        """Save all session results"""
        if not self.session_results:
            print("No results to save.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/interactive_session_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "session_info": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.ollama_model,
                    "total_tests": len(self.session_results)
                },
                "results": self.session_results
            }, f, indent=2, ensure_ascii=False)

        print(f" Session results saved to: {filename}")

    def run_interactive_session(self):
        """Main interactive testing loop"""

        if not self.user_history:
            print(" No user history available for testing")
            return

        # Show sample users
        sample_users = list(self.user_history.keys())[:10]
        print(f"\n Sample Users: {', '.join(sample_users)}")

        while True:
            print(f"\n{'=' * 60}")
            print(" Interactive LLAMAREC Testing Menu")
            print("=" * 60)
            print("1.  Test Single User Recommendation")
            print("2.  Evaluate Specific Item")
            print("3.  Compare Cross-Domain Sources")
            print("4.  View User Profile")
            print("5.  Batch Test Multiple Users")
            print("6.  Save Session Results")
            print("7.  View Session Statistics")
            print("8.  Exit")

            choice = input("\nEnter your choice (1-8): ").strip()

            if choice == '1':
                self.test_single_recommendation()
            elif choice == '2':
                self.test_item_evaluation()
            elif choice == '3':
                self.test_cross_domain_comparison()
            elif choice == '4':
                self.view_user_profile()
            elif choice == '5':
                num_users = int(input("Number of users to test (default 5): ") or "5")
                self.batch_test_users(num_users)
            elif choice == '6':
                self.save_session_results()
            elif choice == '7':
                self.show_session_stats()
            elif choice == '8':
                print("👋 Goodbye!")
                self.save_session_results()
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def test_single_recommendation(self):
        """Interactive single recommendation test"""
        user_id = input("Enter user ID: ").strip()

        if user_id not in self.user_history:
            print(f"❌ User {user_id} not found")
            return

        # Show available domains
        user_data = self.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]
        print(f"Available domains: {', '.join(domains)}")

        source_domain = input("Source domain: ").strip()
        target_domain = input("Target domain: ").strip()

        if source_domain not in domains or target_domain not in domains:
            print("❌ Invalid domain(s)")
            return

        # Optional parameters
        num_recs = int(input("Number of recommendations (default 3): ") or "3")
        temp = float(input("Temperature 0.1-1.0 (default 0.7): ") or "0.7")

        print(f"\n🔄 Generating {source_domain} → {target_domain} recommendations...")

        result = self.generate_recommendation(user_id, source_domain, target_domain, num_recs, temp)

        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n🎯 Recommendations:\n{result['recommendations']}")

    def test_item_evaluation(self):
        """Interactive item evaluation test"""
        user_id = input("Enter user ID: ").strip()

        if user_id not in self.user_history:
            print(f"❌ User {user_id} not found")
            return

        user_data = self.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]
        print(f"Available domains: {', '.join(domains)}")

        source_domain = input("Source domain: ").strip()
        target_domain = input("Target domain: ").strip()
        target_item = input("Item to evaluate: ").strip()

        if source_domain not in domains:
            print("❌ Invalid source domain")
            return

        print(f"\n🎯 Evaluating '{target_item}'...")

        result = self.evaluate_item_preference(user_id, source_domain, target_domain, target_item)

        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"\n📊 Evaluation:\n{result['evaluation']}")

    def test_cross_domain_comparison(self):
        """Interactive cross-domain comparison"""
        user_id = input("Enter user ID: ").strip()

        if user_id not in self.user_history:
            print(f"❌ User {user_id} not found")
            return

        user_data = self.user_history[user_id]
        domains = [d for d, data in user_data.items() if data.get('count', 0) > 0]
        print(f"Available domains: {', '.join(domains)}")

        target_domain = input("Target domain: ").strip()
        source_domains = [d for d in domains if d != target_domain]

        if not source_domains:
            print("❌ Need at least 2 domains for comparison")
            return

        self.compare_recommendations(user_id, target_domain, source_domains)

    def view_user_profile(self):
        """Interactive user profile viewer"""
        user_id = input("Enter user ID: ").strip()
        self.display_user_profile(user_id)

    def show_session_stats(self):
        """Show current session statistics"""
        if not self.session_results:
            print("No tests performed in this session yet.")
            return

        print(f"\n📈 Session Statistics")
        print("=" * 30)
        print(f"Total Tests: {len(self.session_results)}")

        # Domain pair counts
        domain_pairs = defaultdict(int)
        for result in self.session_results:
            if "error" not in result:
                pair = f"{result['source_domain']} → {result['target_domain']}"
                domain_pairs[pair] += 1

        print(f"\nDomain Pair Frequency:")
        for pair, count in sorted(domain_pairs.items()):
            print(f"  {pair}: {count}")

        # Average response time
        response_times = [r.get('response_time', 0) for r in self.session_results if 'response_time' in r]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"\nAverage Response Time: {avg_time:.2f}s")


def main():
    """Main function to run interactive tester"""
    tester = InteractiveLlamaRecTester(
        user_history_path="data/splits/user_history.json",
        ollama_model="llama3:8b"
    )

    if not tester.user_history:
        print("❌ Cannot proceed without user history")
        return

    print("\n🚀 Starting Interactive Testing Session...")
    tester.run_interactive_session()


if __name__ == "__main__":
    main()