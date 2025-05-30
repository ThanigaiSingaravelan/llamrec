#!/usr/bin/env python3
"""
LLAMAREC with Ollama - Using your existing user_history.json

This script runs cross-domain recommendations using your existing data
and Ollama's local Llama 3 model.
"""

import json
import os
import requests
import random
from datetime import datetime
from typing import Dict, List


class OllamaLlamaRec:
    """LLAMAREC using Ollama for cross-domain recommendations."""

    def __init__(self, user_history_path="data/splits/user_history.json",
                 ollama_model="llama3:8b", ollama_url="http://localhost:11434"):
        self.user_history_path = user_history_path
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.user_history = self.load_user_history()

        print(f"🚀 LLAMAREC with Ollama {ollama_model}")
        print("=" * 50)
        self.test_ollama_connection()

    def test_ollama_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]

                if self.ollama_model in model_names:
                    print(f"✅ Connected to Ollama. Model {self.ollama_model} is ready!")
                else:
                    print(f"❌ Model {self.ollama_model} not found.")
                    print(f"Available models: {model_names}")
                    print(f"Run: ollama pull {self.ollama_model}")
                    return False
            else:
                print(f"❌ Ollama API returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")
            return False
        return True

    def load_user_history(self) -> Dict:
        """Load your existing user history."""
        if not os.path.exists(self.user_history_path):
            print(f"❌ User history not found: {self.user_history_path}")
            return {}

        try:
            with open(self.user_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)

            print(f"✅ Loaded user history with {len(history)} users")

            # Show available domains
            domains = set()
            for user_data in history.values():
                domains.update(user_data.keys())
            print(f"📚 Available domains: {', '.join(sorted(domains))}")

            return history
        except Exception as e:
            print(f"❌ Error loading user history: {e}")
            return {}

    def call_ollama(self, prompt: str, max_tokens: int = 512) -> str:
        """Call Ollama API with a prompt."""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": max_tokens
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Longer timeout for larger responses
            )

            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return ""

        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return ""

    def format_items(self, items: List, max_items: int = 5) -> str:
        """Format items for prompt."""
        if not items:
            return "[]"

        # Limit number of items
        items = items[:max_items]
        formatted = []

        for item in items:
            if isinstance(item, dict):
                title = item.get('title', item.get('asin', str(item)))
            else:
                title = str(item)

            # Clean title
            title = title.replace('"', "'")
            formatted.append(f'"{title}"')

        return f"[{', '.join(formatted)}]"

    def generate_cross_domain_recommendations(self, user_id: str, source_domain: str,
                                              target_domain: str, num_recs: int = 3) -> str:
        """Generate cross-domain recommendations for a user."""

        if user_id not in self.user_history:
            return f"❌ User {user_id} not found"

        user_data = self.user_history[user_id]

        if source_domain not in user_data or target_domain not in user_data:
            return f"❌ Domains not available for user {user_id}"

        source_items = user_data[source_domain].get('liked', [])
        if not source_items:
            return f"❌ No liked items in {source_domain} for user {user_id}"

        # Format the prompt using your specified format
        formatted_items = self.format_items(source_items)

        prompt = f"""Based on the user's preferences in the {source_domain}, generate personalised recommendations for an item from the {target_domain}.

User's highly-rated items from {source_domain}: {formatted_items}
Target domain Dt: {target_domain}
Task: Recommend top {num_recs} items from domain Dt that the user is most likely to enjoy, along with a brief explanation for each recommendation.

Output Format:
1. item [Explanation]
2. item [Explanation]
3. item [Explanation]

Recommendations:"""

        return self.call_ollama(prompt)

    def evaluate_item_preference(self, user_id: str, source_domain: str,
                                 target_domain: str, target_item: str) -> str:
        """Evaluate if a user would like a specific item."""

        if user_id not in self.user_history:
            return f"❌ User {user_id} not found"

        user_data = self.user_history[user_id]
        source_items = user_data.get(source_domain, {}).get('liked', [])

        if not source_items:
            return f"❌ No liked items in {source_domain} for user {user_id}"

        formatted_items = self.format_items(source_items)

        prompt = f"""Based on the user's preferences in the {source_domain}, predict whether they would enjoy an item from the {target_domain}.

The user has given high ratings to the following items from {source_domain}: {formatted_items}
Predict if the user would enjoy the item titled "{target_item}" from {target_domain}: {target_domain}.
Provide a brief explanation for your prediction.

Answer (Yes/No with explanation):"""

        return self.call_ollama(prompt, max_tokens=256)

    def run_sample_evaluation(self, num_users: int = 50):
        """Run sample cross-domain evaluation."""
        print(f"\n🧪 Running sample evaluation with {num_users} users...")
        print("=" * 60)

        if not self.user_history:
            print("❌ No user history available")
            return

        # Get sample users
        users = list(self.user_history.keys())[:num_users]
        results = []

        for i, user_id in enumerate(users, 1):
            print(f"\n👤 User {i}/{len(users)}: {user_id}")
            print("-" * 40)

            user_data = self.user_history[user_id]
            domains = list(user_data.keys())

            if len(domains) < 2:
                print("⚠️  User has data in less than 2 domains, skipping...")
                continue

            # Try different domain combinations
            for source_domain in domains:
                for target_domain in domains:
                    if source_domain == target_domain:
                        continue

                    source_items = user_data[source_domain].get('liked', [])
                    target_items = user_data[target_domain].get('liked', [])

                    if not source_items or not target_items:
                        continue

                    print(f"\n🔄 {source_domain} → {target_domain}")

                    # Generate recommendations
                    recommendations = self.generate_cross_domain_recommendations(
                        user_id, source_domain, target_domain
                    )

                    print(f"📝 Recommendations:\n{recommendations}")

                    # Test with a real item the user liked
                    test_item = target_items[0]
                    test_item_title = test_item.get('title', str(test_item))

                    evaluation = self.evaluate_item_preference(
                        user_id, source_domain, target_domain, test_item_title
                    )

                    print(f"\n🎯 Evaluation for '{test_item_title}':\n{evaluation}")

                    results.append({
                        "user_id": user_id,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "recommendations": recommendations,
                        "test_item": test_item_title,
                        "evaluation": evaluation,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Only test one combination per user to keep output manageable
                    break
                break

        # Save results
        os.makedirs("results", exist_ok=True)
        results_file = f"results/ollama_llamarec_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": self.ollama_model,
                "timestamp": datetime.now().isoformat(),
                "num_users_tested": len(users),
                "total_recommendations": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"✅ Evaluation complete! Tested {len(results)} cross-domain scenarios.")

    def interactive_demo(self):
        """Interactive demo for testing recommendations."""
        print(f"\n🎮 Interactive LLAMAREC Demo")
        print("=" * 40)

        if not self.user_history:
            print("❌ No user history available")
            return

        # Show available users
        users = list(self.user_history.keys())[:10]  # Show first 10 users
        print(f"Available users (showing first 10): {', '.join(users)}")

        while True:
            print(f"\n{'=' * 50}")
            user_id = input("Enter user ID (or 'quit' to exit): ").strip()

            if user_id.lower() == 'quit':
                break

            if user_id not in self.user_history:
                print(f"❌ User {user_id} not found")
                continue

            user_data = self.user_history[user_id]
            domains = list(user_data.keys())
            print(f"Available domains for {user_id}: {', '.join(domains)}")

            source_domain = input("Enter source domain: ").strip()
            target_domain = input("Enter target domain: ").strip()

            if source_domain not in domains or target_domain not in domains:
                print("❌ Invalid domain")
                continue

            print(f"\n🔄 Generating {source_domain} → {target_domain} recommendations...")

            recommendations = self.generate_cross_domain_recommendations(
                user_id, source_domain, target_domain
            )

            print(f"\n📝 Recommendations:\n{recommendations}")


def main():
    """Main function to run LLAMAREC with Ollama."""

    # Initialize LLAMAREC with Ollama
    llamarec = OllamaLlamaRec()

    if not llamarec.user_history:
        print("❌ Cannot proceed without user history")
        return

    print(f"\n🎯 What would you like to do?")
    print("1. Run sample evaluation (automated)")
    print("2. Interactive demo (manual testing)")
    print("3. Both")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice in ['1', '3']:
        llamarec.run_sample_evaluation(num_users=50)  # Test with 3 users

    if choice in ['2', '3']:
        llamarec.interactive_demo()

    print(f"\n🎉 LLAMAREC with Ollama demo complete!")


if __name__ == "__main__":
    main()