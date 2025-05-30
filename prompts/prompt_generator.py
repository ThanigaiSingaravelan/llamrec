#!/usr/bin/env python3
"""
Basic Prompt Generator Functions for LLAMAREC
Compatible with the test file imports
"""

import os
import json
import pandas as pd
import random
import re
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Basic template for compatibility
TEMPLATE = """Based on the user's preferences in {source_domain}, generate personalised recommendations for an item from the {target_domain}.

User's highly-rated items from {source_domain}: {user_history}
Target domain Dt: {target_domain}
Task: Recommend top {num_recs} items from domain Dt that the user is most likely to enjoy, along with a brief explanation for each recommendation.

Output Format:
1. item [Explanation]
2. item [Explanation]
3. item [Explanation]

Recommendations:"""


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data with error handling."""
    try:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()


def get_item_descriptions(df: pd.DataFrame) -> Dict[str, str]:
    """Extract item descriptions from DataFrame."""
    if df.empty:
        return {}

    item_map = {}
    has_title = "title" in df.columns

    for _, row in df.iterrows():
        asin = row["asin"]
        if has_title and pd.notnull(row.get("title")):
            title = str(row["title"]).strip()
            item_map[asin] = title
        elif "summary" in df.columns and pd.notnull(row.get("summary")):
            summary = str(row["summary"]).strip()
            item_map[asin] = summary
        else:
            item_map[asin] = f"Item {asin}"

    return item_map


def get_top_items(df: pd.DataFrame, user_id: str, item_map: Dict[str, str],
                  max_items: int = 5, min_rating: float = 4.0) -> str:
    """Get top-rated items for a user."""
    if df.empty or user_id not in df['reviewerID'].values:
        return "No items found"

    user_items = df[df["reviewerID"] == user_id]

    # Filter by minimum rating
    high_rated = user_items[user_items["overall"] >= min_rating]
    if len(high_rated) < 2:  # Fallback to lower rating
        high_rated = user_items[user_items["overall"] >= 3.5]

    if high_rated.empty:
        return "No items found"

    # Sort by rating
    sorted_items = high_rated.sort_values("overall", ascending=False)

    # Select items
    selected_items = []
    for _, row in sorted_items.head(max_items).iterrows():
        asin = row["asin"]
        if asin in item_map:
            title = item_map[asin]
            selected_items.append(f'"{title}"')

    return "\n".join(f"• {item}" for item in selected_items) if selected_items else "No items found"


def get_output_items(df: pd.DataFrame, user_id: str, item_map: Dict[str, str],
                     max_items: int = 3, min_rating: float = 4.0) -> Optional[str]:
    """Get expected output items for evaluation."""
    if df.empty or user_id not in df['reviewerID'].values:
        return None

    user_items = df[df["reviewerID"] == user_id]
    high_rated = user_items[user_items["overall"] >= min_rating]

    if len(high_rated) < max_items:
        high_rated = user_items[user_items["overall"] >= 3.5]

    if high_rated.empty:
        return None

    # Sort and select items
    sorted_items = high_rated.sort_values("overall", ascending=False)
    selected_asins = sorted_items["asin"].head(max_items).tolist()

    output_lines = []
    for i, asin in enumerate(selected_asins, 1):
        if asin in item_map:
            title = item_map[asin].split(" - ")[0]  # Get just the title part
            output_lines.append(f'{i}. "{title}" - Matches the user\'s demonstrated preferences')

    return "\n".join(output_lines) if output_lines else None


def generate_prompts(source_df: pd.DataFrame, target_df: pd.DataFrame,
                     source_map: Dict[str, str], target_map: Dict[str, str],
                     source_domain: str, target_domain: str,
                     max_prompts: int = 100) -> List[Dict]:
    """Generate prompts for cross-domain recommendations."""

    if source_df.empty or target_df.empty:
        print(f"Warning: Empty dataframes for {source_domain} -> {target_domain}")
        return []

    # Find common users
    source_users = set(source_df["reviewerID"].unique())
    target_users = set(target_df["reviewerID"].unique())
    common_users = list(source_users.intersection(target_users))

    if not common_users:
        print(f"Warning: No common users between {source_domain} and {target_domain}")
        return []

    random.shuffle(common_users)
    prompts = []

    for user_id in tqdm(common_users[:max_prompts], desc=f"Generating {source_domain} -> {target_domain}"):
        # Check if user has enough items in both domains
        source_count = len(source_df[source_df["reviewerID"] == user_id])
        target_count = len(target_df[target_df["reviewerID"] == user_id])

        if source_count < 2 or target_count < 1:
            continue

        # Get user's source domain items
        user_history = get_top_items(source_df, user_id, source_map, max_items=5)
        if user_history == "No items found":
            continue

        # Get expected output
        expected_output = get_output_items(target_df, user_id, target_map)
        if not expected_output:
            continue

        # Create prompt
        prompt_text = TEMPLATE.format(
            source_domain=source_domain,
            target_domain=target_domain,
            user_history=user_history,
            num_recs=3
        )

        prompt_data = {
            "input": prompt_text,
            "output": expected_output,
            "user_id": user_id,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "template_type": "standard",
            "source_items_count": len(user_history.split('\n')),
            "metadata": {
                "user_history": user_history,
                "expected_items": expected_output.count('\n') + 1 if expected_output else 0
            }
        }

        prompts.append(prompt_data)

    print(f"Generated {len(prompts)} prompts for {source_domain} -> {target_domain}")
    return prompts


# For compatibility with existing imports
def main():
    """Main function placeholder."""
    print("Prompt generator module loaded successfully")


if __name__ == "__main__":
    main()