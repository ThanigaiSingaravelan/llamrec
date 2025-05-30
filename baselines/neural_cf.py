import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import json
import os
import re
import argparse
import sys
import numpy as np

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from quantitative_evaluator import RecommendationEvaluator
except ImportError:
    try:
        from evaluations.quantitative_evaluator import RecommendationEvaluator
    except ImportError:
        print("Warning: Could not import RecommendationEvaluator. Creating minimal version...")


        # Minimal evaluator for NCF
        class RecommendationEvaluator:
            def __init__(self, user_history_path):
                self.user_history_path = user_history_path
                with open(user_history_path, 'r') as f:
                    self.user_history = json.load(f)

            def evaluate_recommendations(self, results_file, k=3):
                return {"message": "Basic evaluation completed"}

            def generate_evaluation_report(self, results):
                return "NCF Baseline Evaluation Complete"


class NCFDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = df['reviewerID'].map(user2idx).values
        self.items = df['asin'].map(item2idx).values
        self.labels = df['overall'].values.astype('float32')

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc_layers(x).squeeze()


def train_ncf(df, embed_dim=32, epochs=5, batch_size=256, lr=0.001):
    """Train Neural Collaborative Filtering model"""
    print(f"Training NCF with {len(df)} interactions...")

    user_ids = df['reviewerID'].unique()
    item_ids = df['asin'].unique()
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {iid: i for i, iid in enumerate(item_ids)}

    print(f"Users: {len(user2idx)}, Items: {len(item2idx)}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = NCFDataset(train_df, user2idx, item2idx)
    val_dataset = NCFDataset(val_df, user2idx, item2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = NCF(len(user2idx), len(item2idx), embed_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        for users, items, ratings in train_loader:
            users = users.long()
            items = items.long()
            preds = model(users, items)
            loss = criterion(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model, user2idx, item2idx


def recommend_ncf(model, user_id, user2idx, item2idx, k=5):
    """Generate recommendations using trained NCF model"""
    model.eval()
    uid = user2idx.get(user_id)
    if uid is None:
        return []

    with torch.no_grad():
        user_tensor = torch.tensor([uid] * len(item2idx))
        item_tensor = torch.tensor(list(item2idx.values()))
        scores = model(user_tensor, item_tensor).numpy()

    top_indices = scores.argsort()[-k:][::-1]
    inv_item2idx = {v: k for k, v in item2idx.items()}
    return [inv_item2idx[i] for i in top_indices]


def clean_title(text):
    """Clean title for matching"""
    if pd.isna(text):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower().strip())


def evaluate_ncf_with_ground_truth(df, user_history_path, domain, output_path="results/ncf_results.json", k=3):
    """Evaluate NCF with ground truth comparison"""
    print(f"🤖 Starting NCF Evaluation for {domain}")
    print("=" * 50)

    # Train the model
    model, user2idx, item2idx = train_ncf(df)
    users = df['reviewerID'].unique()

    # Load metadata if available
    base_dir = os.path.dirname(os.path.dirname(output_path))
    meta_path = os.path.join(base_dir, "data", "processed", f"{domain}_metadata.csv")

    asin2title = {}
    if os.path.exists(meta_path):
        print(f"Loading metadata from {meta_path}")
        try:
            meta_df = pd.read_csv(meta_path)
            asin2title = {row['asin']: row['title'] for _, row in meta_df.iterrows()
                          if pd.notnull(row.get('title'))}
            print(f"Loaded {len(asin2title)} item titles")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    else:
        print(f"Metadata file not found: {meta_path}")

    # Create clean title mapping for matching
    asin2clean = {asin: clean_title(title) for asin, title in asin2title.items()}

    # Load user history
    print(f"Loading user history from {user_history_path}")
    with open(user_history_path, 'r') as f:
        user_history = json.load(f)

    # Generate recommendations and compare
    skipped = 0
    mismatches = 0
    results = []

    print(f"Generating recommendations for {len(users)} users...")

    for i, user_id in enumerate(users):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(users)} users processed")

        if user_id not in user2idx:
            continue

        if user_id not in user_history or domain not in user_history[user_id]:
            skipped += 1
            continue

        liked_items = user_history[user_id][domain].get("liked")
        if not liked_items:
            skipped += 1
            continue

        # Generate recommendations
        rec_items = recommend_ncf(model, user_id, user2idx, item2idx, k=k)
        rec_titles = [asin2title.get(asin, f"Item_{asin}") for asin in rec_items]
        liked_titles = [item.get("title", "") for item in liked_items]

        # Check for matches using clean titles
        cleaned_recs = {clean_title(t) for t in rec_titles}
        cleaned_likes = {clean_title(t) for t in liked_titles}
        overlap = cleaned_recs.intersection(cleaned_likes)

        if not overlap:
            mismatches += 1

        # Format recommendations as text
        rec_text = "\n".join([f"{i + 1}. {title} - Neural CF recommendation"
                              for i, title in enumerate(rec_titles)])

        results.append({
            "user_id": user_id,
            "source_domain": domain,
            "target_domain": domain,
            "recommendations": rec_text,
            "success": True,
            "model_type": "NCF",
            "timestamp": pd.Timestamp.now().isoformat()
        })

    print(f"\n📊 NCF Evaluation Results:")
    print(f"✅ Generated recommendations for {len(results)} users")
    print(f"⏭️ Skipped {skipped} users (insufficient data)")
    print(f"🔍 {mismatches} users had no matching titles in liked items")
    print(f"📈 Match rate: {(len(results) - mismatches) / len(results) * 100:.1f}%")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "model_info": {
                "type": "Neural Collaborative Filtering",
                "domain": domain,
                "users_trained": len(user2idx),
                "items_trained": len(item2idx),
                "recommendations_generated": len(results)
            },
            "results": results
        }, f, indent=2)

    print(f"💾 Results saved to: {output_path}")

    # Run evaluation if evaluator is available
    try:
        evaluator = RecommendationEvaluator(user_history_path)
        eval_results = evaluator.evaluate_recommendations(output_path, k=k)
        report = evaluator.generate_evaluation_report(eval_results)
        print("\n📋 Evaluation Report:")
        print(report)
        return eval_results
    except Exception as e:
        print(f"⚠️ Evaluation step failed: {e}")
        return {"message": "NCF training completed, but evaluation failed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCF evaluation")
    parser.add_argument("--dataset", required=True, help="Path to CSV with user-item ratings")
    parser.add_argument("--user_history", required=True, help="Path to user history JSON")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., Books, Movies_and_TV)")
    parser.add_argument("--output", default="results/ncf_results.json", help="Path to save results")
    parser.add_argument("--k", type=int, default=3, help="Top-K items for evaluation")
    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)

    if not os.path.exists(args.user_history):
        print(f"❌ User history file not found: {args.user_history}")
        sys.exit(1)

    # Load and validate dataset
    print(f"📁 Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)

    required_columns = ['reviewerID', 'asin', 'overall']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"✅ Dataset loaded: {len(df)} interactions, {df['reviewerID'].nunique()} users, {df['asin'].nunique()} items")

    # Run evaluation
    evaluate_ncf_with_ground_truth(
        df,
        user_history_path=args.user_history,
        domain=args.domain,
        output_path=args.output,
        k=args.k
    )