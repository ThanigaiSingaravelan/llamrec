import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from quantitative_evaluator import RecommendationEvaluator
import json
import os
import re
import argparse


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
    user_ids = df['reviewerID'].unique()
    item_ids = df['asin'].unique()
    user2idx = {uid: i for i, uid in enumerate(user_ids)}
    item2idx = {iid: i for i, iid in enumerate(item_ids)}

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = NCFDataset(train_df, user2idx, item2idx)
    val_dataset = NCFDataset(val_df, user2idx, item2idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = NCF(len(user2idx), len(item2idx), embed_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for users, items, ratings in train_loader:
            users = users.long()
            items = items.long()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} - Loss: {loss.item():.4f}")

    return model, user2idx, item2idx


def recommend_ncf(model, user_id, user2idx, item2idx, k=5):
    model.eval()
    uid = user2idx.get(user_id)
    if uid is None:
        return []

    user_tensor = torch.tensor([uid] * len(item2idx))
    item_tensor = torch.tensor(list(item2idx.values()))
    scores = model(user_tensor, item_tensor).detach().numpy()

    top_indices = scores.argsort()[-k:][::-1]
    inv_item2idx = {v: k for k, v in item2idx.items()}
    return [inv_item2idx[i] for i in top_indices]


def clean_title(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def evaluate_ncf_with_ground_truth(df, user_history_path, domain, output_path="results/ncf_results.json", k=3):
    model, user2idx, item2idx = train_ncf(df)
    users = df['reviewerID'].unique()

    meta_path = f"data/processed/{domain}_metadata.csv"
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        asin2title = {row['asin']: row['title'] for _, row in meta_df.iterrows() if pd.notnull(row.get('title'))}
    else:
        asin2title = {}

    asin2clean = {asin: clean_title(title) for asin, title in asin2title.items()}

    with open(user_history_path, 'r') as f:
        user_history = json.load(f)

    skipped = 0
    mismatches = 0
    results = []
    for user_id in users:
        if user_id not in user2idx:
            continue
        if user_id not in user_history or domain not in user_history[user_id]:
            skipped += 1
            continue
        liked_items = user_history[user_id][domain].get("liked")
        if not liked_items:
            skipped += 1
            continue

        rec_items = recommend_ncf(model, user_id, user2idx, item2idx, k=k)
        rec_titles = [asin2title.get(asin, asin) for asin in rec_items]
        liked_titles = [item.get("title", "") for item in liked_items]

        cleaned_recs = {clean_title(t) for t in rec_titles}
        cleaned_likes = {clean_title(t) for t in liked_titles}
        overlap = cleaned_recs.intersection(cleaned_likes)

        if not overlap:
            mismatches += 1

        rec_text = "\n".join([f"{i+1}. {title} - Neural CF recommendation" for i, title in enumerate(rec_titles)])

        results.append({
            "user_id": user_id,
            "source_domain": domain,
            "target_domain": domain,
            "recommendations": rec_text
        })

    print(f"✅ Generated recommendations for {len(results)} users, skipped {skipped}.")
    print(f"🔍 {mismatches} users had no matching recommended titles in their liked items.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"results": results}, f, indent=2)

    evaluator = RecommendationEvaluator(user_history_path)
    eval_results = evaluator.evaluate_recommendations(output_path, k=k)
    print(evaluator.generate_evaluation_report(eval_results))

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCF evaluation")
    parser.add_argument("--dataset", required=True, help="Path to CSV with user-item ratings")
    parser.add_argument("--user_history", required=True, help="Path to user history JSON")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., Books, Movies_and_TV)")
    parser.add_argument("--output", default="results/ncf_results.json", help="Path to save results")
    parser.add_argument("--k", type=int, default=10, help="Top-K items for evaluation")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    evaluate_ncf_with_ground_truth(df, args.user_history, args.domain, output_path=args.output, k=args.k)
