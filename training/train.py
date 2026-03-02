import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings

from models.hierarchical_model import HierarchicalLLMRec
from utils.metrics import recall_at_k, ndcg_at_k
from utils.config import get_device, DEFAULT_CONFIG


def train():

    config = DEFAULT_CONFIG
    device = get_device()
    print(f"Using device: {device}")

    # ---------------------------
    # Load Dataset
    # ---------------------------
    ratings, movies = load_ml1m()
    user_sequences = create_user_sequences(ratings)

    dataset = SequentialDataset(user_sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # ---------------------------
    # Build Item Embeddings
    # ---------------------------
    item_embeddings = build_item_embeddings()
    hidden_dim = item_embeddings.shape[1]
    num_items = item_embeddings.shape[0]

    # ---------------------------
    # Build Model
    # ---------------------------
    model = HierarchicalLLMRec(num_items, hidden_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Training Loop
    # ---------------------------
    model.train()

    for epoch in range(config["training"]["epochs"]):

        total_loss = 0
        total_recall = 0
        total_ndcg = 0
        num_batches = 0

        for sequences, targets in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)

            logits = model(sequences)

            # MovieLens IDs start at 1 → shift
            targets_adjusted = targets - 1

            loss = criterion(logits, targets_adjusted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Metrics
            total_recall += recall_at_k(
                logits, targets_adjusted,
                k=config["model"]["top_k"]
            ).item()

            total_ndcg += ndcg_at_k(
                logits, targets_adjusted,
                k=config["model"]["top_k"]
            ).item()

            num_batches += 1

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {total_loss:.4f} | "
            f"Recall@{config['model']['top_k']}: {total_recall/num_batches:.4f} | "
            f"NDCG@{config['model']['top_k']}: {total_ndcg/num_batches:.4f}"
        )


if __name__ == "__main__":
    train()
