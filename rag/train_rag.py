import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings

from models.hierarchical_model import HierarchicalLLMRec
from rag.retriever import Retriever
from rag.rag_model import RAGSequentialRec

from utils.metrics import recall_at_k, ndcg_at_k
from utils.config import load_config, get_device


def train():

    config = load_config()
    device = get_device(config)

    print(f"Using device: {device}")

    # -----------------------
    # Load Dataset
    # -----------------------
    ratings, movies = load_ml1m(config["dataset"]["data_path"])
    user_sequences = create_user_sequences(ratings)

    dataset = SequentialDataset(
        user_sequences,
        max_seq_len=config["model"]["max_seq_len"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # -----------------------
    # Build Item Embeddings
    # -----------------------
    item_embeddings = build_item_embeddings()
    item_embeddings = item_embeddings.to(device)

    num_items = item_embeddings.shape[0]
    hidden_dim = item_embeddings.shape[1]

    # -----------------------
    # Build Base Model
    # -----------------------
    base_model = HierarchicalLLMRec(num_items, hidden_dim)
    base_model = base_model.to(device)

    # -----------------------
    # Build Retriever
    # -----------------------
    retriever = Retriever(
        item_embeddings.cpu(),
        top_k=config["retrieval"]["top_k"],
        similarity=config["retrieval"]["similarity"]
    )

    # -----------------------
    # Build RAG Model
    # -----------------------
    model = RAGSequentialRec(
        base_model=base_model,
        retriever=retriever,
        item_embeddings=item_embeddings,
        hidden_dim=hidden_dim
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(config["training"]["epochs"]):

        model.train()

        total_loss = 0
        total_recall = 0
        total_ndcg = 0
        num_batches = 0

        for sequences, targets in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)

            logits = model(sequences)

            # MovieLens IDs start from 1
            targets = targets - 1

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Metrics
            total_recall += recall_at_k(
                logits,
                targets,
                config["model"]["top_k"]
            ).item()

            total_ndcg += ndcg_at_k(
                logits,
                targets,
                config["model"]["top_k"]
            ).item()

            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recall = total_recall / num_batches
        avg_ndcg = total_ndcg / num_batches

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Recall@{config['model']['top_k']}: {avg_recall:.4f} | "
            f"NDCG@{config['model']['top_k']}: {avg_ndcg:.4f}"
        )


if __name__ == "__main__":
    train()
