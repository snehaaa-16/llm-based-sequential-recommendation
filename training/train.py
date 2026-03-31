import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings

from models.hierarchical_model import HierarchicalLLMRec

from training.evaluate import evaluate_model, measure_inference_time

from utils.config import load_config, get_device, set_seed


def train():

    # -----------------------
    # Config + Setup
    # -----------------------
    config = load_config()
    set_seed(config["seed"])

    device = get_device(config)
    print(f"Using device: {device}")

    # -----------------------
    # Dataset
    # -----------------------
    ratings, _ = load_ml1m(config["dataset"]["data_path"])
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
    # Item Embeddings
    # -----------------------
    item_embeddings = build_item_embeddings()
    item_embeddings = item_embeddings.to(device)

    # -----------------------
    # Model
    # -----------------------
    model = HierarchicalLLMRec(
        item_embeddings,
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)

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
        num_batches = 0

        for sequences, targets, padding_mask in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(sequences, padding_mask)

            # MovieLens IDs start from 1
            targets = targets - 1

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # -----------------------
        # Evaluation
        # -----------------------
        metrics = evaluate_model(
            model,
            dataloader,
            device,
            ks=[10, 20]
        )

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    # -----------------------
    # Inference Timing
    # -----------------------
    measure_inference_time(model, dataloader, device)


if __name__ == "__main__":
    train()
