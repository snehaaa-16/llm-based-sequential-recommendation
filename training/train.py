import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from data.preprocess import (
    load_ml1m,
    create_user_sequences,
    train_val_test_split
)

from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings

from models.hierarchical_model import HierarchicalLLMRec

from training.evaluate import (
    evaluate_model,
    measure_inference_time
)

from utils.config import (
    load_config,
    get_device,
    set_seed
)


def train():

    # -----------------------
    # Config + Setup
    # -----------------------
    config = load_config()

    set_seed(config["seed"])

    device = get_device(config)

    print(f"Using device: {device}")

    # -----------------------
    # Dataset + Split
    # -----------------------
    ratings, _ = load_ml1m(
        config["dataset"]["data_path"]
    )

    user_sequences = create_user_sequences(ratings)

    train_seq, val_targets, test_targets = (
        train_val_test_split(user_sequences)
    )

    # -----------------------
    # Datasets
    # -----------------------
    train_dataset = SequentialDataset(
        train_seq,
        max_seq_len=config["model"]["max_seq_len"]
    )

    val_dataset = SequentialDataset(
        train_seq,
        max_seq_len=config["model"]["max_seq_len"]
    )

    # -----------------------
    # DataLoaders
    # -----------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        pin_memory=True
    )

    # -----------------------
    # Targets
    # -----------------------
    val_targets_list = list(val_targets.values())
    test_targets_list = list(test_targets.values())

    # -----------------------
    # Item Embeddings
    # -----------------------
    item_embeddings = build_item_embeddings().to(device)

    # -----------------------
    # Model
    # -----------------------
    model = HierarchicalLLMRec(
        item_embeddings=item_embeddings,
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        pooling=config["model"]["pooling"],
        causal=config["model"].get("causal", False)
    ).to(device)

    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"]
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    # -----------------------
    # Checkpoint Setup
    # -----------------------
    best_recall = 0.0

    checkpoint_dir = config["logging"].get(
        "checkpoint_path",
        "checkpoints"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(config["training"]["epochs"]):

        model.train()

        total_loss = 0
        total_samples = 0

        for sequences, targets, padding_mask in train_loader:

            batch_size = sequences.size(0)

            sequences = sequences.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()

            # -----------------------
            # Mixed Precision
            # -----------------------
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                logits = model(sequences, padding_mask)

                targets = targets - 1

                loss = criterion(logits, targets)

            # -----------------------
            # Backward
            # -----------------------
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()

        avg_loss = total_loss / total_samples

        # -----------------------
        # Validation
        # -----------------------
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            ks=[10, 20],
            external_targets=val_targets_list
        )

        # -----------------------
        # Save Best Model
        # -----------------------
        current_recall = val_metrics["recall@10"]

        if current_recall > best_recall:

            best_recall = current_recall

            checkpoint_path = os.path.join(
                checkpoint_dir,
                "best_model.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "recall@10": best_recall
                },
                checkpoint_path
            )

            print(f"Saved best model -> {checkpoint_path}")

        # -----------------------
        # Logging
        # -----------------------
        print(
            f"\nEpoch {epoch+1}/{config['training']['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            + " | ".join(
                [f"{k}: {v:.4f}" for k, v in val_metrics.items()]
            )
        )

    # -----------------------
    # Final Test Evaluation
    # -----------------------
    print("\nFinal Test Evaluation")

    test_metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        ks=[10, 20],
        external_targets=test_targets_list
    )

    # -----------------------
    # Inference Benchmark
    # -----------------------
    measure_inference_time(
        model,
        val_loader,
        device=device
    )

    return model, test_metrics


if __name__ == "__main__":
    train()
