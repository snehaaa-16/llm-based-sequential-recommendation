import time
import torch

from utils.metrics import compute_metrics


def evaluate_model(
    model,
    dataloader,
    device="cpu",
    ks=[10, 20],
    external_targets=None
):
    """
    Evaluate model.

    If external_targets is provided:
        uses those instead of dataset targets (for val/test split)
    """

    model.eval()

    total_metrics = {f"recall@{k}": 0 for k in ks}
    total_metrics.update({f"ndcg@{k}": 0 for k in ks})

    num_batches = 0
    global_idx = 0  # track position for external targets

    with torch.no_grad():
        for sequences, targets, padding_mask in dataloader:

            batch_size = sequences.size(0)

            sequences = sequences.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(sequences, padding_mask)

            # -----------------------
            # Use correct targets
            # -----------------------
            if external_targets is not None:
                batch_targets = external_targets[
                    global_idx: global_idx + batch_size
                ]
                batch_targets = torch.tensor(
                    batch_targets,
                    dtype=torch.long,
                    device=device
                )
                global_idx += batch_size
            else:
                batch_targets = targets.to(device)

            # shift indexing
            batch_targets = batch_targets - 1

            metrics = compute_metrics(logits, batch_targets, ks)

            for key in metrics:
                total_metrics[key] += metrics[key]

            num_batches += 1

    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= num_batches

    print("Evaluation Results:")
    for key, value in total_metrics.items():
        print(f"{key}: {value:.4f}")

    return total_metrics


def measure_inference_time(model, dataloader, device="cpu"):

    model.eval()

    start = time.time()

    with torch.no_grad():
        for sequences, _, padding_mask in dataloader:
            sequences = sequences.to(device)
            padding_mask = padding_mask.to(device)
            model(sequences, padding_mask)

    total_time = time.time() - start

    print(f"Inference Time: {total_time:.4f} seconds")

    return total_time
