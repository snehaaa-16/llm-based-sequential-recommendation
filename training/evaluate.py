import time
import torch

from utils.metrics import compute_metrics


def evaluate_model(model, dataloader, device="cpu", ks=[10, 20]):
    """
    Evaluate model using Recall@K and NDCG@K
    """

    model.eval()

    total_metrics = {f"recall@{k}": 0 for k in ks}
    total_metrics.update({f"ndcg@{k}": 0 for k in ks})

    num_batches = 0

    with torch.no_grad():
        for sequences, targets, padding_mask in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device)

            logits = model(sequences, padding_mask)

            # MovieLens IDs start from 1
            targets = targets - 1

            metrics = compute_metrics(logits, targets, ks)

            for key in metrics:
                total_metrics[key] += metrics[key]

            num_batches += 1

    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= num_batches

    # Print results
    print("Evaluation Results:")
    for key, value in total_metrics.items():
        print(f"{key}: {value:.4f}")

    return total_metrics


def measure_inference_time(model, dataloader, device="cpu"):
    """
    Measures total inference time
    """

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
