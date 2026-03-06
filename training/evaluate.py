import time
import torch

from utils.metrics import recall_at_k, ndcg_at_k


def evaluate_model(model, dataloader, device="cpu", k=10):
    """
    Evaluates model using Recall@K and NDCG@K
    """

    model.eval()

    total_recall = 0
    total_ndcg = 0
    num_batches = 0

    with torch.no_grad():
        for sequences, targets in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)

            logits = model(sequences)

            # MovieLens IDs start from 1
            targets = targets - 1

            total_recall += recall_at_k(logits, targets, k).item()
            total_ndcg += ndcg_at_k(logits, targets, k).item()

            num_batches += 1

    avg_recall = total_recall / num_batches
    avg_ndcg = total_ndcg / num_batches

    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")

    return avg_recall, avg_ndcg
    total_time = end - start
    print(f"Inference Time: {total_time:.4f} seconds")

    return total_time
