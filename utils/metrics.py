import torch


def recall_at_k(logits, targets, k=10):
    """
    Recall@K (batch-wise)
    """

    k = min(k, logits.size(1))

    top_k = torch.topk(logits, k=k, dim=1).indices  # (B, K)
    targets = targets.view(-1, 1)

    hits = (top_k == targets).any(dim=1).float()

    return hits.mean()


def ndcg_at_k(logits, targets, k=10):
    """
    Proper NDCG@K implementation (batch-wise)
    """

    k = min(k, logits.size(1))

    top_k = torch.topk(logits, k=k, dim=1).indices  # (B, K)
    targets = targets.view(-1, 1)

    # Compare with target → (B, K)
    hits = (top_k == targets).float()

    # Create rank positions: 1, 2, ..., K
    device = logits.device
    ranks = torch.arange(1, k + 1, device=device).float()

    # Compute DCG
    dcg = (hits / torch.log2(ranks + 1)).sum(dim=1)

    # IDCG = 1 / log2(1 + 1) = 1
    idcg = torch.ones_like(dcg)

    ndcg = dcg / idcg

    return ndcg.mean()


def compute_metrics(logits, targets, ks=[10, 20]):
    """
    Compute multiple metrics at once
    """

    results = {}

    for k in ks:
        results[f"recall@{k}"] = recall_at_k(logits, targets, k).item()
        results[f"ndcg@{k}"] = ndcg_at_k(logits, targets, k).item()

    return results
