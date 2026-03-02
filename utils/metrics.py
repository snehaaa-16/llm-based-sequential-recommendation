import torch


def recall_at_k(logits, targets, k=10):
    """
    Computes Recall@K
    logits: (B, num_items)
    targets: (B,)
    """

    top_k = torch.topk(logits, k=k, dim=1).indices  # (B, K)

    targets = targets.view(-1, 1)  # (B, 1)

    hits = (top_k == targets).any(dim=1).float()  # (B,)

    return hits.mean()


def ndcg_at_k(logits, targets, k=10):
    """
    Computes NDCG@K
    logits: (B, num_items)
    targets: (B,)
    """

    top_k = torch.topk(logits, k=k, dim=1).indices  # (B, K)

    targets = targets.view(-1, 1)

    hits = (top_k == targets).nonzero(as_tuple=False)

    if hits.size(0) == 0:
        return torch.tensor(0.0, device=logits.device)

    # hits[:, 1] gives rank positions (0-indexed)
    ranks = hits[:, 1] + 1  # convert to 1-indexed

    ndcg = (1.0 / torch.log2(ranks.float() + 1)).mean()

    return ndcg
