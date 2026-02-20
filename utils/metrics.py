import torch


def recall_at_k(logits, targets, k=10):
    top_k = torch.topk(logits, k=k, dim=1).indices
    hits = (top_k == targets.unsqueeze(1)).sum().float()
    return hits / targets.size(0)


def ndcg_at_k(logits, targets, k=10):
    top_k = torch.topk(logits, k=k, dim=1).indices
    hits = (top_k == targets.unsqueeze(1)).float()

    if hits.sum() == 0:
        return torch.tensor(0.0)

    rank = torch.where(hits == 1)[1] + 1
    return (1.0 / torch.log2(rank.float() + 1)).mean()