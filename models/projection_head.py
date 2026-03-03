import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, item_embeddings, temperature=1.0):
        super().__init__()

        # Register as buffer (not trainable)
        self.register_buffer("item_embeddings", item_embeddings)

        self.temperature = temperature

    def forward(self, user_representation):

        # Normalize for stable similarity
        user_representation = F.normalize(user_representation, dim=1)
        item_embeddings = F.normalize(self.item_embeddings, dim=1)

        logits = torch.matmul(user_representation, item_embeddings.T)

        logits = logits / self.temperature

        return logits
