import torch
import torch.nn as nn

from .recommendation_llm import RecommendationLLM
from .projection_head import ProjectionHead


class HierarchicalLLMRec(nn.Module):
    def __init__(self, item_embeddings, hidden_dim=512):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Register item embeddings as non-trainable buffer
        self.register_buffer("item_embeddings", item_embeddings)

        # Sequence model
        self.rec_llm = RecommendationLLM(hidden_dim)

        # Ranking head
        self.projection_head = ProjectionHead(item_embeddings)

    def forward(self, sequence_ids, padding_mask=None):
        """
        sequence_ids: (B, L)
        padding_mask: (B, L) optional
        """

        device = sequence_ids.device

        # Convert item IDs → embedding indices
        seq_ids = sequence_ids.clone()

        mask = seq_ids == 0
        seq_ids = seq_ids - 1
        seq_ids[mask] = 0

        # Vectorized embedding lookup
        sequence_embeddings = self.item_embeddings[seq_ids]

        # Zero-out padding tokens
        sequence_embeddings[mask] = 0.0

        # Transformer sequence modeling
        user_representation = self.rec_llm(
            sequence_embeddings,
            padding_mask
        )

        # Final ranking
        logits = self.projection_head(user_representation)

        return logits
