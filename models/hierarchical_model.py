import torch
import torch.nn as nn
import torch.nn.functional as F

from .recommendation_llm import RecommendationLLM
from .projection_head import ProjectionHead


class HierarchicalLLMRec(nn.Module):
    def __init__(
        self,
        item_embeddings,
        hidden_dim=512,
        dropout=0.1,
        pooling="last",
        causal=False
    ):
        super().__init__()

        # Register item embeddings (not trainable)
        self.register_buffer("item_embeddings", item_embeddings)

        self.hidden_dim = hidden_dim

        self.embedding_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Sequence model (configurable)
        self.rec_llm = RecommendationLLM(
            hidden_dim=hidden_dim,
            pooling=pooling,
            causal=causal
        )

        # Ranking head
        self.projection_head = ProjectionHead(item_embeddings)

    def forward(self, sequence_ids, padding_mask=None):
        """
        sequence_ids: (B, L)
        padding_mask: (B, L) where True = padding
        """

        # -----------------------
        # ID → index conversion
        # -----------------------
        seq_ids = (sequence_ids - 1).clamp(min=0)

        # -----------------------
        # Embedding lookup
        # -----------------------
        sequence_embeddings = self.item_embeddings[seq_ids]

        # -----------------------
        # Zero-out padding embeddings (IMPORTANT)
        # -----------------------
        if padding_mask is not None:
            sequence_embeddings = sequence_embeddings.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0
            )

        # -----------------------
        # Regularization
        # -----------------------
        sequence_embeddings = self.embedding_dropout(sequence_embeddings)

        # -----------------------
        # Sequence modeling
        # -----------------------
        user_representation = self.rec_llm(
            sequence_embeddings,
            padding_mask
        )

        # -----------------------
        # Normalize before similarity
        # -----------------------
        user_representation = self.layer_norm(user_representation)
        user_representation = F.normalize(user_representation, dim=1)

        # -----------------------
        # Ranking
        # -----------------------
        logits = self.projection_head(user_representation)

        return logits
