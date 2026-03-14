import torch
import torch.nn as nn

from .recommendation_llm import RecommendationLLM
from .projection_head import ProjectionHead


class HierarchicalLLMRec(nn.Module):
    def __init__(
        self,
        item_embeddings,
        hidden_dim=512,
        dropout=0.1
    ):
        super().__init__()

        # Register item embeddings as buffer (not trainable)
        self.register_buffer("item_embeddings", item_embeddings)

        self.hidden_dim = hidden_dim
        self.embedding_dropout = nn.Dropout(dropout)

        # Sequence model
        self.rec_llm = RecommendationLLM(hidden_dim)

        # Ranking head
        self.projection_head = ProjectionHead(item_embeddings)

    def forward(self, sequence_ids, padding_mask=None):
        """
        sequence_ids: (B, L)
        padding_mask: (B, L) optional
        """

        # Convert MovieLens IDs (1..N) → embedding indices (0..N-1)
        seq_ids = sequence_ids - 1

        # Clamp negatives caused by padding (0 -> -1)
        seq_ids = seq_ids.clamp(min=0)

        # Efficient embedding lookup
        sequence_embeddings = self.item_embeddings[seq_ids]

        # Apply embedding dropout (regularization)
        sequence_embeddings = self.embedding_dropout(sequence_embeddings)

        # Sequence modeling
        user_representation = self.rec_llm(
            sequence_embeddings,
            padding_mask
        )

        # Ranking scores
        logits = self.projection_head(user_representation)

        return logits
