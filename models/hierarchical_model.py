import torch.nn as nn
from .recommendation_llm import RecommendationLLM
from .projection_head import ProjectionHead


class HierarchicalLLMRec(nn.Module):
    def __init__(self, num_items, hidden_dim=512):
        super().__init__()

        self.rec_llm = RecommendationLLM(hidden_dim)
        self.projection_head = ProjectionHead(hidden_dim, num_items)

    def forward(self, sequence_embeddings):
        user_representation = self.rec_llm(sequence_embeddings)
        logits = self.projection_head(user_representation)
        return logits