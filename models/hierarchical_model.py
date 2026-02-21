import torch
import torch.nn as nn
from .recommendation_llm import RecommendationLLM
from .projection_head import ProjectionHead


class HierarchicalLLMRec(nn.Module):
    def __init__(self, item_embeddings, hidden_dim=512):
        super().__init__()

        self.item_embeddings = item_embeddings
        self.hidden_dim = hidden_dim

        self.rec_llm = RecommendationLLM(hidden_dim)
        self.projection_head = ProjectionHead(hidden_dim, len(item_embeddings))

    def forward(self, sequence_ids):

        batch_size, seq_len = sequence_ids.shape

        sequence_embeddings = []

        for i in range(batch_size):
            seq_embed = []
            for item_id in sequence_ids[i]:
                item_id = item_id.item()

                if item_id in self.item_embeddings:
                    seq_embed.append(self.item_embeddings[item_id])
                else:
                    seq_embed.append(torch.zeros(self.hidden_dim))

            seq_embed = torch.stack(seq_embed)
            sequence_embeddings.append(seq_embed)

        sequence_embeddings = torch.stack(sequence_embeddings)

        user_representation = self.rec_llm(sequence_embeddings)
        logits = self.projection_head(user_representation)

        return logits