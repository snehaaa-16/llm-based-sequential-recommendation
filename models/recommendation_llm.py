import torch
import torch.nn as nn


class RecommendationLLM(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, sequence_embeddings):
        outputs = self.transformer(sequence_embeddings)
        return outputs.mean(dim=1)