import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) *
            (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RecommendationLLM(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        max_seq_len=50
    ):
        super().__init__()

        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_embeddings, padding_mask=None):
        """
        sequence_embeddings: (B, L, D)
        padding_mask: (B, L) where True indicates padding
        """

        x = self.positional_encoding(sequence_embeddings)

        outputs = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        outputs = self.layer_norm(outputs)

        # mean pooling
        user_representation = outputs.mean(dim=1)

        return user_representation
