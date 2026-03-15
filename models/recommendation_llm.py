import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) *
            (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )

        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RecommendationLLM(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        max_seq_len=50,
        pooling="last"
    ):
        super().__init__()

        self.pooling = pooling

        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        if pooling == "attention":
            self.attention_pool = nn.Linear(hidden_dim, 1)

    def attention_pooling(self, outputs, padding_mask):

        scores = self.attention_pool(outputs).squeeze(-1)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)

        weights = F.softmax(scores, dim=1)

        pooled = torch.sum(outputs * weights.unsqueeze(-1), dim=1)

        return pooled

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

        if self.pooling == "last":
            user_representation = outputs[:, -1]

        elif self.pooling == "mean":
            user_representation = outputs.mean(dim=1)

        elif self.pooling == "attention":
            user_representation = self.attention_pooling(
                outputs,
                padding_mask
            )

        else:
            raise ValueError("Invalid pooling type")

        return user_representation
