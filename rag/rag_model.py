import torch
import torch.nn as nn
import torch.nn.functional as F


class RAGSequentialRec(nn.Module):
    def __init__(
        self,
        base_model,
        retriever,
        item_embeddings,
        hidden_dim=512,
        dropout=0.1,
        retrieval_fusion="attention"
    ):
        super().__init__()

        self.base_model = base_model
        self.retriever = retriever
        self.retrieval_fusion = retrieval_fusion

        # Register item embeddings
        self.register_buffer("item_embeddings", item_embeddings)

        # Fusion components
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Attention fusion
        if retrieval_fusion == "attention":
            self.retrieval_attention = nn.Linear(hidden_dim, 1)

    # -----------------------
    # Retrieval aggregation
    # -----------------------
    def aggregate_retrieval(self, retrieved_embeds):

        # Normalize embeddings (important for similarity)
        retrieved_embeds = F.normalize(retrieved_embeds, dim=-1)

        if self.retrieval_fusion == "mean":
            return retrieved_embeds.mean(dim=1)

        if self.retrieval_fusion == "attention":
            scores = self.retrieval_attention(retrieved_embeds).squeeze(-1)
            weights = F.softmax(scores, dim=1)
            return torch.sum(retrieved_embeds * weights.unsqueeze(-1), dim=1)

        raise ValueError("Invalid retrieval fusion strategy")

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, sequence_ids):

        device = sequence_ids.device

        # -----------------------
        # Convert IDs → indices
        # -----------------------
        seq_ids = (sequence_ids - 1).clamp(min=0)
        padding_mask = sequence_ids == 0

        # -----------------------
        # Embedding lookup
        # -----------------------
        sequence_embeddings = self.item_embeddings[seq_ids]

        # FIX: zero-out padding embeddings
        sequence_embeddings = sequence_embeddings.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0
        )

        # -----------------------
        # Base user representation
        # -----------------------
        user_rep = self.base_model.rec_llm(
            sequence_embeddings,
            padding_mask
        )

        # Normalize for retrieval
        user_rep = F.normalize(user_rep, dim=1)

        # -----------------------
        # Retrieval
        # -----------------------
        retrieved_indices = self.retriever.retrieve(user_rep)
        retrieved_indices = retrieved_indices.to(device)

        retrieved_embeds = self.item_embeddings[retrieved_indices]

        # -----------------------
        # Aggregate retrieved items
        # -----------------------
        retrieved_rep = self.aggregate_retrieval(retrieved_embeds)

        # -----------------------
        # Fusion
        # -----------------------
        fusion_input = torch.cat([user_rep, retrieved_rep], dim=-1)

        gate = torch.sigmoid(self.fusion_gate(fusion_input))

        fused_rep = gate * user_rep + (1 - gate) * retrieved_rep

        fused_rep = self.layer_norm(self.dropout(fused_rep))

        # Normalize before ranking
        fused_rep = F.normalize(fused_rep, dim=1)

        # -----------------------
        # Ranking
        # -----------------------
        logits = self.base_model.projection_head(fused_rep)

        return logits
