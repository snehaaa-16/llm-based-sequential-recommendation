import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(
        self,
        base_model,
        retriever,
        item_embeddings,
        hidden_dim=512,
        dropout=0.1
    ):
        super().__init__()

        self.base_model = base_model
        self.retriever = retriever
        self.hidden_dim = hidden_dim

        # Register item embeddings as non-trainable buffer
        self.register_buffer("item_embeddings", item_embeddings)

        # Fusion layers
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence_ids):

        device = sequence_ids.device

        # ---------------------------------
        # Convert item IDs → embeddings
        # ---------------------------------

        seq_ids = sequence_ids.clone()

        padding_mask = seq_ids == 0
        seq_ids = seq_ids - 1
        seq_ids[padding_mask] = 0

        sequence_embeddings = self.item_embeddings[seq_ids]
        sequence_embeddings[padding_mask] = 0.0

        # ---------------------------------
        # User representation (baseline)
        # ---------------------------------

        user_rep = self.base_model.rec_llm(
            sequence_embeddings,
            padding_mask
        )

        # ---------------------------------
        # Retrieval step
        # ---------------------------------

        retrieved_indices = self.retriever.retrieve(user_rep)
        retrieved_indices = retrieved_indices.to(device)

        retrieved_embeds = self.item_embeddings[retrieved_indices]

        # Aggregate retrieved items
        retrieved_embeds = retrieved_embeds.mean(dim=1)

        # ---------------------------------
        # Fusion
        # ---------------------------------

        fusion_input = torch.cat([user_rep, retrieved_embeds], dim=1)

        gate_values = self.gate(self.fusion_layer(fusion_input))

        fused_rep = gate_values * user_rep + (1 - gate_values) * retrieved_embeds

        fused_rep = self.layer_norm(self.dropout(fused_rep))

        # ---------------------------------
        # Final ranking
        # ---------------------------------

        logits = self.base_model.projection_head(fused_rep)

        return logits
