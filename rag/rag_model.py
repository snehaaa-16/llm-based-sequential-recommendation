import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(self, base_model, retriever, item_embeddings, hidden_dim=512):
        super().__init__()

        self.base_model = base_model
        self.retriever = retriever
        self.hidden_dim = hidden_dim

        # Register item embeddings as buffer (not trainable)
        self.register_buffer("item_embeddings", item_embeddings)

        # Gated fusion
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence_ids):

        device = sequence_ids.device

        # ---------------------------------------
        # 1️⃣ Convert item IDs → embeddings (vectorized)
        # ---------------------------------------

        # Padding assumed as 0
        # Movie IDs assumed starting from 1
        seq_ids = sequence_ids.clone()
        mask = seq_ids == 0
        seq_ids = seq_ids - 1
        seq_ids[mask] = 0  # avoid negative index

        sequence_embeddings = self.item_embeddings[seq_ids]
        sequence_embeddings[mask] = 0.0

        # ---------------------------------------
        # 2️⃣ Baseline user representation
        # ---------------------------------------

        user_rep = self.base_model.rec_llm(sequence_embeddings)

        # ---------------------------------------
        # 3️⃣ Retrieval
        # ---------------------------------------

        indices = self.retriever.retrieve(user_rep)
        indices = indices.to(device)

        # Vectorized retrieval embedding lookup
        retrieved_embeds = self.item_embeddings[indices]  # (B, K, D)

        # Aggregate retrieved embeddings (mean over K)
        retrieved_embeds = retrieved_embeds.mean(dim=1)

        # ---------------------------------------
        # 4️⃣ Gated Fusion
        # ---------------------------------------

        concat = torch.cat([user_rep, retrieved_embeds], dim=1)
        gate = self.sigmoid(self.gate_layer(concat))

        fused_rep = gate * user_rep + (1 - gate) * retrieved_embeds

        # ---------------------------------------
        # 5️⃣ Final Ranking
        # ---------------------------------------

        logits = self.base_model.projection_head(fused_rep)

        return logits
