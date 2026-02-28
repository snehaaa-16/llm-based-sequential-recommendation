import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(self, base_model, retriever, item_embeddings, hidden_dim=512):
        super().__init__()

        self.base_model = base_model
        self.retriever = retriever
        self.item_embeddings = item_embeddings

        # Gated fusion mechanism
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence_embeddings):

        # Baseline user representation
        user_rep = self.base_model.rec_llm(sequence_embeddings)

        # Retrieve similar items
        indices = self.retriever.retrieve(user_rep)

        retrieved_embeds = []
        for batch_idx in range(indices.shape[0]):
            items = indices[batch_idx]
            emb = self.item_embeddings[items]
            emb = emb.mean(dim=0)
            retrieved_embeds.append(emb)

        retrieved_embeds = torch.stack(retrieved_embeds)

        # Concatenate
        concat = torch.cat([user_rep, retrieved_embeds], dim=1)

        # Compute gating values
        gate = self.sigmoid(self.gate_layer(concat))

        # Gated fusion
        fused_rep = gate * user_rep + (1 - gate) * retrieved_embeds

        logits = self.base_model.projection_head(fused_rep)

        return logits
