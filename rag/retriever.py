import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(self, base_model, retriever, item_embeddings, hidden_dim=512):
        super().__init__()

        self.base_model = base_model
        self.retriever = retriever
        self.item_embeddings = item_embeddings

        # Learnable fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, sequence_embeddings):

        # Baseline user representation
        user_rep = self.base_model.rec_llm(sequence_embeddings)

        # Retrieve top-K items
        indices = self.retriever.retrieve(user_rep)

        retrieved_embeds = []
        for batch_idx in range(indices.shape[0]):
            items = indices[batch_idx]
            emb = self.item_embeddings[items]
            emb = emb.mean(dim=0)  # aggregate retrieved items
            retrieved_embeds.append(emb)

        retrieved_embeds = torch.stack(retrieved_embeds)

        # Concatenate user representation + retrieved representation
        fused_input = torch.cat([user_rep, retrieved_embeds], dim=1)

        # Learnable fusion
        fused_rep = self.fusion(fused_input)

        # Final ranking
        logits = self.base_model.projection_head(fused_rep)

        return logits
