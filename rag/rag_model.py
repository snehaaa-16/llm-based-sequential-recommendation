import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(self, base_model, retriever, item_embeddings):
        super().__init__()
        self.base_model = base_model
        self.retriever = retriever
        self.item_embeddings = item_embeddings

    def forward(self, sequence_embeddings):

        # Get baseline user representation
        user_rep = self.base_model.rec_llm(sequence_embeddings)

        # Retrieve similar items
        indices = self.retriever.retrieve(user_rep)

        # Get retrieved embeddings
        retrieved_embeds = []
        for batch_idx in range(indices.shape[0]):
            items = indices[batch_idx]
            emb = self.item_embeddings[items]
            emb = emb.mean(dim=0)  # simple mean fusion
            retrieved_embeds.append(emb)

        retrieved_embeds = torch.stack(retrieved_embeds)

        # Fuse baseline + retrieved
        fused_rep = (user_rep + retrieved_embeds) / 2

        logits = self.base_model.projection_head(fused_rep)

        return logits
