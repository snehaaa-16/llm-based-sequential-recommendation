import torch
import torch.nn as nn


class RAGSequentialRec(nn.Module):
    def __init__(self, base_model, retriever):
        super().__init__()
        self.base_model = base_model
        self.retriever = retriever

    def forward(self, sequence_embeddings):
        retrieved = self.retriever.retrieve(sequence_embeddings.mean(dim=1))
        # Combine retrieved embeddings
        logits = self.base_model(sequence_embeddings)
        return logits