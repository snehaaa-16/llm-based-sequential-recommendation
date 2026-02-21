import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings
from models.hierarchical_model import HierarchicalLLMRec


def train():

    ratings, movies = load_ml1m()
    user_sequences = create_user_sequences(ratings)

    dataset = SequentialDataset(user_sequences)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    item_embeddings = build_item_embeddings()

    model = HierarchicalLLMRec(item_embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(2):
        total_loss = 0

        for sequences, targets in dataloader:

            logits = model(sequences)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()