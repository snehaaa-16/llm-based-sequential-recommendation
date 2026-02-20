import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from models.hierarchical_model import HierarchicalLLMRec


def train():

    ratings, movies = load_ml1m()
    user_sequences = create_user_sequences(ratings)

    dataset = SequentialDataset(user_sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_items = ratings["movie_id"].nunique() + 1

    model = HierarchicalLLMRec(num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(3):
        total_loss = 0

        for sequences, targets in dataloader:

            # Fake embedding for now
            sequences = sequences.float().unsqueeze(-1).repeat(1, 1, 512)

            logits = model(sequences)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()