import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocess import load_ml1m, create_user_sequences
from data.dataset import SequentialDataset
from data.item_embedding_builder import build_item_embeddings

from models.hierarchical_model import HierarchicalLLMRec
from rag.retriever import Retriever
from rag.rag_model import RAGSequentialRec


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load Dataset
    # -----------------------
    ratings, movies = load_ml1m()
    user_sequences = create_user_sequences(ratings)

    dataset = SequentialDataset(user_sequences)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -----------------------
    # Build Item Embeddings
    # -----------------------
    item_embeddings = build_item_embeddings()
    item_embeddings = item_embeddings.to(device)

    # -----------------------
    # Build Baseline Model
    # -----------------------
    num_items = item_embeddings.shape[0]
    hidden_dim = item_embeddings.shape[1]

    base_model = HierarchicalLLMRec(num_items, hidden_dim)
    base_model = base_model.to(device)

    # -----------------------
    # Build Retriever
    # -----------------------
    retriever = Retriever(item_embeddings.cpu())

    # -----------------------
    # Build RAG Model
    # -----------------------
    model = RAGSequentialRec(
        base_model=base_model,
        retriever=retriever,
        item_embeddings=item_embeddings,
        hidden_dim=hidden_dim
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training Loop
    # -----------------------
    model.train()

    for epoch in range(2):
        total_loss = 0

        for sequences, targets in dataloader:

            sequences = sequences.to(device)
            targets = targets.to(device)

            logits = model(sequences)

            # IMPORTANT:
            # MovieLens IDs start from 1, so adjust target index
            targets = targets - 1

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
