import torch
import os
from models.item_llm import ItemLLM
from data.preprocess import load_ml1m


def build_item_embeddings(data_path="data/ml-1m", save_path="data/item_embeddings.pt"):

    # If already computed, load from disk
    if os.path.exists(save_path):
        print("Loading precomputed item embeddings...")
        return torch.load(save_path)

    print("Building item embeddings from scratch...")

    ratings, movies = load_ml1m(data_path)

    model = ItemLLM()
    model.eval()

    movie_texts = (movies["title"] + " " + movies["genres"]).tolist()
    movie_ids = movies["movie_id"].tolist()

    with torch.no_grad():
        embeddings = model(movie_texts)

    item_embedding_tensor = embeddings.cpu()
    torch.save(item_embedding_tensor, save_path)

    print("Item embeddings saved.")

    return item_embedding_tensor
