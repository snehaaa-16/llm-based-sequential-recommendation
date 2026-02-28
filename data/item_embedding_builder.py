import torch
import os
from models.item_llm import ItemLLM
from data.preprocess import load_ml1m


def build_item_embeddings(data_path="data/ml-1m",
                          save_path="data/item_embeddings.pt"):

    if os.path.exists(save_path):
        print("Loading cached item embeddings...")
        return torch.load(save_path)

    print("Building item embeddings...")

    _, movies = load_ml1m(data_path)

    model = ItemLLM()
    model.eval()

    movie_texts = (movies["title"] + " " + movies["genres"]).tolist()

    with torch.no_grad():
        embeddings = model(movie_texts).cpu()

    torch.save(embeddings, save_path)
    print("Saved item embeddings.")

    return embeddings
