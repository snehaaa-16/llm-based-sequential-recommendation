import torch
import os

from models.item_llm import ItemLLM
from data.preprocess import load_ml1m
from utils.config import load_config, get_device


def build_item_embeddings():

    config = load_config()
    device = get_device(config)

    data_path = config["dataset"]["data_path"]
    save_path = config["item_llm"]["embedding_cache"]

    # -------------------------
    # Load cached embeddings
    # -------------------------
    if os.path.exists(save_path):
        print("Loading cached item embeddings...")
        return torch.load(save_path)

    print("Building item embeddings...")

    # -------------------------
    # Load dataset
    # -------------------------
    _, movies = load_ml1m(data_path)

    movie_texts = (
        movies["title"] + " " + movies["genres"]
    ).tolist()

    # -------------------------
    # Initialize ItemLLM
    # -------------------------
    model = ItemLLM(
        model_name=config["item_llm"]["model_name"],
        max_length=config["item_llm"]["max_length"],
        freeze_encoder=config["item_llm"]["freeze_encoder"],
        device=device
    )

    model.eval()

    # -------------------------
    # Compute embeddings
    # -------------------------
    with torch.no_grad():
        embeddings = model(movie_texts)

    embeddings = embeddings.cpu()

    # -------------------------
    # Save cache
    # -------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)

    print(f"Saved item embeddings to {save_path}")

    return embeddings
