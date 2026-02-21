import torch
from models.item_llm import ItemLLM
from data.preprocess import load_ml1m


def build_item_embeddings(data_path="data/ml-1m"):
    ratings, movies = load_ml1m(data_path)

    model = ItemLLM()
    model.eval()

    movie_texts = (movies["title"] + " " + movies["genres"]).tolist()
    movie_ids = movies["movie_id"].tolist()

    with torch.no_grad():
        embeddings = model(movie_texts)

    item_embedding_dict = {
        movie_id: embeddings[i]
        for i, movie_id in enumerate(movie_ids)
    }

    return item_embedding_dict