import torch
import faiss


class Retriever:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings
        self.index = self.build_index()

    def build_index(self):
        dim = self.item_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.item_embeddings.numpy())
        return index

    def retrieve(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding.numpy(), k)
        return indices