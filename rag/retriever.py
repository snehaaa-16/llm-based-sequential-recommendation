import torch
import faiss
import torch.nn.functional as F


class Retriever:
    def __init__(self, item_embeddings, top_k=5, similarity="l2"):
        """
        item_embeddings: tensor (num_items, hidden_dim)
        """
        self.top_k = top_k
        self.similarity = similarity

        # Ensure embeddings are on CPU for FAISS
        self.item_embeddings = item_embeddings.detach().cpu()

        # Optional normalization for cosine similarity
        if similarity == "cosine":
            self.item_embeddings = F.normalize(self.item_embeddings, dim=1)

        self.index = self._build_index()

    def _build_index(self):
        dim = self.item_embeddings.shape[1]

        if self.similarity == "cosine":
            index = faiss.IndexFlatIP(dim)  # inner product
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(self.item_embeddings.numpy())
        return index

    def retrieve(self, query_embeddings):
        """
        query_embeddings: (B, hidden_dim)
        returns: indices (B, top_k)
        """

        queries = query_embeddings.detach().cpu()

        if self.similarity == "cosine":
            queries = F.normalize(queries, dim=1)

        queries = queries.numpy()

        _, indices = self.index.search(queries, self.top_k)

        return torch.tensor(indices, dtype=torch.long)
