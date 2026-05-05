import torch
import faiss
import numpy as np
import torch.nn.functional as F


class Retriever:
    def __init__(
        self,
        item_embeddings,
        top_k=5,
        similarity="cosine",
        use_gpu=False,
        index_type="flat"  # "flat" or "ivf"
    ):
        """
        item_embeddings: (num_items, hidden_dim)
        """

        self.top_k = top_k
        self.similarity = similarity
        self.use_gpu = use_gpu
        self.index_type = index_type

        # Convert to CPU numpy float32 (required by FAISS)
        embeddings = item_embeddings.detach().cpu().numpy().astype(np.float32)

        # Normalize if cosine
        if similarity == "cosine":
            embeddings = self._normalize_np(embeddings)

        self.dim = embeddings.shape[1]

        # Build FAISS index
        self.index = self._build_index(embeddings)

    # -----------------------
    # Normalization (numpy)
    # -----------------------
    def _normalize_np(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norm

    # -----------------------
    # Build Index
    # -----------------------
    def _build_index(self, embeddings):

        if self.similarity == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            metric = faiss.METRIC_L2

        # Choose index type
        if self.index_type == "ivf":
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, metric)

            index.train(embeddings)
        else:
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(self.dim)
            else:
                index = faiss.IndexFlatL2(self.dim)

        index.add(embeddings)

        # Optional GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    # -----------------------
    # Retrieve
    # -----------------------
    def retrieve(self, query_embeddings):
        """
        query_embeddings: (B, hidden_dim)
        returns: (B, top_k)
        """

        queries = query_embeddings.detach().cpu().numpy().astype(np.float32)

        # Normalize if cosine
        if self.similarity == "cosine":
            queries = self._normalize_np(queries)

        distances, indices = self.index.search(queries, self.top_k)

        return torch.from_numpy(indices).long()

    # -----------------------
    # Optional: rebuild index
    # -----------------------
    def rebuild(self, new_item_embeddings):
        """
        Update FAISS index with new embeddings
        """
        embeddings = new_item_embeddings.detach().cpu().numpy().astype(np.float32)

        if self.similarity == "cosine":
            embeddings = self._normalize_np(embeddings)

        self.index.reset()
        self.index.add(embeddings)
