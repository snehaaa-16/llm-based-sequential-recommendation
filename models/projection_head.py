import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        item_embeddings,
        temperature=1.0,
        learnable_temperature=False,
        normalize_items=True
    ):
        super().__init__()

        # Register item embeddings as buffer
        self.register_buffer("item_embeddings", item_embeddings)

        self.normalize_items = normalize_items

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))

        # Optional cache for normalized embeddings
        if normalize_items:
            self.register_buffer(
                "normalized_items",
                F.normalize(item_embeddings, dim=1)
            )
        else:
            self.normalized_items = item_embeddings

    def forward(self, user_representation):
        """
        user_representation: (B, D)
        returns logits: (B, num_items)
        """

        # Normalize user embeddings
        user_representation = F.normalize(user_representation, dim=1)

        # Use cached normalized embeddings if enabled
        item_embeddings = (
            self.normalized_items if self.normalize_items
            else self.item_embeddings
        )

        logits = torch.matmul(user_representation, item_embeddings.T)

        logits = logits / self.temperature.clamp(min=1e-6)

        return logits
