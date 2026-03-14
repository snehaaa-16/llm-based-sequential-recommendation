import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ItemLLM(nn.Module):
    def __init__(
        self,
        model_name="t5-small",
        max_length=64,
        freeze_encoder=True,
        device=None,
        output_dim=None
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.max_length = max_length

        # Load tokenizer + encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_dim = self.encoder.config.hidden_size

        # Optional projection layer
        if output_dim and output_dim != hidden_dim:
            self.projection = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = hidden_dim

        # Freeze encoder if needed
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.to(self.device)

    def masked_mean_pool(self, hidden_states, attention_mask):
        """
        Efficient masked mean pooling
        hidden_states: (B, L, D)
        attention_mask: (B, L)
        """

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(self, texts, batch_size=32):

        embeddings = []

        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda"
                ):
                    outputs = self.encoder(**inputs)

            pooled = self.masked_mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            pooled = self.projection(pooled)

            embeddings.append(pooled.cpu())

        return torch.cat(embeddings, dim=0)
