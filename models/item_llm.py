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
        output_dim=None,
        normalize=True
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.max_length = max_length
        self.normalize = normalize

        # Load tokenizer + encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_dim = self.encoder.config.hidden_size

        # Optional projection
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
        self.encoder.eval()

    # -----------------------
    # Masked Mean Pooling
    # -----------------------
    def masked_mean_pool(self, hidden_states, attention_mask):

        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, texts, batch_size=32):

        all_embeddings = []

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=(self.device.type == "cuda")
                ):
                    outputs = self.encoder(**inputs)

            hidden_states = outputs.last_hidden_state

            pooled = self.masked_mean_pool(
                hidden_states,
                inputs["attention_mask"]
            )

            pooled = self.projection(pooled)

            # Optional normalization (important for retrieval similarity)
            if self.normalize:
                pooled = F.normalize(pooled, dim=1)

            all_embeddings.append(pooled.detach().cpu())

        return torch.cat(all_embeddings, dim=0)
