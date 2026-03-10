import torch
import torch.nn as nn
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

        self.device = device if device else torch.device(
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
            self.projection = None
            self.output_dim = hidden_dim

        # Freeze encoder if required
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.to(self.device)

    def masked_mean_pooling(self, hidden_states, attention_mask):
        """
        hidden_states: (B, L, D)
        attention_mask: (B, L)
        """
        mask = attention_mask.unsqueeze(-1)
        masked_embeddings = hidden_states * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, texts, batch_size=32):
        """
        texts: list[str]
        Returns: (num_items, embedding_dim)
        """

        all_embeddings = []

        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():

                # Automatic mixed precision if GPU
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.encoder(**inputs)

            hidden_states = outputs.last_hidden_state

            pooled = self.masked_mean_pooling(
                hidden_states,
                inputs["attention_mask"]
            )

            if self.projection is not None:
                pooled = self.projection(pooled)

            all_embeddings.append(pooled.detach().cpu())

        return torch.cat(all_embeddings, dim=0)
