import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class ItemLLM(nn.Module):
    def __init__(
        self,
        model_name="t5-small",
        max_length=64,
        freeze_encoder=True,
        device=None
    ):
        super().__init__()

        self.device = device if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.max_length = max_length

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.to(self.device)

    def forward(self, texts, batch_size=32):
        """
        texts: list of strings
        Returns: (num_items, hidden_dim)
        """

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.encoder(**inputs)

            hidden_states = outputs.last_hidden_state  # (B, L, D)
            attention_mask = inputs["attention_mask"]  # (B, L)

            # Masked mean pooling (ignore padding tokens)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts

            all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0)
