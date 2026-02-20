import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class ItemLLM(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state

        return hidden_states.mean(dim=1)