import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, num_items):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_items)

    def forward(self, user_representation):
        return self.linear(user_representation)