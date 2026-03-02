import yaml
import torch


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CONFIG = {
    "training": {
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 1e-3
    },
    "model": {
        "hidden_dim": 512,
        "top_k": 10
    },
    "retrieval": {
        "top_k": 5
    }
}
