import yaml
import torch
import random
import numpy as np
from copy import deepcopy


def load_config(config_path="configs/config.yaml"):
    """
    Loads YAML config and merges with defaults
    """
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    config = merge_dicts(DEFAULT_CONFIG, user_config)
    return config


def merge_dicts(default, override):
    """
    Recursively merge two dictionaries
    """
    result = deepcopy(default)

    for k, v in override.items():
        if k in result and isinstance(result[k], dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v

    return result


def get_device(config=None):
    """
    Returns correct device based on config
    """
    if config and config.get("device", {}).get("use_cuda", True):
        if torch.cuda.is_available():
            return torch.device("cuda")

    return torch.device("cpu")


def set_seed(seed=42):
    """
    Ensures reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Default Config
# -------------------------
DEFAULT_CONFIG = {
    "seed": 42,

    "training": {
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "num_workers": 0
    },

    "model": {
        "hidden_dim": 512,
        "max_seq_len": 20,
        "dropout": 0.1,
        "pooling": "last",
        "top_k": 10
    },

    "retrieval": {
        "top_k": 5,
        "strategy": "attention"  # mean / attention
    },

    "item_llm": {
        "model_name": "t5-small",
        "max_length": 64,
        "freeze_encoder": True,
        "embedding_cache": "data/item_embeddings.pt"
    },

    "dataset": {
        "data_path": "data/ml-1m"
    },

    "logging": {
        "log_interval": 1,
        "save_checkpoint": False
    },

    "device": {
        "use_cuda": True
    }
}
