import time
import torch


def measure_inference_time(model, dataloader, device="cpu"):
    model.eval()
    start = time.time()

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            model(sequences)

    end = time.time()
    print(f"Inference Time: {end - start:.4f} seconds")
