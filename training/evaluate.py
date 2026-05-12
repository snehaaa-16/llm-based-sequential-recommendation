import time
import torch

from utils.metrics import compute_metrics


def evaluate_model(
    model,
    dataloader,
    device="cpu",
    ks=[10, 20],
    external_targets=None,
    use_amp=True
):
    """
    Evaluate model using Recall@K and NDCG@K
    """

    model.eval()

    total_metrics = {
        f"recall@{k}": 0.0 for k in ks
    }

    total_metrics.update({
        f"ndcg@{k}": 0.0 for k in ks
    })

    total_samples = 0
    global_idx = 0

    # Pre-convert external targets once
    if external_targets is not None:
        external_targets = torch.tensor(
            external_targets,
            dtype=torch.long
        )

    with torch.no_grad():

        for sequences, targets, padding_mask in dataloader:

            batch_size = sequences.size(0)

            sequences = sequences.to(device)
            padding_mask = padding_mask.to(device)

            # -----------------------
            # Mixed precision inference
            # -----------------------
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and use_amp)
            ):
                logits = model(sequences, padding_mask)

            # -----------------------
            # Targets
            # -----------------------
            if external_targets is not None:

                batch_targets = external_targets[
                    global_idx: global_idx + batch_size
                ].to(device)

                global_idx += batch_size

            else:
                batch_targets = targets.to(device)

            # MovieLens IDs start from 1
            batch_targets = batch_targets - 1

            metrics = compute_metrics(
                logits,
                batch_targets,
                ks
            )

            # Weighted accumulation
            for key in metrics:
                total_metrics[key] += metrics[key] * batch_size

            total_samples += batch_size

    # True dataset average
    for key in total_metrics:
        total_metrics[key] /= total_samples

    # -----------------------
    # Print results
    # -----------------------
    print("\nEvaluation Results")

    for key, value in total_metrics.items():
        print(f"{key}: {value:.4f}")

    return total_metrics


def measure_inference_time(
    model,
    dataloader,
    device="cpu",
    use_amp=True
):
    """
    Measure inference latency + throughput
    """

    model.eval()

    total_samples = 0

    # GPU timing sync
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():

        for sequences, _, padding_mask in dataloader:

            batch_size = sequences.size(0)

            sequences = sequences.to(device)
            padding_mask = padding_mask.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and use_amp)
            ):
                model(sequences, padding_mask)

            total_samples += batch_size

    # GPU timing sync
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start

    throughput = total_samples / total_time

    print("\nInference Performance")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} samples/sec")

    return {
        "total_time": total_time,
        "throughput": throughput
    }
