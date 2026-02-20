import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len=20):
        self.samples = []

        for user, items in user_sequences.items():
            if len(items) < 3:
                continue

            input_seq = items[:-1][-max_seq_len:]
            target = items[-1]

            self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, target = self.samples[idx]

        return torch.tensor(sequence), torch.tensor(target)