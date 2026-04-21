import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len=20):
        self.max_seq_len = max_seq_len
        self.samples = []

        for user, items in user_sequences.items():

            if len(items) < 2:
                continue

            input_seq = items[:-1][-max_seq_len:]
            target = items[-1]

            padded_seq, padding_mask = self._pad_sequence(input_seq)

            self.samples.append((
                torch.tensor(padded_seq, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(padding_mask, dtype=torch.bool)
            ))

    def __len__(self):
        return len(self.samples)

    def _pad_sequence(self, sequence):
        padding_length = self.max_seq_len - len(sequence)

        padded_sequence = [0] * padding_length + sequence
        padding_mask = [True] * padding_length + [False] * len(sequence)

        return padded_sequence, padding_mask

    def __getitem__(self, idx):
        return self.samples[idx]
