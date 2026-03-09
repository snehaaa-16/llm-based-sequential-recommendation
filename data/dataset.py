import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len=20):
        """
        user_sequences: dict {user_id: [item1, item2, ...]}
        """
        self.max_seq_len = max_seq_len
        self.samples = []

        for user, items in user_sequences.items():

            # Need at least 2 interactions
            if len(items) < 2:
                continue

            input_seq = items[:-1]
            target = items[-1]

            # Keep only the most recent interactions
            input_seq = input_seq[-max_seq_len:]

            self.samples.append((input_seq, target))

    def __len__(self):
        return len(self.samples)

    def _pad_sequence(self, sequence):
        """
        Left-pad sequence so the most recent items stay at the end.
        """
        padding_length = self.max_seq_len - len(sequence)

        padded_sequence = [0] * padding_length + sequence

        padding_mask = [True] * padding_length + [False] * len(sequence)

        return padded_sequence, padding_mask

    def __getitem__(self, idx):

        sequence, target = self.samples[idx]

        padded_sequence, padding_mask = self._pad_sequence(sequence)

        return (
            torch.tensor(padded_sequence, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(padding_mask, dtype=torch.bool)
        )
