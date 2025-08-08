# src/data_utils/datasets.py
import torch

class SSDataset(torch.utils.data.Dataset):
    """
    A simple dataset wrapper for seismic data stored in a dictionary of tensors.
    Each key in the encodings dictionary corresponds to a data component (e.g., 'inputs_embeds', 'labels').
    """
    def __init__(self, encodings):
        self.encodings = encodings
        if not isinstance(self.encodings, dict) or 'inputs_embeds' not in self.encodings:
            raise ValueError("Encodings must be a dictionary containing at least 'inputs_embeds'.")

    def __getitem__(self, idx):
        # Ensure all tensors are cloned and detached to prevent in-place modification issues
        # and to ensure they are on the correct device if moved later.
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['inputs_embeds'])

# SSDataset is the default dataset for seismic data.
# Additional dataset classes should be defined separately if required.
