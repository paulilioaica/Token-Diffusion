import torch
from torch.utils.data import Dataset, DataLoader

class TinyShakespeareDataset(Dataset):
    def __init__(self, path, seq_len):
        raw_text = None
        with open(path, 'r') as f:
            raw_text = f.read()

        self.seq_len = seq_len
        chars = sorted(list(set(raw_text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.text = [self.char_to_idx[c] for c in raw_text]
        
    def __len__(self):
        return len(self.text) - self.seq_len
    
    def __getitem__(self, idx):
        input_seq = self.text[idx:idx + self.seq_len]
        target_seq = self.text[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

