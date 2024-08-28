import torch
from torch.utils.data import Dataset, DataLoader
import random

class TinyShakespeareDataset(Dataset):
    def __init__(self, path, seq_len, noise_type='replacement', noise_rate=0):
        # Read and preprocess the text
        with open(path, 'r') as f:
            raw_text = f.read().lower()
        
        self.seq_len = seq_len
        chars = sorted(list(set(raw_text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.text = [self.char_to_idx[c] for c in raw_text]
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        
    def __len__(self):
        return len(self.text) - self.seq_len
    
    def apply_noise(self, sequence):
        noisy_sequence = sequence.clone()

        if self.noise_type == 'replacement':
            # Replace some tokens with random ones
            for i in range(len(noisy_sequence)):
                if random.random() < self.noise_rate:
                    noisy_sequence[i] = random.randint(0, self.vocab_size - 1)
        
        elif self.noise_type == 'dropout':
            # Replace some tokens with a special "unknown" token
            for i in range(len(noisy_sequence)):
                if random.random() < self.noise_rate:
                    noisy_sequence[i] = self.char_to_idx.get('[UNK]', 0)
        
        elif self.noise_type == 'shuffle':
            # Shuffle tokens within a small window
            window_size = int(len(noisy_sequence) * self.noise_rate)
            for i in range(0, len(noisy_sequence) - window_size, window_size):
                window = noisy_sequence[i:i + window_size]
                random.shuffle(window)
                noisy_sequence[i:i + window_size] = window
        
        return noisy_sequence
    
    def __getitem__(self, idx):
        # Get a clean sequence from the text
        clean_seq = self.text[idx:idx + self.seq_len]
        
        # Convert to tensor
        clean_seq = torch.tensor(clean_seq)
        
        # Generate a noisy version of the sequence
        # noisy_seq = self.apply_noise(clean_seq)
        
        #for now lets return the following seq_len characters
        noisy_seq = self.text[idx + self.seq_len : idx + 2*self.seq_len]
        noisy_seq = torch.tensor(noisy_seq)
        return noisy_seq, clean_seq
