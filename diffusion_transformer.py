import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenDiffusionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_iterations, max_seq_len, noise_std=0.1):
        super(TokenDiffusionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=1  # A single layer of transformer decoder
        )
        self.num_iterations = num_iterations
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.noise_std = noise_std

    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.size()

        # Add noise to the input tokens
        noisy_tokens = input_tokens + torch.randn_like(input_tokens) * self.noise_std

        x = self.embedding(noisy_tokens)  # Shape: (batch_size, seq_len, embedding_dim)
        memory = x

        for i in range(self.num_iterations):
            noise_std = self.noise_std * (1 - i / (self.num_iterations - 1))
            x = x + torch.randn_like(x) * noise_std

            x = self.transformer_decoder(tgt=x, memory=memory)
            logits = self.output_projection(x)  # Shape: (batch_size, seq_len, vocab_size)

        return logits