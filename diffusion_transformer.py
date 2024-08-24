import torch
import torch.nn as nn


class TokenDiffusionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, nhead, max_seq_len, num_iterations, noise_std=0.1, dropout=0.1):
        super(TokenDiffusionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        self.noise_std = noise_std

    def forward(self, input_tokens, tgt_mask=None):
        batch_size, seq_len = input_tokens.size()

        # Embedding the input tokens
        token_embeddings = self.embedding(input_tokens)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Start with random noise
        x = torch.randn(batch_size, seq_len, self.embedding.embedding_dim).to(self.embedding.weight.device)
        
        for i in range(self.num_iterations):
            # Noise decreases over iterations
            noise_std = self.noise_std * (1 - i / (self.num_iterations - 1))
            x = x + torch.randn_like(x) * noise_std

            # Condition the denoising process on the token embeddings
            x = self.transformer_decoder(tgt=x, memory=token_embeddings, tgt_mask=tgt_mask)

        # Final projection to vocabulary size
        logits = self.output_projection(x)  # Shape: (batch_size, seq_len, vocab_size)
        
        return logits