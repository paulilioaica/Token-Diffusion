import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x, context):
        # Apply multihead attention with a residual connection
        attn_output, _ = self.multihead_attn(x, context, context)
        x = x + attn_output  # Residual connection
        return x
    
class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_length, num_steps, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))
        self.num_steps = num_steps
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        
        # Guided attention layer
        self.guided_attention = GuidedAttentionLayer(embed_dim, num_heads)
        
        # Noise predictor
        self.noise_predictor = nn.Linear(embed_dim, embed_dim)

        
    def add_noise(self, x, t):
        alphas = self.noise_schedule(t).view(-1, 1, 1)
        noise = torch.randn_like(x)
        noised_x = alphas.sqrt() * x + (1 - alphas).sqrt() * noise
        return noised_x, noise
    
    def noise_schedule(self, t):
        return 1 - (t / self.num_steps)
    
    def generate(self, tokens, t):
        #same as forward but without the embedding part

        # Add noise
        noised_x, noise = self.add_noise(tokens, t)

        # Reshape for batch processing
        batch_size, seq_len, embed_dim = noised_x.shape
        noised_x = noised_x.view(batch_size, seq_len, embed_dim)
        tokens = tokens.view(batch_size, seq_len, embed_dim)
        
        # Apply guided attention
        guided_x = self.guided_attention(noised_x, tokens)

        # Pass through transformer encoder
        encoded = self.transformer_encoder(guided_x)

        # Predict the noise
        predicted_noise = self.noise_predictor(encoded)
        
        return predicted_noise, noise
    
    def forward(self, tokens, t):
        # Get original embeddings
        original_embed = self.embedding(tokens) + self.position_encoding[:, :tokens.shape[1], :]
        
        # Add noise
        noised_x, noise = self.add_noise(original_embed, t)
        
        # Reshape for batch processing
        batch_size, seq_len, embed_dim = noised_x.shape
        noised_x = noised_x.view(batch_size, seq_len, embed_dim)
        original_embed = original_embed.view(batch_size, seq_len, embed_dim)

        # Apply guided attention
        guided_x = self.guided_attention(noised_x, original_embed)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(guided_x)
        
        # Predict the noise
        predicted_noise = self.noise_predictor(encoded)
        
        return predicted_noise, noise
