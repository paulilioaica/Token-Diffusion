import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, predicted_noise, actual_noise):
        
        # Compute loss (Mean Squared Error between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, actual_noise)
        
        return loss