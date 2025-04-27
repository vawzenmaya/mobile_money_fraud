import torch
import torch.nn as nn
import numpy as np
import joblib
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_embeddings(self, x):
        return self.encoder(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(BYOL, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.projector = MLP(latent_dim, hidden_dim, 32)
        self.predictor = MLP(32, hidden_dim, 32)
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        self.target_projector = MLP(latent_dim, hidden_dim, 32)

    def forward(self, x1, x2=None):
        # Modified to make x2 optional for inference
        z1 = self.encoder(x1)
        
        if x2 is None:  # During inference, we only need embeddings
            return z1
            
        p1 = self.projector(z1)
        q1 = self.predictor(p1)
        
        with torch.no_grad():
            z2 = self.target_encoder(x2)
            p2 = self.target_projector(z2)
            
        return q1, p2, z1

    def get_embeddings(self, x):
        return self.encoder(x)

class StudentModel(nn.Module):
    def __init__(self, input_dim):
        super(StudentModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
