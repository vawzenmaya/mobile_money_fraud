import torch
import torch.nn as nn

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
        self._copy_weights()

    def _copy_weights(self):
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data.copy_(online_params.data)
        for online_params, target_params in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_params.data.copy_(online_params.data)

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        if x2 is None:
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
