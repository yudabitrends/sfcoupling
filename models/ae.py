from torch import nn

from .mlp import build_mlp


class YAutoencoder(nn.Module):
    def __init__(self, dy: int, latent_dim: int = 64, hidden: int = 256, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = [hidden] * depth
        self.encoder = build_mlp(dy, hidden_dims, latent_dim, dropout=dropout)
        self.decoder = build_mlp(latent_dim, hidden_dims, dy, dropout=dropout)

    def forward(self, y):
        z = self.encoder(y)
        y_hat = self.decoder(z)
        return y_hat, z
