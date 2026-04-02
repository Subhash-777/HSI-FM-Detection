import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation attention for spectral bands."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        scale = self.se(x).unsqueeze(-1)  # (B, C, 1)
        return x * scale


class SpectralBranch1D(nn.Module):
    def __init__(self, n_bands=204, output_dim=128, architecture='simple'):
        super().__init__()
        self.n_bands = n_bands
        self.output_dim = output_dim

        # Wider first layer + SE attention blocks
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

        # SE attention on raw bands before encoding
        self.band_attention = nn.Sequential(
            nn.Linear(n_bands, n_bands // 8),
            nn.ReLU(inplace=True),
            nn.Linear(n_bands // 8, n_bands),
            nn.Sigmoid()
        )

        print(f"SpectralBranch1D (SE): n_bands={n_bands}, output_dim={output_dim}")

    def forward(self, x):
        # x: (B, n_bands)
        attn = self.band_attention(x)   # (B, n_bands) — learn which bands matter
        x = x * attn                    # attended spectrum
        return self.encoder(x)
