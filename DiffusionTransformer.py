import torch
import torch.nn as nn
from SinusoidalPosEmb import SinusoidalPosEmb

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(fn[0].in_features)

    def forward(self, x):
        return x + self.fn(self.norm(x))

class DiffusionTransformer(nn.Module):
    def __init__(
        self, state_dim, action_dim, time_dim=32, hidden_dim=256, num_layers=6, num_heads=8, ff_hidden=512
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.input_proj = nn.Linear(state_dim + action_dim + hidden_dim, hidden_dim)

        encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_hidden,
                dropout=0.1,
                batch_first=True,
                activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.transformer = nn.Sequential(*encoder_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, action_dim)
        )

    def forward(self, state, noisy_action, time_step):
        B = state.size(0)

        # Time embedding
        time_emb = self.time_mlp(time_step)  # [B, hidden_dim]

        # Concatenate inputs
        x = torch.cat([state, noisy_action, time_emb], dim=-1)  # [B, D]
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, hidden_dim]

        # Pass through stacked transformer layers
        x = self.transformer(x)  # [B, 1, hidden_dim]

        # Final projection
        return self.output_proj(x.squeeze(1))  # [B, action_dim]
