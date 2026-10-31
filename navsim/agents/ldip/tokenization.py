import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryVocabularyTokenizer(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # Input features per timestep:
        # x, y, sin(h), cos(h), dx, dy, d(sin_h), d(cos_h) -> 8 channels
        in_channels = 8

        # 1D CNN over the time dimension (40 timesteps)
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),  # Downsamples time to 20
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, output_dim, kernel_size=3, padding=1, stride=2),  # Downsamples time to 10
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

        # Final projection to exact C dimension
        self.fc = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, vocab):
        """
        vocab: Tensor of shape (L, 40, 3) where 3 is [x, y, heading]
        Returns: Tensor of shape (L, C) where C is output_dim
        """
        L, T, _ = vocab.shape

        # 1. Feature Engineering
        x_y = vocab[..., :2]  # (L, 40, 2)
        heading = vocab[..., 2:3]  # (L, 40, 1)

        sin_h = torch.sin(heading)  # (L, 40, 1)
        cos_h = torch.cos(heading)  # (L, 40, 1)

        # Base continuous features: (L, 40, 4)
        base_features = torch.cat([x_y, sin_h, cos_h], dim=-1)

        # Calculate step-to-step differences (velocities)
        # Pad the first timestep with zeros so shape remains (L, 40, 4)
        deltas = torch.zeros_like(base_features)
        deltas[:, 1:, :] = base_features[:, 1:, :] - base_features[:, :-1, :]

        # Combine everything: (L, 40, 8)
        enriched_features = torch.cat([base_features, deltas], dim=-1)

        # 2. Reshape for Conv1d: (Batch, Channels, Length) -> (L, 8, 40)
        enriched_features = enriched_features.transpose(1, 2)

        # 3. Pass through Temporal CNN
        # Out shape: (L, output_dim, 10)
        conv_out = self.conv_net(enriched_features)

        # 4. Global Average Pooling (or Max Pooling) to squash the time dimension
        # Out shape: (L, output_dim, 1) -> (L, output_dim)
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)

        # 5. Final MLP projection
        encoded_vocab = self.fc(pooled)

        return encoded_vocab
