import torch
import torch.nn as nn

from end2end.helper.helper_architecture import SinusoidalMultiScaleLocEncoder


class PolygonPairClassifier(nn.Module):
    def __init__(self, shape_feat_dim=14, num_frequencies=4,
                 hidden_layers=None, dropout_rate=0.2, 
                 activation_name="LeakyReLU", 
                 sinusoidal_mode="simple",
                 min_freq=1.0, max_freq=5600.0):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.fourier = SinusoidalMultiScaleLocEncoder(
            loc_encoding_dim=num_frequencies * 2,
            mode=sinusoidal_mode,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        self.shape_feat_dim = shape_feat_dim - 2
        self.fourier_dim = self.fourier.output_dim
        total_input_dim = 2 * (self.shape_feat_dim + self.fourier_dim)

        activation = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(0.01),
            "ELU": nn.ELU()
        }[activation_name]

        self.input_block = nn.Sequential(
            nn.Linear(total_input_dim, hidden_layers[0]),
            activation,
            nn.BatchNorm1d(hidden_layers[0]),
            nn.Dropout(dropout_rate)
        )

        self.hidden_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation,
                nn.BatchNorm1d(hidden_layers[i + 1]),
                nn.Dropout(dropout_rate)
            ) for i in range(len(hidden_layers) - 1)
        ])

        self.output_block = nn.Linear(hidden_layers[-1], 1)

    def encode_polygon(self, features):
        centroid = features[:, [5, 6]].unsqueeze(1)
        other_feats = torch.cat([features[:, :5], features[:, 7:]], dim=1)  # ohne centroid
        fourier_enc = self.fourier(centroid).squeeze(1)
        return torch.cat([other_feats, fourier_enc], dim=1)

    def forward(self, poly1_feats, poly2_feats):
        enc1 = self.encode_polygon(poly1_feats)
        enc2 = self.encode_polygon(poly2_feats)
        x = torch.cat([enc1, enc2], dim=1)
        x = self.input_block(x)
        x = self.hidden_blocks(x)
        return self.output_block(x)