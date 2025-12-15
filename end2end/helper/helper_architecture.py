import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch

# =============================================================================
# 1. Learnable Fourier Features
# =============================================================================

class LearnableFourierFeatures(nn.Module):
    def __init__(self, input_dim=2, num_frequencies=8, sigma=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(num_frequencies, input_dim) * sigma)

    def forward(self, x_coords):
        """
        Nimmt einen Tensor mit Koordinaten der Form (B, N, 2) entgegen.
        """
        proj = 2 * torch.pi * torch.matmul(x_coords, self.B.t())  # (B, N, F)

        # Chain sine and cosine to learn phase shift
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, N, 2*F)

# =============================================================================
# 2. Sinusoidal MultiScale Location Encoder
# =============================================================================

class SinusoidalMultiScaleLocEncoder(nn.Module):
    def __init__(self, loc_encoding_dim=8, mode: str = 'simple',
                 min_freq=1.0, max_freq=5600.0):
        super().__init__()
        """
            min_freq: niedrigste Frequenz in 1/°
            max_freq: höchste Frequenz in 1/°, z. B. ~5600 für ~5m Auflösung
        """
        self.loc_encoding_dim = loc_encoding_dim
        self.mode = mode

        valid_modes = ['simple', 'multiscale_fixed', 'multiscale_learnable', 'multiscale_dotproduct']
        if mode not in valid_modes:
            raise ValueError(f"Ungültiger Modus '{mode}'. Wähle einen aus {valid_modes}.")
        if loc_encoding_dim <= 4:
            raise ValueError("loc_encoding_dim muss positiv sein.")

        self.register_buffer('norm_factor', torch.tensor([180.0, 90.0]))
        self.register_buffer('pi', torch.tensor(torch.pi))
        self.output_dim = loc_encoding_dim

        if mode == 'simple':
            self.output_dim = 4
        elif mode == 'multiscale_fixed':
            num_scales = loc_encoding_dim // 4
            freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_scales)
            self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float32))
        elif mode == 'multiscale_learnable':
            num_scales = loc_encoding_dim // 4
            init_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_scales)
            self.freqs = nn.Parameter(torch.tensor(init_freqs, dtype=torch.float32))
            self.output_dim = loc_encoding_dim
        elif mode == 'multiscale_dotproduct':
            num_scales = loc_encoding_dim // 2
            # Logarithmically distributed frequencies + random directions
            freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_scales)
            dirs = torch.nn.functional.normalize(torch.randn(num_scales, 2), dim=-1)
            self.B = nn.Parameter(torch.tensor(freqs, dtype=torch.float32).unsqueeze(1) * dirs)

    def forward(self, x_coords):
        B, N, _ = x_coords.shape
        norm_coords = x_coords / self.norm_factor.to(x_coords.device, x_coords.dtype)

        if self.mode == 'simple':
            angles = norm_coords * self.pi
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        elif self.mode == 'multiscale_dotproduct':
            angles = torch.einsum('bni,ki->bnk', norm_coords, self.B) * self.pi
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            coords_expanded = norm_coords.unsqueeze(2)
            freqs_expanded = self.freqs.view(1, 1, -1, 1)
            scaled_coords = coords_expanded * freqs_expanded
            angles = scaled_coords * self.pi
            sines = torch.sin(angles)
            cosines = torch.cos(angles)
            return torch.cat([sines, cosines], dim=-1).view(B, N, -1)

# =============================================================================
# 3. Cyclic Relative Positional Encoding
# =============================================================================

class CyclicRelativePosEncoding(nn.Module):
    def __init__(self, d_pos_enc: int, max_freq: float = 10000.0):
        super().__init__()
        if d_pos_enc % 2 != 0:
            raise ValueError(f"d_pos_enc must be even, got {d_pos_enc}")
        self.d_pos_enc = d_pos_enc
        inv_freq = 1.0 / (max_freq ** (torch.arange(0, d_pos_enc, 2, dtype=torch.float32) / d_pos_enc))
        self.register_buffer('inv_freq', inv_freq)

    def _sinus_cosine_encoding(self, values: torch.Tensor) -> torch.Tensor:
        angles = values.unsqueeze(-1) * self.inv_freq
        return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=-2)

    def forward(self, x_coords: torch.Tensor) -> torch.Tensor:
        B, N, C = x_coords.shape
        if C != 2:
            raise ValueError(f"Expected 2D coordinates, got {C}D")
        if N <= 1:
            return torch.zeros(B, N, self.d_pos_enc, device=x_coords.device, dtype=x_coords.dtype)

        deltas = torch.roll(x_coords, shifts=-1, dims=1) - x_coords
        segment_lengths = torch.norm(deltas, p=2, dim=-1)
        perimeter = segment_lengths.sum(dim=1, keepdim=True).clamp(min=1e-8)
        cumulative_dist = F.pad(torch.cumsum(segment_lengths, dim=1)[:, :-1], (1, 0), "constant", 0)
        rel_pos = cumulative_dist / perimeter
        return self._sinus_cosine_encoding(rel_pos)

# =============================================================================
# 4. masked_random_roll
# =============================================================================

def masked_random_roll(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    device = x.device

    if mask is None:
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    valid_len = (~mask).sum(dim=1)
    shifts = (torch.rand(B, device=device) * valid_len.float()).long()
    arange_N = torch.arange(N, device=device)
    rolled_indices = (arange_N[None, :] - shifts[:, None]) % valid_len.view(-1, 1)
    is_padded = arange_N[None, :] >= valid_len.view(-1, 1)
    final_indices = torch.where(is_padded, arange_N, rolled_indices)
    final_indices = final_indices.unsqueeze(-1).expand(-1, -1, D)
    rolled_x = torch.gather(x, dim=1, index=final_indices)

    return rolled_x

# =============================================================================
# 5. Graph Construction
# =============================================================================

class FourierPolygonGraph(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def polygon_to_graph(self, polygon_tensor) -> Data:
        polygon_tensor = polygon_tensor.to(self.device)
        n = polygon_tensor.size(0)

        # Random cyclic rotation
        shift = torch.randint(0, n, (1,), device=self.device).item()
        polygon_tensor = torch.roll(polygon_tensor, shifts=-shift, dims=0)

        i = torch.arange(n, device=self.device)
        j = (i + 1) % n
        edge_index = torch.stack([torch.cat([i, j]), torch.cat([j, i])], dim=0)
        return Data(pos=polygon_tensor, edge_index=edge_index)

# =============================================================================
# 5. Graph Construction new
# =============================================================================

class PolygonGraphBuilder(nn.Module):
    def __init__(self,
                 loc_encoding_dim: int = 64,
                 loc_encoding_min_freq: float = 1.0,
                 loc_encoding_max_freq: float = 5600.0,
                 loc_encoding_type: str = 'multiscale_learnable',
                 use_edge_attr: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        self.loc_encoder = SinusoidalMultiScaleLocEncoder(
            loc_encoding_dim=loc_encoding_dim,
            min_freq=loc_encoding_min_freq,
            max_freq=loc_encoding_max_freq,
            mode=loc_encoding_type
        )

        self.output_dim = self.loc_encoder.output_dim

        # Edge dimensions: dx, dy (2) + length (1) = 3
        # + sine/cosine angle (2) = 5
        self.edge_dim = 3 if use_edge_attr else 0


    def forward(self, data):
        # --- 1. Node features (absolute position) ---
        # We use the positions from the batch directly.
        data.x = self.loc_encoder(data.pos.unsqueeze(0))[0]
        

        # --- 2. Edge features ---
        if self.use_edge_attr:
            src, dst = data.edge_index

            # Vector between nodes (dx, dy) -> Rotation & scaling variant
            dirs = data.pos[dst] - data.pos[src]

            # Euclidean distance -> Scaling variant
            lengths = torch.norm(dirs, dim=1, keepdim=True)
            
            # --- Edge angle (absolute) ---
            # Generalisation removes points, but the ‘direction’ of the line remains similar.
            # angles = torch.atan2(dirs[:, 1], dirs[:, 0]).unsqueeze(1)
            # We encode the angle as sin/cos to avoid the jump from pi to -pi.
            # angles_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

            # Assembling: [dx, dy, length, sin_angle, cos_angle] angles_emb
            data.edge_attr = torch.cat([dirs, lengths], dim=1)

        return data

class PolygonMessagePassing(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr='mean') # 'add', 'mean', 'max'
        self.msg_net = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim) 
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Creates a message from the features of the neighbouring node and the edge
        return self.msg_net(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        # Updates the node vector based on aggregated messages
        return self.update_net(torch.cat([x, aggr_out], dim=-1))