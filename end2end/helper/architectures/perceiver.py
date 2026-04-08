import torch
import torch.nn as nn
import torch.nn.functional as F
from end2end.helper.helper_architecture import SinusoidalMultiScaleLocEncoder

class PolygonPerceiver(nn.Module):
    """Initializes polygon perceiver with configurable feature extraction/fusion/processing"""
    def __init__(self,
                 d_model=128, d_latents=128, num_latents=32, num_heads=4, num_self_layers=2,
                 d_pos_enc=32, embedding_dim=128, dim_feedforward_factor=4,
                 num_pool_queries=1, latent_dropout=0.1, use_layernorm_inputs=True,
                 loc_encoding_type="simple", loc_encoding_dim=32,
                 loc_encoding_min_freq=1000.0, loc_encoding_max_freq=5600.0):
        super().__init__()

        # --- Feature extractors ---
        # 1. Fourier features on absolute coordinates (for translation variance)
        self.loc_encoder = SinusoidalMultiScaleLocEncoder(
            loc_encoding_dim=loc_encoding_dim,
            mode=loc_encoding_type,
            min_freq=loc_encoding_min_freq,
            max_freq=loc_encoding_max_freq,
        )

        # 2. Vectors to the centre of gravity (for rotation and scaling variance)
        self.pose_feature_dim = d_model // 2
        self.pose_proj = nn.Linear(2, self.pose_feature_dim)

        # --- Feature Fusion ---
        raw_feature_dim = (self.loc_encoder.output_dim + self.pose_feature_dim + d_pos_enc)

        self.feature_fusion = nn.Sequential(
            nn.Linear(raw_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.input_norm = nn.LayerNorm(d_model) if use_layernorm_inputs else nn.Identity()

        # --- Perceiver ---
        # Latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, d_latents))
        self.latent_dropout = nn.Dropout(latent_dropout)

        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_latents,
            num_heads=num_heads,
            kdim=d_model,
            vdim=d_model,
            batch_first=True,
            dropout=0.0
        )
        self.cross_norm = nn.LayerNorm(d_latents)
        
        # Self-attention in latent space
        dim_feedforward = d_latents * dim_feedforward_factor
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_latents,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
            norm_first=True,
            dropout=0.0
        )
        self.latent_transformer = nn.TransformerEncoder(transformer_layer, num_self_layers)

        # Attention pooling for the final output
        self.pool_query = nn.Parameter(torch.randn(1, num_pool_queries, d_latents))
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_latents,
            num_heads=num_heads,
            batch_first=True
        )

        # Output-Projection
        self.embedding_proj = nn.Linear(d_latents * num_pool_queries, embedding_dim)
        

    def forward(self, x: torch.Tensor, cyclic_pe: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = x.size(0)
        
        # --- 1. feature extraction ---
        # a) Absolute position of each point via Fourier features
        fourier_features = self.loc_encoder(x)

        # b) Absolute orientation & scaling via vectors to the centre of gravity
        with torch.no_grad():
            # Computes centroid using mask or mean
            if mask is not None:
                valid_points_mask = ~mask.unsqueeze(-1)
                num_points = valid_points_mask.sum(dim=1, keepdim=True)
                centroid = (x * valid_points_mask).sum(dim=1, keepdim=True) / (num_points + 1e-6)
            else:
                centroid = x.mean(dim=1, keepdim=True)
        
        centroid_vectors = x - centroid
        pose_features = self.pose_proj(centroid_vectors)

        # --- 2. Feature Fusion ---
        features = torch.cat([
            fourier_features,
            pose_features,
            cyclic_pe,
        ], dim=-1)
        
        features = self.feature_fusion(features)
        features = self.input_norm(features)

        # --- 3. Perceiver processing ---
        # Cross-Attention
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        latents_normed = self.cross_norm(latents)
        update, _ = self.cross_attention(
            latents_normed, features, features,
            key_padding_mask=mask
        )
        latents = latents + self.latent_dropout(update)

        # Self-Attention
        latents = self.latent_transformer(latents, src_key_padding_mask=None)

        # Pooling
        query = self.pool_query.expand(B, -1, -1)
        pooled_latent, _ = self.attention_pool(query=query, key=latents, value=latents)
        pooled_latent = pooled_latent.reshape(B, -1)

        # Output
        embedding = self.embedding_proj(pooled_latent)
        embedding = F.normalize(embedding, dim=-1)

        return embedding