from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, global_add_pool, global_mean_pool, TransformerConv, global_max_pool
from torch_geometric.nn import MessagePassing, global_max_pool, BatchNorm, LayerNorm
from helper.helper_architecture import PolygonMessagePassing, PolygonGraphBuilder, CrossAttentionPooling
from torch_geometric.data import Batch
from torch_geometric.utils import softmax

# =============================================================================
# 1. Encoder Graph Isomorphism Network
# =============================================================================

class PolygonGINEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim, 
                 hidden_dim=32,
                 embedding_dim=128,
                 num_layers=5,
                 dropout=0.1,
                 pooling_strategy='attention'):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                BatchNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            # GINConv wird durch GINEConv ersetzt, das `edge_dim` als Argument erhält
            self.gin_layers.append(GINEConv(nn_block, train_eps=True, edge_dim=edge_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Jumping Knowledge-Verbindung bleibt erhalten
        self.jump = nn.Linear(num_layers * hidden_dim, hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        if pooling_strategy == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        self.pooling_strategy = pooling_strategy

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.input_proj(x)
        layer_outputs = []

        for conv, norm in zip(self.gin_layers, self.batch_norms):
            x_res = x
            
            x_norm = norm(x)
            
            x_conv = conv(x_norm, edge_index, edge_attr=edge_attr)
            
            x = x_res + x_conv
            
            layer_outputs.append(x)

        # Jumping Knowledge
        x_cat = torch.cat(layer_outputs, dim=-1)
        x_final = self.jump(x_cat)

        # Pooling
        if self.pooling_strategy == 'attention':
            att_weights = self.attention_pool(x_final)
            pooled = global_add_pool(x_final * att_weights, batch)
        else:
            pooled = global_add_pool(x_final, batch)

        embedding = self.output_proj(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return embedding
        
# =============================================================================
# 2. Encoder Graph Attention Network v2
# =============================================================================

class PolygonGATV2Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim,
                 hidden_dim=32,
                 embedding_dim=128,
                 num_heads=4,
                 num_layers=4,
                 dropout=0.1,
                 pooling_strategy='attention'):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # Der Input für den ersten Layer ist hidden_dim, für die folgenden hidden_dim * num_heads
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads

            # Die letzte Schicht hat nur einen Head und keine Verkettung
            is_last_layer = (i == num_layers - 1)
            heads = 1 if is_last_layer else num_heads
            out_channels = hidden_dim if is_last_layer else hidden_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=not is_last_layer,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            norm_dim = hidden_dim * heads if not is_last_layer else hidden_dim
            self.layer_norms.append(nn.LayerNorm(norm_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        if pooling_strategy == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        self.pooling_strategy = pooling_strategy

        # Eine Liste für die Projektionslayer der Residuen-Verbindungen
        self.residual_projs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            is_last_layer = (i == num_layers - 1)
            out_channels = hidden_dim if is_last_layer else hidden_dim * num_heads
            
            # Wenn die Dimensionen nicht übereinstimmen, füge einen Projektionslayer hinzu
            if in_channels != out_channels:
                self.residual_projs.append(nn.Linear(in_channels, out_channels))
            else:
                self.residual_projs.append(nn.Identity())

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_proj(x)

        for i, (gat_layer, layer_norm, res_proj) in enumerate(zip(self.gat_layers, self.layer_norms, self.residual_projs)):
            x_res = x  # Residuum vom vorherigen Layer speichern

            # GAT-Layer anwenden
            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            
            # Residuum auf die richtige Dimension projizieren und addieren
            x = x + res_proj(x_res)
            
            # Normalisierung und Aktivierung
            x = layer_norm(x)
            x = F.gelu(x) # GELU ist oft eine gute Wahl

        if self.pooling_strategy == 'attention':
            att_weights = self.attention_pool(x)
            pooled = global_add_pool(x * att_weights, batch)
        else:
            pooled = global_mean_pool(x, batch)

        embedding = self.output_proj(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return embedding

# =============================================================================
# 3. Encoder MY OWN Message Passing Mechanismus
# =============================================================================

class PolygonMessageEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim,
                 hidden_dim=64,
                 embedding_dim=128,
                 num_layers=3,
                 dropout=0.1,
                 pooling_strategy='cross_attention',
                 num_queries=8):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = dropout

        self.mp_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.mp_layers.append(
                PolygonMessagePassing(
                    in_dim=hidden_dim,
                    edge_dim=edge_dim,
                    out_dim=hidden_dim
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'cross_attention':
            self.cross_attention_pool = CrossAttentionPooling(hidden_dim, num_queries, num_heads=4)
            # Cross-attention returns 2 * hidden_dim (max + mean)
            pooled_dim = 2 * hidden_dim
        elif pooling_strategy == 'attention':
            self.attention_pool_linear = nn.Linear(hidden_dim, 1)
            pooled_dim = hidden_dim
        else:
            pooled_dim = hidden_dim

        self.output_proj = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_pos = data.pos

        # 1. Input-Projektion
        x = self.input_proj(x)

       # 2. Message Passing Layers mit Pre-LayerNorm
        for layer, norm in zip(self.mp_layers, self.layer_norms):
            # Pre-LayerNorm: Normierung VOR der Schicht
            x_normalized = norm(x)

            # Message Passing auf normalisierte Features
            x_update = layer(x_normalized, edge_index, edge_attr)

            # Aktivierung und Dropout
            x_update = F.relu(x_update)
            x_update = F.dropout(x_update, p=self.dropout, training=self.training)

            # Residual Connection: Original + Update
            x = x + x_update

        # 3. Graph Pooling
        if self.pooling_strategy == 'cross_attention':
            pooled = self.cross_attention_pool(x, node_pos, batch)
        elif self.pooling_strategy == 'attention':
            att_logits = self.attention_pool_linear(x).squeeze(-1)
            att_weights = softmax(att_logits, batch)
            pooled = global_add_pool(x * att_weights.unsqueeze(-1), batch)
        else:
            pooled = global_mean_pool(x, batch)

        # 4. Finale Projektion zum Embedding
        embedding = self.output_proj(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return embedding


class GraphPolygonEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = 64,
                 embedding_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 pooling_strategy: str = 'attention',
                 loc_encoding_dim: int = 64,
                 loc_encoding_min_freq: float = 1.0,
                 loc_encoding_max_freq: float = 5600.0,
                 loc_encoding_type: str = "multiscale_learnable",
                 graph_encoder_type: str = "gat",
                 use_edge_attr: bool = True,
                 lap_pe_k: int = 10,
                 norm_type: str = 'batch' 
                 ):
        super().__init__()

        self.graph_builder = PolygonGraphBuilder(
            loc_encoding_dim=loc_encoding_dim,
            loc_encoding_min_freq=loc_encoding_min_freq,
            loc_encoding_max_freq=loc_encoding_max_freq,
            loc_encoding_type=loc_encoding_type,
            use_edge_attr=use_edge_attr
        )
    
        if graph_encoder_type == "gat":
            self.encoder = PolygonGATV2Encoder(
                input_dim=self.graph_builder.output_dim,
                edge_dim=self.graph_builder.edge_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                pooling_strategy=pooling_strategy
            )
        elif graph_encoder_type == "gine":
            self.encoder = PolygonGINEEncoder(
                input_dim=self.graph_builder.output_dim,
                edge_dim=self.graph_builder.edge_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling_strategy=pooling_strategy
            )
        elif graph_encoder_type == "mp":
            self.encoder = PolygonMessageEncoder(
                input_dim=self.graph_builder.output_dim,
                edge_dim=self.graph_builder.edge_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling_strategy=pooling_strategy
            )

        elif graph_encoder_type == "GraphTransformer":
            self.encoder = PolygonGraphTransformerEncoder(
                input_dim=self.graph_builder.output_dim,
                edge_dim=self.graph_builder.edge_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                pooling_strategy=pooling_strategy,
                use_laplacian_pe=True,
                lap_pe_k=lap_pe_k,
                norm_type=norm_type
            )


    def forward(self, data_batch: Batch) -> torch.Tensor:
        full_data_batch = self.graph_builder(data_batch)

        embedding = self.encoder(full_data_batch)

        return embedding

# =============================================================================
# 4. Encoder Graph Transformer
# =============================================================================

class GraphTransformerBlock(nn.Module):
    """
    A single graph transformer block.
    Structure: Norm -> Attention (with edge features) -> Residual -> Norm -> Feed Forward -> Residual
    """
    def __init__(self, in_dim, out_dim, heads, dropout, edge_dim, norm_type='layer'):
        super().__init__()
        
        # 1. Attention Layer
        # This layer also calculates attention scores based on edge attributes.
        # Important: out_channels per head is out_dim // heads.
        self.attn = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads, 
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True, # Learns a gating mechanism (bias) in the attention step
            concat=True # Concatenates the heads (default for Transformer)
        )
        
        # 2. Normalization
        # Standard transformers use LayerNorm.
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)
        else:
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        
        # 3. Feed Forward Network (FFN)
        # Standard MLP block as in any transformer
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        self.dropout_val = dropout

    def forward(self, x, edge_index, edge_attr):
        # --- Attention Sub-Layer ---
        x_in = x
        
        # Pre-normalization (often more stable training for deep networks)
        x = self.norm1(x)
        
        # Attention calculation with edge features
        x = self.attn(x, edge_index, edge_attr=edge_attr)
        
        # Residual Connection
        x = x_in + F.dropout(x, p=self.dropout_val, training=self.training)
        
        # --- Feed Forward Sub-Layer ---
        x_in = x
        x = self.norm2(x)
        x = self.ffn(x)
        
        # Residual Connection
        x = x_in + F.dropout(x, p=self.dropout_val, training=self.training)
        
        return x


class PolygonGraphTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim,
                 hidden_dim=64,
                 embedding_dim=128,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1,
                 pooling_strategy='max',
                 use_laplacian_pe=False, # Adjustment parameter: Positional Encoding
                 lap_pe_k=10,  # Adjustment parameter: Number of LapPE dimensions
                 norm_type='layer'):     # Adjustment parameter: Normalization type
        super().__init__()

        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Optional: Laplacian Positional Encoding (LapPE)
        self.use_laplacian_pe = use_laplacian_pe
        if use_laplacian_pe:
            self.pe_proj = nn.Linear(lap_pe_k, hidden_dim) 

        # Stack of transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerBlock(
                    in_dim=hidden_dim, 
                    out_dim=hidden_dim, 
                    heads=num_heads, 
                    dropout=dropout, 
                    edge_dim=edge_dim,
                    norm_type=norm_type
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Pooling strategy
        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 1. Feature Projection
        x = self.input_proj(x)

        # 2. Additive Positional Encoding (if enabled)
        # Graph transformers require positional information because attention is inherently "position-blind".
        if self.use_laplacian_pe and hasattr(data, 'eigvecs'):
            # Random sign flip trick from [1] for robustness
            eigvecs = data.eigvecs
            if self.training:
                sign_flip = torch.rand(eigvecs.size(1), device=x.device) < 0.5
                sign_flip = sign_flip.float() * 2 - 1
                eigvecs = eigvecs * sign_flip.unsqueeze(0)
            
            # Add projected eigenvectors to node feature
            x = x + self.pe_proj(eigvecs)

        # 3. Passing through the transformer blocks
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # 4. Pooling (creating graph embedding)
        if self.pooling_strategy == 'attention':
            att_weights = self.attention_pool(x)
            pooled = global_add_pool(x * att_weights, batch)
        elif self.pooling_strategy == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.pooling_strategy == 'max':
            pooled = global_max_pool(x, batch)
        else:
            pooled = global_mean_pool(x, batch)

        # 5. Final Embedding
        embedding = self.output_proj(pooled)
        embedding = F.normalize(embedding, dim=-1)
        return embedding

