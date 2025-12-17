from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, global_add_pool, global_mean_pool, TransformerConv, global_max_pool
from torch_geometric.nn import MessagePassing, global_max_pool, BatchNorm, LayerNorm
from helper.helper_architecture import PolygonMessagePassing, PolygonGraphBuilder
from torch_geometric.data import Batch
from torch_geometric.utils import softmax, to_dense_batch

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
                 pooling_strategy='attention',
                 use_jumping_knowledge=False):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_jumping_knowledge = use_jumping_knowledge

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
        if self.use_jumping_knowledge:
            self.jump = nn.Linear(num_layers * hidden_dim, hidden_dim)
        else:
            self.jump = None

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
        layer_outputs = [] if self.use_jumping_knowledge else None

        for conv, norm in zip(self.gin_layers, self.batch_norms):
            x_res = x
            x_norm = norm(x)
            x_conv = conv(x_norm, edge_index, edge_attr=edge_attr)
            x = x_res + x_conv
            
            if self.use_jumping_knowledge:
                layer_outputs.append(x)

        # Jumping Knowledge
        if self.use_jumping_knowledge:
            x_final = torch.cat(layer_outputs, dim=-1)
            x_final = self.jump(x_final)
        else:
            x_final = x

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
        if pooling_strategy == 'attention':
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
                 num_latents: int = 16,
                 k_eigvecs: int = 10,
                 num_perceiver_iterations: int = 2,
                 dropout: float = 0.1,
                 pooling_strategy: str = 'attention',
                 loc_encoding_dim: int = 64,
                 loc_encoding_min_freq: float = 1.0,
                 loc_encoding_max_freq: float = 5600.0,
                 loc_encoding_type: str = "multiscale_learnable",
                 graph_encoder_type: str = "gat",
                 use_edge_attr: bool = True
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

        elif graph_encoder_type == "GraphPerceiver":
            self.encoder = PolygonTransformerPerceiver(
                input_dim=self.graph_builder.output_dim,
                edge_dim=self.graph_builder.edge_dim,
                hidden_dim=hidden_dim,
                k_eigvecs=k_eigvecs,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                heads=num_heads,
                num_latents=8,
                num_perceiver_iterations=num_perceiver_iterations,
                dropout=dropout
            )


    def forward(self, data_batch: Batch) -> torch.Tensor:
        full_data_batch = self.graph_builder(data_batch)

        embedding = self.encoder(full_data_batch)

        return embedding

# =============================================================================
# 4. Encoder Graph Transformer with Perceiver Aggregator
# =============================================================================

# --- 1. Der Perceiver Aggregator ---
class PerceiverAggregator(nn.Module):
    def __init__(self, input_dim, num_latents=64, num_heads=4, 
                 dropout=0.1, num_iterations=2):
        super().__init__()
        self.num_latents = num_latents
        self.num_iterations = num_iterations
        self.latents = nn.Parameter(torch.randn(1, num_latents, input_dim) * 0.02)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(input_dim)
        
        # Self Attention
        self.self_attn = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.self_norm = nn.LayerNorm(input_dim)
        
        # FFN (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        x_dense, mask = to_dense_batch(x, batch)
        B = x_dense.shape[0]
        latents = self.latents.expand(B, -1, -1)
        padding_mask = ~mask
        
        for _ in range(self.num_iterations):
            # 1. Cross-Attention (Read from graph)
            attn_out, _ = self.cross_attn(
                query=latents, key=x_dense, value=x_dense,
                key_padding_mask=padding_mask
            )
            latents = self.cross_norm(latents + self.dropout(attn_out))
            
            # 2. Self-Attention (Process)
            self_out, _ = self.self_attn(latents, latents, latents)
            latents = self.self_norm(latents + self.dropout(self_out))
            
            # 3. FFN
            latents = self.ffn_norm(latents + self.ffn(latents))
        
        return latents
        
class LocalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim, heads, dropout):
        super().__init__()
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout
        )
        self.norm1 = LayerNorm(hidden_dim)
        
        # FFN Block
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # Pre-Norm Style (stabiler für tiefe Netze)
        x = x + self.conv(self.norm1(x), edge_index, edge_attr=edge_attr)
        x = x + self.ffn(self.norm2(x))
        return x

class LapPENodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Projiziert die k Eigenvektoren auf die hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, eigvecs):
        """
        x: [num_nodes, hidden_dim] (Node Features nach input_proj)
        eigvecs: [num_nodes, k_eigvecs] (Vom Dataset)
        """
        if eigvecs is None:
            return x
        
        # PE Embedding berechnen
        pe_emb = self.mlp(eigvecs)

        return x + pe_emb

class PolygonTransformerPerceiver(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim,
                 hidden_dim=64,
                 embedding_dim=128,
                 k_eigvecs=10,
                 num_layers=4,
                 heads=4,
                 num_latents=32,
                 num_perceiver_iterations=2,
                 dropout=0.1):
        super().__init__()
    
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. Positional Encoding Encoder
        self.pe_encoder = LapPENodeEncoder(
            in_dim=k_eigvecs, 
            hidden_dim=hidden_dim, 
            dropout=dropout
        )
        
        # Lokale Transformer Blocks
        self.local_blocks = nn.ModuleList([
            LocalTransformerBlock(hidden_dim, edge_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Perceiver Aggregator
        self.global_aggregator = PerceiverAggregator(
            hidden_dim,
            num_latents=num_latents,
            num_heads=heads,
            dropout=dropout,
            num_iterations=num_perceiver_iterations
        )
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch, eigvecs = \
            data.x, data.edge_index, data.edge_attr, data.batch, data.eigvecs
        
        # 1. Input Projection
        x = self.input_proj(x)
        x = self.pe_encoder(x, eigvecs)
        
        # 2. Local Message Passing
        for block in self.local_blocks:
            x = block(x, edge_index, edge_attr)
        
        # 3. Global Aggregation
        latents = self.global_aggregator(x, batch)
        
        # 4. Pool & Project
        graph_embedding = latents.mean(dim=1)
        out = self.output_proj(graph_embedding)
        out = F.normalize(out, dim=-1)
        
        return out
