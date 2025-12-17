import torch
import math
import json
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Tuple, Optional, Any
from helper.polygonaugmenter import PolygonAugmenter
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

class PairPolygonDataset(Dataset):
    def __init__(
            self,
            parquet_path: str,
            intersection_path: str,
            hard_candidates_path: str,
            negative_strategies: Dict[str, Any],
            positive_ratio: float = 0.5,
            model_mode: str = "graph",
            augmenter=None,
            k_eigvecs: int = 10  # Number of eigenvectors for graph transformer
    ):
        self.parquet_path = parquet_path
        self.model_mode = model_mode
        self.k_eigvecs = k_eigvecs

        df = pd.read_csv(intersection_path)
        self.intersection_pairs = df.values.astype(str).tolist()

        df = pd.read_parquet(parquet_path)
        self.polygon_ids = df['polygon_id'].unique()
        self.id_to_idx = {pid: i for i, pid in enumerate(self.polygon_ids)}

        self.lookup_table = {}
        for row in df.itertuples():
            self.lookup_table[(row.polygon_id, row.variation)] = torch.tensor(np.vstack(row.coordinates), dtype=torch.float)

        self.polygon_variations = df.groupby('polygon_id')['variation'].apply(list).to_dict()

        self.augmenter = augmenter if augmenter else PolygonAugmenter()
        self.positive_ratio = positive_ratio
        self.negative_strategies = {k: float(v) for k, v in negative_strategies.items()}

        if not math.isclose(sum(self.negative_strategies.values()), 1.0):
            raise ValueError("Probabilities in negative_strategies must sum to 1.")

        with open(hard_candidates_path, 'r') as f:
            self.hard_candidates = json.load(f)

    def __len__(self) -> int:
        return len(self.polygon_ids) * 10

    def _random_roll(self, pos: torch.Tensor) -> torch.Tensor:
        n = pos.size(0)
        if n < 2:
            return pos
        roll = torch.randint(0, n, (1,)).item()
        return pos.roll(-roll, dims=0)

    def _load_polygon(self, poly_id: str, variation: Optional[str] = None) -> torch.Tensor:
        if variation is None:
            variation = np.random.choice(self.polygon_variations[poly_id])

        return self.lookup_table[(poly_id, variation)].clone()

    def _compute_ring_pe(self, num_nodes: int, k: int) -> torch.Tensor:
        """
        Analytically calculates the Laplacian eigenvectors for a ring graph.
        Very fast (O(N)), no expensive eigen decomposition required.
        """
        if num_nodes < 3:
            return torch.zeros((num_nodes, k))

        # Indices n and frequencies
        n = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
        # We need k features, i.e. k/2 frequencies (since sin+cos pairs)
        num_freqs = (k + 1) // 2
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32).unsqueeze(0)

        # Argument: 2 * pi * k * n / N
        arg = 2 * torch.pi * freqs * n / num_nodes

        # Sine and cosine components
        cos_part = torch.cos(arg) * (2.0 / num_nodes) ** 0.5
        sin_part = torch.sin(arg) * (2.0 / num_nodes) ** 0.5
        
        pe = torch.cat([cos_part, sin_part], dim=1)
        return pe[:, :k]

    def _build_graph_data(self, coordinates: torch.Tensor) -> Data:
        """Creates PyG Data with ring topology and LapPE."""
        num_nodes = coordinates.size(0)

        # 1. Create edges (ring: 0-1, 1-2, ..., N-0)
        source = torch.arange(num_nodes, dtype=torch.long)
        target = torch.roll(source, -1)
        edge_index = torch.stack([source, target], dim=0)
        # Make it undirected
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # 2. Analytical Positional Encodings
        eigvecs = self._compute_ring_pe(num_nodes, self.k_eigvecs)

        return Data(pos=coordinates, edge_index=edge_index, eigvecs=eigvecs)

    def __getitem__(self, index: int):
        anchor_idx = index % len(self.polygon_ids)
        anchor_id = self.polygon_ids[anchor_idx]
        other_id = anchor_id

        if np.random.random() < self.positive_ratio:
            variations = self.polygon_variations[anchor_id]
            if len(variations) >= 2 and np.random.random() > 0.16:
                v1, v2 = np.random.choice(variations, 2, replace=False)
            else:
                v1 = v2 = variations[0]

            poly1 = self._load_polygon(anchor_id, v1)
            poly2 = self._load_polygon(anchor_id, v2)
            strategy = f"{v1}-{v2}"
        else:
            poly1 = self._load_polygon(anchor_id)
            strategy = np.random.choice(list(self.negative_strategies.keys()), p=list(self.negative_strategies.values()))

            if strategy == 'modify':
                poly2 = poly1.clone()
                poly2 = self.augmenter(poly2)
                other_id = str(random.randint(100000, 999999))

            elif strategy == 'same_center_different_shape':
                other_id = np.random.choice([pid for pid in self.polygon_ids if pid != anchor_id])
                poly2 = self._load_polygon(other_id)
                poly2 = self.augmenter(poly1, second_polygon=poly2)

            elif strategy == 'random_other':
                other_id = np.random.choice([pid for pid in self.polygon_ids if pid != anchor_id])
                poly2 = self._load_polygon(other_id)

            elif strategy == 'cluster':
                candidates = self.hard_candidates.get(anchor_id, [])
                if candidates:
                    other_id = np.random.choice(candidates)
                    poly2 = self._load_polygon(other_id)
                else:
                    other_id = np.random.choice([pid for pid in self.polygon_ids if pid != anchor_id])
                    poly2 = self._load_polygon(other_id)

            elif strategy == 'intersecting':
                rand_pair = random.choice(self.intersection_pairs)
                anchor_id = rand_pair[0]
                other_id = rand_pair[1]

                poly1 = self._load_polygon(anchor_id, "v0")
                poly2 = self._load_polygon(other_id, "v0")

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        # Random Roll
        poly1 = self._random_roll(poly1)
        poly2 = self._random_roll(poly2)

        if self.model_mode == "graph":
            poly1 = self._build_graph_data(poly1)
            poly2 = self._build_graph_data(poly2)

            
        return poly1, poly2, int(anchor_id), int(other_id), strategy

def collate_graph_pairs(batch: list) -> dict:
    graphs1, graphs2, ids1, ids2, strategy = zip(*batch)
    return {
        "poly1": Batch.from_data_list(graphs1),
        "poly2": Batch.from_data_list(graphs2),
        "poly1_i": torch.tensor(ids1, dtype=torch.long),
        "poly2_i": torch.tensor(ids2, dtype=torch.long),
        "strategy": strategy
    }


def collate_pair_sequence(batch):
    poly1, poly2, poly1_i, poly2_i, strategy = zip(*batch)
    poly1_lengths = torch.tensor([poly.size(0) for poly in poly1], dtype=torch.long)
    poly2_lengths = torch.tensor([poly.size(0) for poly in poly2], dtype=torch.long)
    
    poly1_padded = pad_sequence(poly1, batch_first=True, padding_value=0.0)
    poly2_padded = pad_sequence(poly2, batch_first=True, padding_value=0.0)
    
    max_len1 = poly1_padded.size(1)
    max_len2 = poly2_padded.size(1)
    
    poly1_mask = (torch.arange(max_len1).unsqueeze(0) >= poly1_lengths.unsqueeze(1))
    poly2_mask = (torch.arange(max_len2).unsqueeze(0) >= poly2_lengths.unsqueeze(1))

    return {
        "poly1": poly1_padded, "poly2": poly2_padded,
        "poly1_mask": poly1_mask, "poly2_mask": poly2_mask,
        "poly1_i": torch.tensor(poly1_i, dtype=torch.long),
        "poly2_i": torch.tensor(poly2_i, dtype=torch.long),
        "poly1_lengths": poly1_lengths, "poly2_lengths": poly2_lengths,
        "strategy": strategy
    }

def split_dataset(dataset, train_ratio, val_ratio, seed=42):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    random.seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))[:2]

def get_dataloader(config, model_mode="graph"):
    augmenter = PolygonAugmenter(**config["dataset"]["polygon_augmentation"])

    full_dataset = PairPolygonDataset(
        parquet_path=config["dataset"]["parquet_path"],
        intersection_path=config["dataset"]["intersection_path"],
        hard_candidates_path=config["dataset"]["hard_candidates_path"],
        negative_strategies=config["dataset"]["negative_strategies"],
        positive_ratio=config["dataset"].get("positive_ratio", 0.5),
        model_mode=model_mode,
        augmenter=augmenter,
        k_eigvecs=config["graph_encoder"].get("k_eigvecs", 10)
    )

    collate_fn = collate_graph_pairs if model_mode == "graph" else collate_pair_sequence

    train_dataset, val_dataset = split_dataset(
        full_dataset,
        train_ratio=config['dataset']['train_split'],
        val_ratio=config['dataset']['val_split']
    )

    def make_loader(dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=config['dataset']["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=config["dataset"]["num_workers"],
            pin_memory=False,
            persistent_workers=config["dataset"]["num_workers"] > 0,
            prefetch_factor=config["dataset"].get("prefetch_factor", 2)
        )

    return (
        make_loader(train_dataset, shuffle=True, drop_last=True),
        make_loader(val_dataset, shuffle=False, drop_last=False)
    )

#=======================
class EvaluationPolygonDataset(Dataset):
    def __init__(self, parquet_path: str, model_mode: str = "graph", k_eigvecs: int = 5):
        self.table = pq.read_table(parquet_path)
        self.model_mode = model_mode
        self.k_eigvecs = k_eigvecs


    def __len__(self) -> int:
        return len(self.table)

    def _compute_ring_pe(self, num_nodes: int, k: int) -> torch.Tensor:
        if num_nodes < 3: return torch.zeros((num_nodes, k))
        n = torch.arange(num_nodes, dtype=torch.float32).unsqueeze(1)
        freqs = torch.arange(1, ((k + 1) // 2) + 1, dtype=torch.float32).unsqueeze(0)
        arg = 2 * np.pi * freqs * n / num_nodes
        pe = torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)
        return pe[:, :k]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        row = self.table.slice(index, 1)
        polygon_id = int(row['polygon_id'].to_pylist()[0])
        coordinates = row['coordinates'].to_pylist()[0]
        polygon = torch.tensor(coordinates, dtype=torch.float)

        if self.model_mode == "graph":
            num_nodes = polygon.size(0)
            source = torch.arange(num_nodes, dtype=torch.long)
            target = torch.roll(source, -1)
            edge_index = to_undirected(torch.stack([source, target], dim=0), num_nodes=num_nodes)
            eigvecs = self._compute_ring_pe(num_nodes, self.k_eigvecs)
            
            polygon = Data(x=polygon, pos=polygon, edge_index=edge_index, eigvecs=eigvecs)

        return polygon, polygon_id

def eval_collate_fn_seq(batch):
    polygons, polygon_ids = zip(*batch)
    # Calculate lengths for masks
    lengths = torch.tensor([poly.size(0) for poly in polygons], dtype=torch.long)
    # Apply padding
    polygons_padded = pad_sequence(polygons, batch_first=True, padding_value=0.0)
    # Create masks (true for padding positions)
    max_len = polygons_padded.size(1)
    idx_range = torch.arange(max_len).unsqueeze(0)
    masks = (idx_range >= lengths.unsqueeze(1))
    return {
        "polygons": polygons_padded,
        "masks": masks,
        "polygon_ids": torch.tensor(polygon_ids, dtype=torch.long),
    }

def eval_collate_fn_graph(batch):
    polygon, polygon_id = zip(*batch)
    return {
        "polygon": Batch.from_data_list(polygon),
        "polygon_id": torch.tensor(polygon_id, dtype=torch.long)
    }

def get_evaluation_dataloader(config, model_mode="graph"):
    k_eigvecs = config["dataset"].get("k_eigvecs", 10)
    dataset = EvaluationPolygonDataset(parquet_path=config["dataset"]["parquet_path"], model_mode=model_mode, k_eigvecs=k_eigvecs)
    eval_collate_fn = eval_collate_fn_graph if model_mode == "graph" else eval_collate_fn_seq

    return DataLoader(
        dataset,
        batch_size=config['dataset'].get("batch_size", 1024),
        shuffle=False,
        drop_last=False,
        collate_fn=eval_collate_fn,
        num_workers=config["dataset"].get("num_workers", 6),
        pin_memory=True,
        persistent_workers=config["dataset"].get("num_workers") > 0,
        prefetch_factor=config["dataset"].get("prefetch_factor", 2)
    )