import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split

class PolygonPairDataset(Dataset):
    def __init__(self, lables, X=None, poly1=None, poly2=None):
        self.labels = lables.astype("int8")
        if poly1 is not None and poly2 is not None:
            self.poly1_feats = poly1.astype("float32")
            self.poly2_feats = poly2.astype("float32")
        else:
            self.poly1_feats = X[0].astype("float32")
            self.poly2_feats = X[1].astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.poly1_feats[idx], dtype=torch.float32),
            torch.tensor(self.poly2_feats[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.int8)
        )

def get_Dataloader(config):
    X_pairs = np.load(config["dataset"]["X_path"])
    y_pairs = np.load(config["dataset"]["y_path"])

    dataset = PolygonPairDataset(y_pairs, X=X_pairs)

    total_size = len(dataset)
    val_size = int(config["dataset"]["val_split"] * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["dataset"]["batch_size"], 
        shuffle=True,
        num_workers=config["dataset"].get("num_workers", 6),
        pin_memory=True,
        persistent_workers=config["dataset"].get("num_workers") > 0,
        prefetch_factor=config["dataset"].get("prefetch_factor", 2)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["dataset"]["batch_size"], 
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 6),
        pin_memory=True,
        persistent_workers=config["dataset"].get("num_workers") > 0,
        prefetch_factor=config["dataset"].get("prefetch_factor", 2)
    )

    return train_loader, val_loader


def inference(model, poly1, poly2):
    dataset = PolygonPairDataset(np.ones(len(poly1)), poly1=poly1, poly2=poly2)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_preds = []
    all_targets = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for poly1, poly2, targets in dataloader:
            # Forward-Pass
            outputs =  torch.sigmoid(model(poly1, poly2)) * 100
            predicted = (outputs > 50).float()

            # Ergebnisse sammeln
            all_preds.extend(predicted.cpu().view(-1).numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().view(-1).numpy())

    return all_preds, all_outputs