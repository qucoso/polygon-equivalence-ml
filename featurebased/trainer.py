import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support

class MLPTrainer:
    def __init__(self, model, config, train_loader, val_loader, logger, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.use_amp = self.config.get("use_amp", False)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["hyperparameter"]["lr"],
            weight_decay=config["hyperparameter"]["weight_decay"],
            fused=(self.device.type == "cuda")
        )

        self.criterion = nn.BCEWithLogitsLoss()

        self.scaler = GradScaler(
            enabled=(self.use_amp and self.device.type == "cuda"),
            init_scale=2.0 ** 16,
            growth_factor=2.0,
            backoff_factor=0.5
        )

        self._init_scheduler()

    def _init_scheduler(self):
            scheduler_type = self.config["scheduler"]["type"]
            num_epochs = self.config["num_epochs"]

            if scheduler_type == "onecycle":
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.config["hyperparameter"]["lr"],
                    total_steps=num_epochs * len(self.train_loader),
                    pct_start=self.config["scheduler"].get("warm_up", 0.1),
                    anneal_strategy='cos',
                    div_factor=25.0,
                    final_div_factor=1e4
                )
            elif scheduler_type == "reduceonplateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min', factor=0.01, patience=2,
                    threshold=1e-5, threshold_mode='rel',
                    cooldown=0, min_lr=0.0, eps=1e-8, 
                )
            else:
                self.scheduler = None


    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_targets = [], []

        loader = tqdm(self.train_loader, desc="Training", leave=False) if self.config["tqdm_loader"] else self.train_loader

        for batch_idx, (poly1, poly2, targets) in enumerate(loader):
            poly1, poly2, targets = poly1.to(self.device), poly2.to(self.device), targets.to(self.device)
            targets_float = targets.float().view(-1, 1)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp and self.scaler.is_enabled()):
                outputs = self.model(poly1, poly2)
                loss = self.criterion(outputs, targets_float)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            predicted = (torch.sigmoid(outputs) > 0.5).int()
            total_loss += loss.item() * targets.size(0)
            total_correct += (predicted.view(-1) == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predicted.cpu().view(-1).numpy())
            all_targets.extend(targets.cpu().numpy())

            if self.config["tqdm_loader"] and batch_idx % 200 == 0:
                loader.set_postfix({
                    "total_loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        _, _, avg_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)

        return {"train_loss": avg_loss, "train_acc": avg_acc, "train_f1": avg_f1}

    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_targets = [], []

        loader = tqdm(self.val_loader, desc="Evaluating", leave=False) if self.config.get("tqdm_loader") else self.val_loader

        with torch.no_grad():
            for poly1, poly2, targets in loader:
                poly1, poly2, targets = poly1.to(self.device), poly2.to(self.device), targets.to(self.device)
                targets_float = targets.float().view(-1, 1)

                with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp and self.scaler.is_enabled()):
                    outputs = self.model(poly1, poly2)
                    loss = self.criterion(outputs, targets_float)

                predicted = (torch.sigmoid(outputs) > 0.5).int()
                total_loss += loss.item() * targets.size(0)
                total_correct += (predicted.view(-1) == targets).sum().item()
                total_samples += targets.size(0)

                all_preds.extend(predicted.cpu().view(-1).numpy())
                all_targets.extend(targets.cpu().numpy())


        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        _, _, avg_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return {"val_loss": avg_loss, "val_acc": avg_acc, "val_f1": avg_f1}