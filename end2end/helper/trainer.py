import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, OneCycleLR, LambdaLR, ReduceLROnPlateau


def embedding_variance_loss(embeddings):
    normalized_embs = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    variance = torch.mean(torch.var(normalized_embs, dim=0))
    return -variance # oder 1 / (variance + 1e-6)

class EmbeddingTrainer:
    def __init__(self, encoder, config, logger, acc_at_k_loader, model_mode="graph", device="cpu"):
        self.device = device
        self.encoder = encoder
        self.config = config
        self.logger = logger
        self.model_mode = model_mode
        self.accumulation_steps = config.get("accumulation_steps", 1)

        self.total_epochs = config["num_epochs"]
        self.active_miner=config.get("active_miner", False)
        self.calculate_threshold = config.get("calculate_threshold", True)
        self.calculate_accuracy = config.get("calculate_accuracy", True)

        self.acc_at_k_loader = acc_at_k_loader

        # Optimizer mit Fused für H100
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=config["hyperparameter"]["lr"],
            fused=True,
            weight_decay=config["hyperparameter"]["weight_decay"]
        )

        cosine_distance = CosineSimilarity()

        if config["miner"]["miner_type"] == "multi":
            self.miner = miners.MultiSimilarityMiner(
                epsilon=self.config["miner"].get("epsilon", 0.1)
                )

            self.metric_criterion = losses.MultiSimilarityLoss(
                alpha=self.config["miner"].get("alpha",1),
                beta=self.config["miner"].get("beta", 20),
                base=self.config["miner"].get("base", 0.5),
                distance=cosine_distance
            )
        elif config["miner"]["miner_type"] == "ntxent":
            self.miner = None # NT-Xent benötigt keinen Miner
            self.metric_criterion = losses.NTXentLoss(
                temperature=self.config["miner"].get("temperature", 0.07),
                # distance=cosine_similarity # Wichtig: NTXentLoss verwendet Ähnlichkeit!
            )
        else:
            self.miner = miners.TripletMarginMiner(
                margin=self.config["miner"].get("triplet_margin", 0.3),
                type_of_triplets=self.config["miner"].get("type_of_triplets", "semihard"),
            )
            self.metric_criterion = losses.TripletMarginLoss(
                margin=self.config["miner"].get("triplet_margin", 0.3),
                distance=cosine_distance
            )

        # Scheduler-Setup
        steps_per_epoch = config["dataset"]["train_len"] // config.get("accumulation_steps", 1)
        total_steps = steps_per_epoch * config["num_epochs"]
        warmup_steps = int(config.get("scheduler", {}).get("warm_up", 0.1) * total_steps)
        lr = config["hyperparameter"]["lr"]

        scheduler_type = config["scheduler"]["type"]
        num_epochs = config["num_epochs"]
        warmup_fraction = config["scheduler"].get("warm_up", 0.1)
        warmup_epochs = int(warmup_fraction * num_epochs)

        if scheduler_type == "cosine":
            # Phase 1: Linear Warmup
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                total_iters=warmup_epochs
            )

            # Phase 2: Cosine Annealing
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=1e-6
            )

            # Kombiniert beide Scheduler
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

        elif scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=total_steps,
                anneal_strategy='cos',
                pct_start=warmup_fraction,
                div_factor=25.0,
                final_div_factor=1e4
            )

        elif scheduler_type == "constant":
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.2,
                patience=3,
                min_lr=1e-7,
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self.scaler = GradScaler(
            enabled=self.device.type == "cuda",
            init_scale=2.0 ** 16,
            growth_factor=2.0,
            backoff_factor=0.5
        )

    def train_epoch(self, dataloader):
        """Führt eine einzelne Trainingsepoche mit Gradient Accumulation und Mixed Precision durch."""
        self.encoder.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        loader = tqdm(dataloader, desc="Training", leave=False) if self.config["tqdm_loader"] else dataloader

        for batch_idx, batch in enumerate(loader):
            if self.model_mode == "graph":
                keys = ["poly1", "poly2", "poly1_i", "poly2_i"]
                poly1, poly2, poly1_i, poly2_i = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                model_inputs = [(poly1,), (poly2,)]
            else:
                keys = ["poly1", "poly2", "poly1_mask", "poly2_mask", "poly1_i", "poly2_i"]
                poly1, poly2, poly1_mask, poly2_mask, poly1_i, poly2_i = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                model_inputs = [(poly1, poly1_mask), (poly2, poly2_mask)]

            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled(), dtype=torch.bfloat16):
                emb1 = self.encoder(*model_inputs[0])
                emb2 = self.encoder(*model_inputs[1])

                all_embs = torch.cat([emb1, emb2], dim=0)

                miner_labels = torch.cat([poly1_i, poly2_i], dim=0)
                if self.miner is None:
                    loss = self.metric_criterion(all_embs, miner_labels)
                else:
                    hard_pairs = self.miner(all_embs, miner_labels)
                    loss = self.metric_criterion(all_embs, miner_labels, hard_pairs)


                # variance_loss = embedding_variance_loss(all_embs)
                # loss += self.config["hyperparameter"].get("variance_weight", 0.01) * variance_loss

                loss = loss / self.accumulation_steps


            # Backward Pass (skaliert)
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.scheduler is not None and isinstance(self.scheduler, OneCycleLR):
                if self.scheduler.last_epoch < self.scheduler.total_steps:
                    self.scheduler.step()

            total_loss += loss.detach().cpu().item() * self.accumulation_steps

            if self.config["tqdm_loader"] and batch_idx % 200 == 0:
                loader.set_postfix({
                    "total_loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate(self, dataloader, k_values=None):
        self.encoder.eval()
        all_pos_distances, all_neg_distances = [], []

        loader = tqdm(dataloader, desc="Evaluating", leave=False) if self.config.get("tqdm_loader") else dataloader

        with torch.no_grad():
            for batch in loader:
                if self.model_mode == "graph":
                    keys = ["poly1", "poly2", "poly1_i", "poly2_i"]
                    poly1, poly2, poly1_i, poly2_i = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                    model_inputs = [(poly1,), (poly2,)]
                else:
                    keys = ["poly1", "poly2", "poly1_mask", "poly2_mask", "poly1_i", "poly2_i"]
                    poly1, poly2, poly1_mask, poly2_mask, poly1_i, poly2_i = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                    model_inputs = [(poly1, poly1_mask), (poly2, poly2_mask)]

                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled(), dtype=torch.bfloat16):
                    emb1 = self.encoder(*model_inputs[0])
                    emb2 = self.encoder(*model_inputs[1])

                    # Berechne die Distanz für jedes Paar
                    distances = 1 - torch.cosine_similarity(emb1, emb2)
                    # distances = torch.nn.functional.pairwise_distance(emb1, emb2, p=2) # Falls du L2-Distanz willst

                # Trenne die Distanzen nach positiven und negativen Paaren
                all_pos_distances.append(distances[poly1_i == poly2_i])
                all_neg_distances.append(distances[poly1_i != poly2_i])

        # Schritt 2: Metriken berechnen
        metrics = {}
        if not all_pos_distances and not all_neg_distances:
            return metrics

        pos_dists = torch.cat(all_pos_distances) if all_pos_distances else torch.tensor([], device=self.device)
        neg_dists = torch.cat(all_neg_distances) if all_neg_distances else torch.tensor([], device=self.device)

        # Berechne durchschnittliche Distanzen
        metrics["Average_Positive_Distance"] = pos_dists.mean().item() if pos_dists.numel() > 0 else 0
        metrics["Average_Negative_Distance"] = neg_dists.mean().item() if neg_dists.numel() > 0 else 0
        metrics["Average_Margin"] = metrics["Average_Negative_Distance"] - metrics["Average_Positive_Distance"]

        # Schritt 3: Optimalen Threshold und zugehörige Metriken berechnen
        if self.calculate_threshold and pos_dists.numel() > 0 and neg_dists.numel() > 0:
            self.logger.info("Calculating best threshold...")
            threshold_metrics = self._calculate_best_threshold_metrics(pos_dists.cpu(), neg_dists.cpu())
            metrics.update(threshold_metrics)

        if self.calculate_accuracy:
            accuracy_at_k_metrics = self._calculate_accuracy_at_k(k_values)
            metrics.update(accuracy_at_k_metrics)

        return metrics

    def _calculate_best_threshold_metrics(self, pos_dists: torch.Tensor, neg_dists: torch.Tensor):
        all_dists = torch.cat([pos_dists, neg_dists])
        all_labels = torch.cat([torch.ones_like(pos_dists), torch.zeros_like(neg_dists)])

        sorted_indices = torch.argsort(all_dists)
        sorted_dists = all_dists[sorted_indices]
        sorted_labels = all_labels[sorted_indices]

        # Zähle TP, FP, etc. effizient
        tps = torch.cumsum(sorted_labels, dim=0)
        fps = torch.cumsum(1 - sorted_labels, dim=0)

        num_pos = pos_dists.size(0)
        num_neg = neg_dists.size(0)

        fns = num_pos - tps
        tns = num_neg - fps

        # Metriken berechnen
        eps = 1e-8
        accuracy = (tps + tns) / (num_pos + num_neg)
        precision = tps / (tps + fps + eps)
        recall = tps / (tps + fns + eps)
        f1_scores = 2 * (precision * recall) / (precision + recall + eps)

        # Beste Indizes finden
        best_acc_idx = torch.argmax(accuracy)
        best_f1_idx = torch.argmax(f1_scores)

        # Threshold als Mittelpunkt zwischen den Werten für mehr Robustheit
        def get_threshold(idx):
            if idx == 0:
                return sorted_dists[0].item() - eps
            return (sorted_dists[idx - 1].item() + sorted_dists[idx].item()) / 2

        return {
            "best_threshold_acc": get_threshold(best_acc_idx),
            "accuracy_at_best_threshold": accuracy[best_acc_idx].item(),
            "best_threshold_f1": get_threshold(best_f1_idx),
            "f1_at_best_threshold": f1_scores[best_f1_idx].item(),
        }

    def _calculate_accuracy_at_k(self, k_values=None):
        if k_values is None:
            k_values = [5, 10, 25, 50]
        k_values = sorted(k_values)
        max_k = max(k_values)
        correct_at_k = {k: 0 for k in k_values}
        all_ids = []
        all_embeddings = []

        dataloader_acc = tqdm(self.acc_at_k_loader, desc="Encoding", leave=False) if self.config["tqdm_loader"] else self.acc_at_k_loader
        for batch in dataloader_acc:

            if self.model_mode == "graph":
                    keys = ["polygons", "polygon_ids"]
                    polygon, ids = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                    model_inputs = (polygon,)
            else:
                keys = ["polygons", "masks", "polygon_ids"]
                polygon, masks, ids = [batch[_].to(self.device, non_blocking=True) for _ in keys]
                model_inputs = (polygon, masks)

            with torch.no_grad():
                emb = self.encoder(*model_inputs)

            all_embeddings.append(emb)
            all_ids.extend(ids)

        embeddings = torch.cat(all_embeddings, dim=0)
        ids = torch.tensor(all_ids, device=self.device)

        candidates = embeddings
        candidate_ids = ids

        eval_bs = self.config["dataset"].get("eval_batch_size", 1024)
        total = len(embeddings)

    
        index_range = tqdm(range(0, total, 1024), desc="Search", leave=False) \
            if self.config.get("tqdm_loader") else range(0, total, eval_bs)

        for i in index_range:
            end = min(i + eval_bs, total)
            anchor = embeddings[i:end]
            anchor_ids = ids[i:end]

            sim = torch.matmul(anchor, candidates.T)
            dist = 1 - sim

            batch_size = anchor.size(0)
            mask = torch.zeros_like(dist, dtype=torch.bool)
            mask[torch.arange(batch_size), torch.arange(i, end)] = True
            dist.masked_fill_(mask, float('inf'))

            _, topk_idx = torch.topk(dist, k=max_k, dim=1, largest=False)
            topk_ids = candidate_ids[topk_idx]

            anchor_ids_expanded = anchor_ids.unsqueeze(1)
            for k in k_values:
                match = (topk_ids[:, :k] == anchor_ids_expanded).any(dim=1)
                correct_at_k[k] += match.sum().item()

        return {f"Accuracy_{k}": correct_at_k[k] / total for k in k_values}

