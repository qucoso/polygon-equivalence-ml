import os
import gc
import yaml
import torch
import logging
import argparse
import mlflow
import optuna
import psutil
import shutil
from optuna.pruners import MedianPruner
from optuna.importance import get_param_importances
from copy import deepcopy
from datetime import datetime

from end2end.helper.trainer import EmbeddingTrainer
from end2end.helper.dataset import get_dataloader, get_evaluation_dataloader
from end2end.helper.architectures.graph import GraphPolygonEncoder
from end2end.helper.architectures.perceiver import PolygonPerceiver


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sequence",
                        help="'sequence' or 'graph'.")
    parser.add_argument("--optuna", action="store_true",
                        help="Enable Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials.")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp to reuse an existing Optuna study DB.")
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        help="path of config file")
    return parser.parse_args()


# =============================================================================
# Logging setup
# =============================================================================

def setup_logger(model_name: str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{model_name}.log")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger()


# =============================================================================
# Helpers
# =============================================================================

def flatten_config(config: dict, sep: str = "_") -> dict:
    """Recursively flatten nested dicts into a single-level dict for MLflow logging."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for nested_key, nested_value in flatten_config(value, sep).items():
                flat[f"{key}{sep}{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


def build_encoder(model_mode: str, config: dict):
    """Instantiate the encoder based on model_mode and the corresponding config section."""
    if model_mode == "graph":
        if "graph_encoder" not in config:
            raise ValueError("Config is missing 'graph_encoder' section for graph mode.")
        return GraphPolygonEncoder(**config["graph_encoder"])
    else:
        if "perceiver_encoder" not in config:
            raise ValueError("Config is missing 'perceiver_encoder' section for sequence mode.")
        return PolygonPerceiver(**config["perceiver_encoder"])


# =============================================================================
# Single training run (also used as Optuna objective)
# =============================================================================

def run_trial(
    config: dict,
    model_mode: str,
    trial: optuna.trial.Trial = None,
    gpu_id: int = 0,
    timestamp: str = None,
):
    trainer = None
    encoder = None
    train_loader = None
    val_loader = None
    acc_at_k_loader = None

    try:
        config = deepcopy(config)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # ---- Optuna: suggest hyperparameters ----
        if trial:
            config["tqdm_loader"] = False

            if model_mode == "graph":
                # Example
                config["graph_encoder"]["hidden_dim"] = trial.suggest_categorical(
                    "hidden_dim", [32, 64, 128]
                )
                

        torch.cuda.reset_peak_memory_stats()

        # ---- Run naming ----
        run_name_prefix = f"trial_{trial.number}" if trial else ""
        name = f"{model_mode}_{timestamp}_{run_name_prefix}".rstrip("_")

        os.makedirs(config["save_path"], exist_ok=True)
        model_save_path = os.path.join(config["save_path"], name + ".pt")

        # ---- Remove the encoder config section that is not needed ----
        loggable_config = deepcopy(config)
        if model_mode == "graph":
            loggable_config.pop("perceiver_encoder", None)
        else:
            loggable_config.pop("graph_encoder", None)

        with mlflow.start_run(run_name=name, nested=bool(trial)):
            mlflow.log_params(flatten_config(loggable_config))
            mlflow.set_tag("encoder", model_mode)

            # ---- Data ----
            logger.info("📦 Loading dataloaders...")
            train_loader, val_loader = get_dataloader(config, model_mode=model_mode)
            acc_at_k_loader = get_evaluation_dataloader(config, model_mode=model_mode)
            config["dataset"]["train_len"] = len(train_loader)

            # ---- Model ----
            logger.info("🧠 Initialising encoder...")
            encoder = build_encoder(model_mode, config)

            logger.info(f"Model architecture:\n{encoder}")
            total_params = sum(p.numel() for p in encoder.parameters())
            trainable_params = sum(
                p.numel() for p in encoder.parameters() if p.requires_grad
            )
            model_size_mb = total_params * 4 / (1024**2)
            logger.info(f"📊 Total parameters:     {total_params:,}")
            logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
            logger.info(f"💾 Estimated size:       {model_size_mb:.2f} MB (float32)")
            mlflow.log_params(
                {
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "model_size_mb": round(model_size_mb, 2),
                }
            )

            encoder = encoder.to(device)
            trainer = EmbeddingTrainer(
                encoder,
                config=config,
                logger=logger,
                acc_at_k_loader=acc_at_k_loader,
                model_mode=model_mode,
                device=device,
            )

            logger.info(f"👟 Training started on device: {device}")

            best_val = -float("inf")
            epochs_no_improve = 0

            # ---- Training loop ----
            for epoch in range(1, config["num_epochs"] + 1):
                logger.info(f"Epoch [{epoch}/{config['num_epochs']}]")

                avg_loss = trainer.train_epoch(train_loader)
                val_results = trainer.evaluate(
                    dataloader=val_loader, k_values=[1, 3, 10]
                )

                # Step the scheduler depending on its type
                if isinstance(
                    trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    trainer.scheduler.step(
                        val_results.get("f1_at_best_threshold", 0.0)
                    )
                elif not isinstance(
                    trainer.scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    trainer.scheduler.step()

                # Log metrics
                mlflow.log_metric("train/avg_loss", avg_loss, step=epoch)
                for key, value in val_results.items():
                    mlflow.log_metric(f"val/{key}", value, step=epoch)

                logger.info("🎯 Evaluation metrics:")
                logger.info("=" * 40)
                logger.info(f"{'train_loss':<30}: {avg_loss:.4f}")
                for key, value in val_results.items():
                    logger.info(f"{key:<30}: {value:.4f}")
                logger.info("=" * 40)

                # ---- Early stopping & checkpointing ----
                primary_metric = "f1_at_best_threshold"
                metric_key = (
                    primary_metric
                    if primary_metric in val_results
                    else "Average_Margin"
                )
                current_val_metric = val_results.get(metric_key, 0.0)

                if current_val_metric > best_val:
                    best_val = current_val_metric
                    epochs_no_improve = 0
                    torch.save(trainer.encoder.state_dict(), model_save_path)
                    logger.info(
                        f"🎉 New best model saved ({metric_key}: {best_val:.4f})"
                    )
                    for key, value in val_results.items():
                        mlflow.log_metric(f"{key}_best", value, step=epoch)
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"No improvement since epoch {epoch - epochs_no_improve} "
                        f"[{epochs_no_improve}/{config['early_stopping']['patience']}]"
                    )

                if epochs_no_improve >= config["early_stopping"]["patience"]:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

                # Optuna pruning check
                if trial:
                    trial.report(current_val_metric, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            logger.info(
                f"✅ Training completed. Best {metric_key}: {best_val:.4f}"
            )
            return best_val

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "CUDA out of memory" in str(e):
            logger.warning(
                f"Trial {trial.number if trial else '?'} failed: CUDA OOM. Pruning."
            )
            raise optuna.exceptions.TrialPruned()
        else:
            logger.error(f"Unexpected RuntimeError: {e}")
            raise e

    finally:
        # ---- Cleanup: free memory after every trial / run ----
        logger.info("🧹 Cleanup...")
        mlflow.end_run()

        del trainer, encoder, train_loader, val_loader, acc_at_k_loader
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            free_mem, total_mem = torch.cuda.mem_get_info()
            logger.info(
                f"GPU mem — total: {total_mem / 1024**3:.2f} GB, "
                f"used: {(total_mem - free_mem) / 1024**3:.2f} GB, "
                f"free: {free_mem / 1024**3:.2f} GB"
            )

        logger.info(
            f"RAM used: {psutil.virtual_memory().used / (1024**3):.2f} GB"
        )
        logger.info("✅ Cleanup complete.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    args = parse_args()
    model_mode = args.model.lower()
    config_path = args.config_path
    assert model_mode in ["sequence", "graph"], "model must be 'sequence' or 'graph'"

    config_path = os.path.join(BASE_DIR, config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = "Location Encodings"
    mlflow.set_experiment(experiment_name)
    timestamp = args.timestamp or datetime.now().strftime("%m%d_%H%M")

    run_identifier = f"{timestamp}_{model_mode}"
    if args.optuna:
        run_identifier += "_Optuna"

    logger = setup_logger(model_name=run_identifier)
    logger.info(
        f"🚀 Starting {model_mode} model with config '{config_path}'..."
    )

    # ==== Optuna hyperparameter search ====
    if args.optuna:
        logger.info("🔍 Starting Optuna hyperparameter optimization...")

        def objective(trial: optuna.trial.Trial):
            return run_trial(
                config=config,
                model_mode=model_mode,
                gpu_id=0,
                trial=trial,
                timestamp=timestamp,
            )

        study_name = f"{model_mode}_Optuna_{timestamp}"
        db_filename = f"{study_name}.db"

        # Use a temporary directory for the SQLite DB during optimization
        tmp_dir = os.environ.get("TMPDIR", "/tmp")
        tmp_db_path = os.path.join(tmp_dir, db_filename)
        final_db_path = os.path.join(os.getcwd(), db_filename)

        if os.path.exists(final_db_path):
            logger.info(f"Found existing DB at {final_db_path}, copying to tmp...")
            shutil.copy2(final_db_path, tmp_db_path)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=f"sqlite:///{tmp_db_path}",
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        )

        try:
            with mlflow.start_run(run_name=f"Optuna_{run_identifier}"):
                try:
                    study.optimize(objective, n_trials=args.n_trials)
                except KeyboardInterrupt:
                    logger.warning("Optuna study manually interrupted.")
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise
        finally:
            # Copy the DB back from tmp to the working directory
            if os.path.exists(tmp_db_path):
                try:
                    shutil.copy2(tmp_db_path, final_db_path)
                    logger.info(f"✅ DB saved to: {final_db_path}")
                    os.remove(tmp_db_path)
                except Exception as e:
                    logger.error(
                        f"❌ Failed to copy DB back (still at {tmp_db_path}): {e}"
                    )
            else:
                logger.warning("No temporary DB file found to copy back.")

        # Summary
        logger.info("=" * 50)
        logger.info("Optuna study complete!")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        logger.info("🏆 Best Trial:")
        logger.info(f"  Value (Best Validation Accuracy): {best_trial.value:.4f}")
        logger.info("  Best Parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        try:
            logger.info("📊 Parameter Importances (most important first):")
            importances = get_param_importances(study)
            for param, importance in importances.items():
                logger.info(f"    {param}: {importance:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate parameter importances: {e}")

    else:
        # --- Single Training Run ---
        logger.info("👟 Starting a single training run on GPU 0...")
        run_trial(
            config=config,
            model_mode=model_mode,
            gpu_id=0,
            trial=None,
            timestamp=timestamp
        )
        logger.info("✅ Training complete!")
