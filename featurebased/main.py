import os
import gc
import yaml
import torch
import optuna
import mlflow
import psutil
import logging
import argparse
from copy import deepcopy
from datetime import datetime
from optuna.pruners import MedianPruner
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.importance import get_param_importances
from dataset import get_Dataloader
from PolygonMLP import PolygonPairClassifier
from trainer import MLPTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def setup_logger(model_name: str, log_dir: str = "logs", timestamp: str = None):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"{model_name}.log")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='a'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()

def flatten_config(config, sep='_'):
    flat_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            nested_flat_config = flatten_config(value, sep)
            for nested_key, nested_value in nested_flat_config.items():
                flat_config[f"{key}{sep}{nested_key}"] = nested_value
        else:
            flat_config[key] = value
    return flat_config


def run_trial(config: dict, trial: optuna.trial.Trial = None):
    try:
        config = deepcopy(config)

        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        if trial:
            config["tqdm_loader"] = False
            config["model"]["num_frequencies"] = trial.suggest_int("num_frequencies", 4, 10, step=2)
            config["model"]["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    
            config["hyperparameter"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            config["hyperparameter"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    
            # Parameter für die Layer-Suche
            min_layers, max_layers = 2, 5
            min_units, max_units, step = 32, 256, 32
    
            # Logik direkt hier mit List Comprehension integrieren
            n_layers = trial.suggest_int("n_hidden_layers", min_layers, max_layers)
            config["model"]["hidden_layers"] = [
                trial.suggest_int(f"n_units_l{i}", min_units, max_units, step=step) 
                for i in range(n_layers)
            ]

            # center = trial.suggest_float('center_freq', 600, 5000, log=True)
            # ratio = trial.suggest_float('ratio', 1.05, 10.0, log=True)
            # config["model"]["min_freq"] = center / ratio
            # config["model"]["max_freq"] = center * ratio

        torch.cuda.reset_peak_memory_stats()

        run_name_prefix = f"trial_{trial.number}" if trial else ""
        name = f"MLP_{timestamp}_{run_name_prefix}"

        os.makedirs(config["save_path"], exist_ok=True)
        model_save_path = os.path.join(config["save_path"], name + ".pt")

        with mlflow.start_run(run_name=name, nested=True if trial else False):
            mlflow.log_params(flatten_config(config))
            mlflow.set_tag("encoder", "MLP")

            logger.info("📦 Lade Dataloader...")

            train_loader, val_loader = get_Dataloader(config)
            config["dataset"]["train_len"] = len(train_loader)

            logger.info("🧠 Initialisiere Encoder...")
            model = PolygonPairClassifier(**config["model"])

            logger.info(f"Modellarchitektur:\n{model}")
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = total_params * 4 / (1024 ** 2)
            logger.info(f"📊 Gesamtparameter: {total_params:,}")
            logger.info(f"💾 Geschätzte Modellgröße: {model_size_mb:.2f} MB (float32)")
            mlflow.log_params({"total_params": total_params})
            mlflow.log_params({"model_size_mb": model_size_mb})


            trainer = MLPTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                logger=logger,
                device=device,
            )

            logger.info(f"Starting training on {device.type} for {config['num_epochs']} epochs.")


            best_val = float("inf")
            epochs_no_improve = 0


            for epoch in range(1, config["num_epochs"] + 1):
                results = trainer.train_epoch()
                results.update(trainer.evaluate())

                for key, value in results.items():
                    mlflow.log_metric(f"{key}", value, step=epoch)

                logger.info("🎯 Finale Evaluationsmetriken:")
                logger.info("=" * 40)
                for key, value in results.items():
                    logger.info(f"{key:<30}: {value:.4f}")
                logger.info("=" * 40)

                metric_name = "val_loss"
                current_val_metric = results.get(metric_name, "val_loss")
                if current_val_metric < best_val:

                    best_val = current_val_metric
                    epochs_no_improve = 0
                    torch.save(trainer.model.state_dict(), model_save_path)
                    logger.info(f"🎉 Neues bestes Modell für diesen Trial gespeichert (loss: {best_val:.4f})")
                    for key, value in results.items():
                        mlflow.log_metric(f"{key}_best", value, step=epoch)
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"Seit {epoch - epochs_no_improve}. Epoche keine Verbesserung [{epochs_no_improve}/{config['early_stopping']['patience']}]")

                if epochs_no_improve >= config["early_stopping"]["patience"]:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

                if trial:
                    trial.report(current_val_metric, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            logger.info(f"✅ Training für Trial abgeschlossen! Beste: {best_val:.4f}")

            return best_val


    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"Trial {trial.number} failed with CUDA Out of Memory. Pruning.")
            raise optuna.exceptions.TrialPruned()
        else:
            trial_num_str = f"trial {trial.number}" if trial else "the trial"
            logger.error(f"An unexpected RuntimeError occurred in trial {trial_num_str}: {e}")
            raise e
    finally:
        logger.info("🧹 Starte Cleanup nach Trial...")
        mlflow.end_run()

        trainer = None
        model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        gc.collect()
        logger.info(f"RAM used: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")

        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            logger.info("🧠 GPU-Speicherstatus:")
            logger.info(f"    🔹 Gesamt : {total_mem / 1024 ** 3:.2f} GB")
            logger.info(f"    🔸 Belegt : {used_mem / 1024 ** 3:.2f} GB")
            logger.info(f"    🔹 Frei   : {free_mem / 1024 ** 3:.2f} GB")

        logger.info("✅ Cleanup abgeschlossen.")


if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    args = parse_args()

    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    run_identifier = f"{timestamp}_MLP"
    if args.optuna:
        run_identifier += "_Optuna"

    logger = setup_logger(model_name=run_identifier)
    logger.info(f"🚀 Starting preparations for MLP model with '{config_path}'...")

    if args.optuna:
        logger.info("🚀 Starting hyperparameter optimization with Optuna...")

        def objective(trial: optuna.trial.Trial):
            return run_trial(
                config=config,
                trial=trial
            )

        if args.timestamp:
            timestamp = args.timestamp
            logger.info(f"Using provided timestamp for study name: {timestamp}")

        study_name = f"MLP_Optuna_{timestamp}"
        storage_url = f"sqlite:///./{study_name}.db"

        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=4)
        )

        with mlflow.start_run(run_name=f"Optuna_{run_identifier}"):
            try:
                study.optimize(objective, n_trials=args.n_trials)
            except KeyboardInterrupt:
                logger.warning("\nOptuna study manually stopped.")

        logger.info("\n" + "=" * 50)
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
            trial=None
        )
        logger.info("✅ Training complete!")