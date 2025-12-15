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

from helper.trainer import EmbeddingTrainer
from helper.dataset import get_dataloader, get_evaluation_dataloader

from helper.architectures.graph import GraphPolygonEncoder
from helper.architectures.perceiver import PolygonPerceiver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sequence", help="sequence or graph. Default: sequence.")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--resume_run_id", type=str, default=None, help="MLflow run ID to resume training from.")
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


def load_encoder_from_mlflow(run_id: str, model_path: str, logger: logging.Logger):
    try:
        logger.info(f"🔄 Lade Modell aus MLflow Run: {run_id}")

        # 1. Daten vom MLflow Run abrufen
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = run.data.tags.get('mlflow.runName', '')

        encoder_kwargs = {}
        encoder_type = None

        # 2. Encoder-Typ bestimmen und Parameter extrahieren
        if 'graph' in run_name.lower():
            logger.info("🧠 Graph-Encoder-Architektur wird wiederhergestellt...")
            encoder_type = 'graph'
            prefix = 'graph_encoder_'
            encoder_kwargs = {
                "hidden_dim": int(params.get(f'{prefix}hidden_dim', 128)),
                "embedding_dim": int(params.get(f'{prefix}embedding_dim', 128)),
                "num_heads": int(params.get(f'{prefix}num_heads', 4)),
                "num_layers": int(params.get(f'{prefix}num_layers', 4)),
                "dropout": float(params.get(f'{prefix}dropout', 0.1)),
                "pooling_strategy": params.get(f'{prefix}pooling_strategy', 'attention'),
                "loc_encoding_dim": int(params.get(f'{prefix}loc_encoding_dim', 128)),
                "loc_encoding_type": params.get(f'{prefix}loc_encoding_type', 'multiscale_learnable'),
                "loc_encoding_min_freq": float(params.get(f'{prefix}loc_encoding_min_freq', 1000.0)),
                "loc_encoding_max_freq": float(params.get(f'{prefix}loc_encoding_max_freq', 5600.0)),
                "graph_encoder_type": params.get(f'{prefix}graph_encoder_type', 'gat'),
                "use_edge_attr": str(params.get(f'{prefix}use_edge_attr', 'False')).lower() == 'true',
                "lap_pe_k": int(params.get(f'{prefix}lap_pe_k', 10)),
            }
            encoder = GraphPolygonEncoder(**encoder_kwargs)

        elif 'sequence' in run_name.lower() or 'sequenz' in run_name.lower():
            logger.info("🧠 Perceiver-Encoder-Architektur wird wiederhergestellt...")
            encoder_type = 'perceiver'
            prefix = 'perceiver_encoder_'
            encoder_kwargs = {
                "d_model": int(params.get(f'{prefix}d_model', 64)),
                "d_latents": int(params.get(f'{prefix}d_latents', 64)),
                "num_latents": int(params.get(f'{prefix}num_latents', 12)),
                "num_heads": int(params.get(f'{prefix}num_heads', 4)),
                "num_cross_layers": int(params.get(f'{prefix}num_cross_layers', 1)),
                "num_self_layers": int(params.get(f'{prefix}num_self_layers', 4)),
                "loc_encoding_dim": int(params.get(f'{prefix}loc_encoding_dim', 8)),
                "loc_encoding_type": params.get(f'{prefix}loc_encoding_type', 'multiscale_learnable'),
                "loc_encoding_min_freq": float(params.get(f'{prefix}loc_encoding_min_freq', 1000.0)),
                "loc_encoding_max_freq": float(params.get(f'{prefix}loc_encoding_max_freq', 5600.0)),
                "d_pos_enc": int(params.get(f'{prefix}d_pos_enc', 16)),
                "embedding_dim": int(params.get(f'{prefix}embedding_dim', 128)),
                "dim_feedforward_factor": int(params.get(f'{prefix}dim_feedforward_factor', 4)),
                "num_pool_queries": int(params.get(f'{prefix}num_pool_queries', 16)),
                "latent_dropout": float(params.get(f'{prefix}latent_dropout', 0.1)),
                "use_layernorm_inputs": str(params.get(f'{prefix}use_layernorm_inputs', 'True')).lower() == 'true',
            }
            encoder = PolygonPerceiver(**encoder_kwargs)

        else:
            logger.warning(f"❌ Fehler: Konnte den Encoder-Typ aus dem Run-Namen '{run_name}' nicht bestimmen.")
            return None

        logger.info("✅ Modellarchitektur erfolgreich erstellt.")
        logger.info("Parameter:", encoder_kwargs)

        checkpoint_path = os.path.join(model_path, f"{run_name}.pt")
        if not os.path.exists(checkpoint_path):
            logger.warning(f"❌ Fehler: Checkpoint-Datei nicht gefunden unter: {checkpoint_path}")
            return None

        logger.info(f"💾 Lade Gewichte von: {checkpoint_path}")
        # `map_location` stellt sicher, dass es auch ohne GPU funktioniert
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        encoder.load_state_dict(checkpoint)
        encoder.train()  # Modell in den Evaluationsmodus setzen

        logger.info(f"✨ Modell '{run_name}' erfolgreich geladen und ist einsatzbereit!")
        return encoder

    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return None


def run_trial(config: dict, model_mode:str, trial: optuna.trial.Trial = None, gpu_id: int = 0,
              timestamp: str = None, resume_model_id=None):
    try:
        config = deepcopy(config)
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        if trial:
            config["tqdm_loader"] = False
            # config["hyperparameter"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
            # config["hyperparameter"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            # config["scheduler"]["type"] = trial.suggest_categorical("scheduler_type", ["onecycle", "cosine"])
            # config["scheduler"]["warm_up"] = trial.suggest_float("warm_up", 0.01, 0.3, step=0.03)

            # config["miner"]["triplet_margin"] = trial.suggest_float("triplet_margin", 0.2, 1.5, step=0.1)
            # config["miner"]["type_of_triplets"] = trial.suggest_categorical(
            #     "type_of_triplets", ["semihard", "hard", "all"]
            #     )

            if model_mode == "graph" :
                # pass
                # config["graph_encoder"]["graph_encoder_type"] = trial.suggest_categorical("graph_encoder_type", ["gin", "gat", "mp"])
                config["graph_encoder"]["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
                config["graph_encoder"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 96])
                config["graph_encoder"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [32, 64, 128])
                config["graph_encoder"]["num_layers"] = trial.suggest_int("num_layers", 2, 10)
                config["graph_encoder"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
                config["graph_encoder"]["loc_encoding_dim"] = trial.suggest_categorical("loc_encoding_dim", [8, 16, 32, 64])
                config["dataset"]["lap_pe_k"] = trial.suggest_categorical("lap_pe_k", [5, 7, 10])
            else:
                d_arch = trial.suggest_categorical("d_arch", [64, 96])
                config["perceiver_encoder"]["d_model"] = d_arch
                config["perceiver_encoder"]["d_latents"] = d_arch

                config["perceiver_encoder"]["num_heads"] = trial.suggest_categorical("num_heads", [4, 8, 16, 32])
                config["perceiver_encoder"]["num_latents"] = trial.suggest_int("num_latents", 6, 16)
                config["perceiver_encoder"]["num_self_layers"] = trial.suggest_int("num_self_layers", 2, 5)
                # config["perceiver_encoder"]["num_cross_layers"] = trial.suggest_int("num_cross_layers", 1, 2)
                # config["perceiver_encoder"]["num_pool_queries"] = trial.suggest_categorical("num_pool_queries", [8, 16, 32, 64])
                config["perceiver_encoder"]["d_pos_enc"] = trial.suggest_categorical("d_pos_enc", [8, 16, 32])
                config["perceiver_encoder"]["loc_encoding_dim"] = trial.suggest_categorical("loc_encoding_dim", [8, 16, 32])
                config["perceiver_encoder"]["embedding_dim"] = trial.suggest_categorical("embedding_dim", [64, 96, 128])
                config["perceiver_encoder"]["latent_dropout"] = trial.suggest_float("latent_dropout", 0.005, 0.15)

        torch.cuda.reset_peak_memory_stats()

        run_name_prefix = f"trial_{trial.number}" if trial else ""
        name = f"{model_mode}_{timestamp}_{run_name_prefix}"

        os.makedirs(config["save_path"], exist_ok=True)
        model_save_path = os.path.join(config["save_path"], name + ".pt")

        with mlflow.start_run(run_name=name, nested=True if trial else False):
            if model_mode == "graph":
                del config['perceiver_encoder']
            else:
                del config['graph_encoder']
            mlflow.log_params(flatten_config(config))
            mlflow.set_tag("encoder", model_mode)

            logger.info("📦 Lade Dataloader...")

            train_loader, val_loader = get_dataloader(config, model_mode=model_mode)
            acc_at_k_loader = get_evaluation_dataloader(config, model_mode=model_mode)

            config["dataset"]["train_len"] = len(train_loader)

            if resume_model_id:
                encoder = load_encoder_from_mlflow(
                    run_id=resume_model_id,
                    model_path=config["save_path"],
                    logger=logger
                )
            else:
                logger.info("🧠 Initialisiere Encoder...")
                if model_mode == "graph":
                    encoder = GraphPolygonEncoder(**config["graph_encoder"])
                else:
                    encoder = PolygonPerceiver(**config["perceiver_encoder"])

            logger.info(f"Modellarchitektur:\n{encoder}")
            total_params = sum(p.numel() for p in encoder.parameters())
            trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024**2)
            logger.info(f"📊 Gesamtparameter: {total_params:,}")
            logger.info(f"🎯 Trainierbare Parameter: {trainable_params:,}")
            logger.info(f"💾 Geschätzte Modellgröße: {model_size_mb:.2f} MB (float32)")
            mlflow.log_params({"total_params": total_params})
            mlflow.log_params({"trainable_params": trainable_params})
            mlflow.log_params({"model_size_mb": model_size_mb})

            encoder = encoder.to(device)
            
            trainer = EmbeddingTrainer(
                encoder,
                config=config,
                logger=logger,
                acc_at_k_loader=acc_at_k_loader,
                model_mode=model_mode,
                device=device
            )

            logger.info(f"👟 Training gestartet auf Device: {device}")

            best_val = -float("inf")
            epochs_no_improve = 0

            for epoch in range(1, config["num_epochs"] + 1):
                logger.info(f"Epoch [{epoch}/{config['num_epochs']}]")

                avg_loss = trainer.train_epoch(train_loader)
                val_results = trainer.evaluate(
                    dataloader=val_loader,
                    k_values=[1, 3, 10],
                )

                if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_results.get("f1_at_best_threshold", 0.0))
                elif not isinstance(trainer.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    trainer.scheduler.step()

                # Logge Metriken für MLflow
                mlflow.log_metric("train/avg_loss", avg_loss, step=epoch)
                for key, value in val_results.items():
                    mlflow.log_metric(f"val/{key}", value, step=epoch)

                logger.info("🎯 Finale Evaluationsmetriken:")
                logger.info("=" * 40)
                logger.info(f"{'train_loss':<30}: {avg_loss:.4f}")
                for key, value in val_results.items():
                    logger.info(f"{key:<30}: {value:.4f}")
                logger.info("=" * 40)

                metric_name = "f1_at_best_threshold"
                metric = metric_name if metric_name in val_results.keys() else "Average_Margin"
                current_val_metric = val_results.get(metric, 0.0)
                if current_val_metric > best_val:
                    best_val = current_val_metric
                    epochs_no_improve = 0
                    torch.save(trainer.encoder.state_dict(), model_save_path)
                    logger.info(f"🎉 Neues bestes Modell für diesen Trial gespeichert ({metric}: {best_val:.4f})")
                    for key, value in val_results.items():
                        mlflow.log_metric(f"{key}_best", value, step=epoch)
                else:
                    epochs_no_improve += 1
                    logger.info(f"Seit {epoch - epochs_no_improve}. Epoche keine Verbesserung [{epochs_no_improve}/{config['early_stopping']['patience']}]")
                
                if epochs_no_improve >= config["early_stopping"]["patience"]:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

                if trial:
                    trial.report(current_val_metric, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            logger.info(f"✅ Training für Trial abgeschlossen! Beste {metric}: {best_val:.4f}")

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

        # === Objekte freigeben (Referenzen löschen) ===
        trainer = None
        encoder = None
        train_loader = None
        val_loader = None
        acc_at_k_loader = None
        encoder = None

        # === CUDA entlasten (nur wenn verfügbar) ===
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        # === RAM & GPU Speicher anzeigen (optional) ===
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
    model_mode = args.model.lower()
    assert model_mode in ["sequence", "graph"], "model_mode must be 'sequence' or 'graph'"

    config_path = os.path.join(BASE_DIR, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # if args.loc_encoding_type: 
    #     config["graph_encoder"]["loc_encoding_type"] = args.loc_encoding_type
    #     config["perceiver_encoder"]["loc_encoding_type"] = args.loc_encoding_type

    experiment_name = f"Location Encodings"
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%m%d_%H%M")

    run_identifier = f"{timestamp}_{model_mode}"
    if args.optuna:
        run_identifier += "_Optuna"

    logger = setup_logger(model_name=run_identifier)
    logger.info(f"🚀 Starting preparations for {model_mode} model with '{config_path}'...")

    if args.optuna:
        logger.info("🚀 Starting hyperparameter optimization with Optuna...")

        def objective(trial: optuna.trial.Trial):
            return run_trial(
                config=config,
                model_mode=model_mode,
                gpu_id=0,
                trial=trial,
                timestamp=timestamp
            )

        if args.timestamp:
            timestamp = args.timestamp
            logger.info(f"Using provided timestamp for study name: {timestamp}")

        study_name = f"{model_mode}_Optuna_{timestamp}"
        db_filename = f"{study_name}.db"

        tmp_dir = os.environ.get("TMPDIR", "/tmp") 
        tmp_db_path = os.path.join(tmp_dir, db_filename)

        final_db_path = os.path.join(os.getcwd(), db_filename)
        logger.info(f"Working with temporary DB at: {tmp_db_path}")

        if os.path.exists(final_db_path):
            logger.info(f"Found existing DB at {final_db_path}. Copying to {tmp_dir}...")
            shutil.copy2(final_db_path, tmp_db_path)

        # Optuna nutzt jetzt den lokalen Pfad
        storage_url = f"sqlite:///{tmp_db_path}"

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=4)
        )

        try:
            with mlflow.start_run(run_name=f"Optuna_{run_identifier}"):
                try:
                    study.optimize(objective, n_trials=args.n_trials)
                except KeyboardInterrupt:
                    logger.warning("\nOptuna study manually stopped.")
        
        except Exception as e:
            logger.error(f"An error occurred during optimization: {e}")
            raise e # Fehler weiterwerfen, damit der Job als Failed markiert wird

        finally:
            # --- SCHRITT 4: Rücksicherung ---
            # Dieser Block wird IMMER ausgeführt, auch bei Crash oder Timeout
            logger.info("🔄 Copying database back from temporary storage to Home...")
            if os.path.exists(tmp_db_path):
                try:
                    shutil.copy2(tmp_db_path, final_db_path)
                    logger.info(f"✅ Database successfully saved to: {final_db_path}")
                    
                    # Optional: Aufräumen in /tmp
                    os.remove(tmp_db_path)
                except Exception as e:
                    logger.error(f"❌ Failed to copy database back! Data is still in {tmp_db_path}. Error: {e}")
            else:
                logger.warning("⚠️ No temporary database file found to copy back.")

        # --- Reporting (nur wenn erfolgreich) ---
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
            model_mode=model_mode,
            gpu_id=0,
            trial=None,
            timestamp=timestamp,
            resume_model_id=args.resume_run_id
        )
        logger.info("✅ Training complete!")
