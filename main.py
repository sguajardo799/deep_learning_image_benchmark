import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
import yaml

from src import data, models
from src.train import train_model
from src.utils import save_loss_plot

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml",
                        help="ruta al archivo de configuración")
    # sobrescrituras opcionales
    parser.add_argument("--mode", help="download_data | train_model | evaluate_model")
    parser.add_argument("--dataset_name")
    parser.add_argument("--model_name")
    parser.add_argument("--task")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_cli(cfg, args):
    # solo pisamos si viene desde CLI
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.dataset_name is not None:
        cfg["dataset_name"] = args.dataset_name
    if args.model_name is not None:
        cfg["model_name"] = args.model_name
    if args.task is not None:
        cfg["task"] = args.task
    return cfg

def main():
    args = parse_arguments()
    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)

    mode = cfg["mode"]
    task = cfg["task"]
    dataset_name = cfg["dataset_name"]
    model_name = cfg["model_name"]
    output_dir = cfg["paths"]["output_dir"]
    cache_dir = cfg["paths"]["cache_dir"]
    metrics_path = cfg["paths"].get("metrics_path")
    if not metrics_path:
        metrics_path = os.path.join(output_dir, "training_metrics.csv")

    if mode == "download_data":
        data.download_data(dataset_name, output_dir=output_dir, cache_dir=cache_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "train_model":
        train_loader, num_classes = data.load_data(
            dataset_name=dataset_name,
            split="train",
            output_dir=output_dir,
            cache_dir=cache_dir,
            batch_size=cfg["train"]["batch_size"],
        )
        val_loader, _ = data.load_data(
            dataset_name=dataset_name,
            split="validation",
            output_dir=output_dir,
            cache_dir=cache_dir,
            batch_size=cfg["train"]["batch_size"],
        )

        model = models.create_model(
            task=task,
            model_name=model_name,
            num_classes=num_classes,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["lr"],
        )

        start = time.time()
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            task=task,
            patience=cfg["train"]["patience"],
            epochs=cfg["train"]["epochs"],
            save_path=cfg["paths"]["save_path"],
            metrics_path=metrics_path,
        )
        end = time.time()
        training_time = end - start
        save_loss_plot(history, training_time, save_path="./data/loss_curve.png")

    elif mode == "evaluate_model":
        # leer modelo y evaluar
        raise NotImplementedError
    else:
        raise ValueError(f"Modo desconocido: {mode}")

if __name__ == "__main__":
    main()
