import json
import os

import matplotlib.pyplot as plt

def save_loss_plot(history, training_time, save_path="training_loss.png", title="Training vs Validation Loss"):
    """
    history: dict con claves "train_loss" y "val_loss"
    training_time: float en segundos
    save_path: ruta de salida
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Val loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # texto con tiempo de entrenamiento
    # lo colocamos dentro del gráfico, arriba a la izquierda
    txt = f"Tiempo: {training_time:.2f} s"
    plt.text(
        0.02, 0.95,
        txt,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_training_summary(history, training_time, save_path):
    """
    Persist total/mean epoch time and last recorded metrics into a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    epochs_ran = 0
    for values in history.values():
        if isinstance(values, list):
            epochs_ran = len(values)
            break

    mean_epoch_time = training_time / epochs_ran if epochs_ran else 0.0
    final_metrics = {}
    for metric_name, values in history.items():
        if values:
            final_metrics[metric_name] = values[-1]

    summary = {
        "total_train_time_seconds": training_time,
        "mean_epoch_time_seconds": mean_epoch_time,
        "epochs_ran": epochs_ran,
        "final_metrics": final_metrics,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
