import csv
import os

import torch
from torch import nn


# -------------------------
# helpers de batch
# -------------------------
def _unpack_classification_batch(batch, device):
    """
    Soporta:
    - batch = (images, labels)
    - batch = {"image": ..., "label": ...}
    """
    if isinstance(batch, dict):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
    else:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def _move_detection_batch(batch, device):
    """
    Espera que el DataLoader de detección entregue algo como:
    batch = (images, targets)
    donde:
      images = [C,H,W] ya stacked o lista de tensores
      targets = list[dict(boxes=..., labels=..., ...)]
    """
    images, targets = batch
    if torch.is_tensor(images):
        images = list(img.to(device) for img in images)
    else:
        images = [img.to(device) for img in images]

    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


class MetricsLogger:
    """Persiste métricas por epoch en un archivo CSV."""

    def __init__(self, path, metric_names):
        self.path = path
        self.metric_names = list(metric_names)
        self.fieldnames = ["epoch"]
        for prefix in ("train", "val"):
            for metric in self.metric_names:
                self.fieldnames.append(f"{prefix}_{metric}")

        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(self, epoch, train_metrics, val_metrics):
        if not self.path:
            return

        row = {"epoch": epoch}
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for metric in self.metric_names:
                row[f"{prefix}_{metric}"] = metrics.get(metric)

        with open(self.path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row)


class BaseRunner:
    """Loop de entrenamiento con early stopping y logging de métricas."""

    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        save_path=None,
        metrics_path=None,
        metric_names=("loss",),
    ):
        if "loss" not in metric_names:
            raise ValueError("metric_names debe contener 'loss' para controlar early stopping")

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_path = save_path
        self.metric_names = tuple(metric_names)
        self.metrics_logger = MetricsLogger(metrics_path, self.metric_names)

    def train_one_epoch(self, loader):  # pragma: no cover - implementado por subclases
        raise NotImplementedError

    def evaluate_one_epoch(self, loader):  # pragma: no cover - implementado por subclases
        raise NotImplementedError

    def fit(self, train_loader, val_loader, epochs, patience):
        best_val_loss = float("inf")
        best_state_dict = None
        no_improve = 0

        history = {}
        for metric in self.metric_names:
            history[f"train_{metric}"] = []
            history[f"val_{metric}"] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)

            for metric in self.metric_names:
                history[f"train_{metric}"].append(train_metrics.get(metric))
                history[f"val_{metric}"].append(val_metrics.get(metric))

            if self.scheduler is not None:
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics)

            display_parts = []
            for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
                for metric in self.metric_names:
                    value = metrics.get(metric)
                    if value is not None:
                        display_parts.append(f"{prefix}_{metric}={value:.4f}")
            metrics_msg = " ".join(display_parts)
            print(f"[{epoch}/{epochs}] {metrics_msg}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                no_improve = 0
                best_state_dict = self.model.state_dict()
                if self.save_path is not None:
                    torch.save(best_state_dict, self.save_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping en epoch {epoch} (patience={patience})")
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return history


class ClassificationRunner(BaseRunner):
    def __init__(self, model, optimizer, device, scheduler=None, save_path=None, metrics_path=None):
        super().__init__(
            model,
            optimizer,
            device,
            scheduler=scheduler,
            save_path=save_path,
            metrics_path=metrics_path,
            metric_names=("loss", "accuracy"),
        )

    def train_one_epoch(self, loader):
        self.model.train()
        return self._run_epoch(loader, train=True)

    def evaluate_one_epoch(self, loader):
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader, train):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            if train:
                self.optimizer.zero_grad()

            images, labels = _unpack_classification_batch(batch, self.device)
            logits = self.model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}


class DetectionRunner(BaseRunner):
    def __init__(self, model, optimizer, device, scheduler=None, save_path=None, metrics_path=None):
        super().__init__(
            model,
            optimizer,
            device,
            scheduler=scheduler,
            save_path=save_path,
            metrics_path=metrics_path,
            metric_names=("loss",),
        )

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()
            images, targets = _move_detection_batch(batch, self.device)
            try:
                loss_dict = self.model(images, targets)
            except AssertionError as e:
                print("Caught AssertionError during training step!")
                print(f"Batch size: {len(images)}")
                print(f"Image shapes: {[img.shape for img in images]}")
                print(f"Targets: {targets}")
                raise e
            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    def evaluate_one_epoch(self, loader):
        # Torchvision detection models return losses only in train mode.
        # We use no_grad to prevent gradient tracking.
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                images, targets = _move_detection_batch(batch, self.device)
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}


class RunnerFactory:
    @staticmethod
    def create(task, **kwargs):
        task = task.lower()
        if task == "classification":
            return ClassificationRunner(**kwargs)
        if task == "detection":
            return DetectionRunner(**kwargs)
        raise ValueError(f"tarea no implementada: {task}")


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    task: str,
    patience: int = 5,
    epochs: int = 20,
    scheduler=None,
    save_path=None,
    metrics_path=None,
):
    """
    Entrena con early stopping usando la loss de validación.
    Devuelve el modelo (posiblemente NO el mejor en memoria) y
    un diccionario con la evolución de las métricas.
    """
    runner = RunnerFactory.create(
        task=task,
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_path=save_path,
        metrics_path=metrics_path,
    )

    history = runner.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
    )

    return runner.model, history
