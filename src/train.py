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
    # aseguramos lista de tensores
    if torch.is_tensor(images):
        # si viene tensor [B,C,H,W], lo pasamos a lista
        images = list(img.to(device) for img in images)
    else:
        images = [img.to(device) for img in images]

    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


# -------------------------
# epoch de entrenamiento
# -------------------------
def train_one_epoch(model, train_loader, optimizer, device, task):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        if task == "classification":
            images, labels = _unpack_classification_batch(batch, device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

        elif task == "detection":
            images, targets = _move_detection_batch(batch, device)
            # modelos de torchvision detection devuelven dict de losses
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        else:
            raise ValueError(f"tarea no implementada: {task}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# -------------------------
# epoch de evaluación
# -------------------------
def evaluate_one_epoch(model, val_loader, device, task):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if task == "classification":
                images, labels = _unpack_classification_batch(batch, device)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)

            elif task == "detection":
                images, targets = _move_detection_batch(batch, device)
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            else:
                raise ValueError(f"tarea no implementada: {task}")

            total_loss += loss.item()

    return total_loss / len(val_loader)


# -------------------------
# loop completo con early stopping
# -------------------------
import time
import torch

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
    save_path: str = None,
):
    """
    Entrena con early stopping usando la loss de validación.
    Devuelve el modelo (posiblemente NO el mejor en memoria) y
    un diccionario con la evolución de las pérdidas.
    """
    model.to(device)
    best_val_loss = float("inf")
    best_state_dict = None
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, task)
        val_loss = evaluate_one_epoch(model, val_loader, device, task)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # scheduler opcional
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"[{epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0

            # guardar en memoria
            best_state_dict = model.state_dict()

            # guardar en disco
            if save_path is not None:
                torch.save(best_state_dict, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping en epoch {epoch} (patience={patience})")
                break

    # al final restauramos el mejor modelo si lo tenemos
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history
