import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from typing import Optional


def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, loss_function: nn.Module, extractor: Optional[nn.Module] = None) -> tuple[float, float]:
    # Set model to train mode
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    use_cuda = (device.type == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_cuda else None

    for waveforms, labels in train_loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if extractor is not None:
            with torch.no_grad():
                specs = extractor(waveforms)
        else:
            specs = waveforms.float()

        if getattr(model, "is_cnn", False):
            inputs = specs.unsqueeze(1) if specs.dim() == 3 else specs
        else:
            inputs = specs.squeeze(1).transpose(
                1, 2) if specs.dim() == 4 else specs.transpose(1, 2)

        # Training loop
        optimizer.zero_grad()
        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return total_loss / total, (correct / total) * 100.0
