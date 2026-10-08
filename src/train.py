import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model: nn.Module, device: torch.device, train_loader: DataLoader, optimizer: optim.Optimizer, loss_function: nn.Module, extractor: nn.Module) -> tuple[float, float]:
    # Set model to train mode
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for waveforms, labels in train_loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            specs = extractor(waveforms)

        if getattr(model, "is_cnn", False):
            inputs = specs.unsqueeze(1) if specs.dim() == 3 else specs
        else:
            inputs = specs.squeeze(1).transpose(1, 2) if specs.dim() == 4 else specs.transpose(1, 2)

        # Training loop
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return total_loss / total, (correct / total) * 100.0
