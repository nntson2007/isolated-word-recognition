import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def eval(model: nn.Module, device: torch.device, eval_loader: DataLoader, loss_function: nn.Module, extractor: nn.Module) -> tuple[float, float]:
    # Set model to eval mode
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for waveforms, labels in eval_loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        specs = extractor(waveforms)

        if getattr(model, "is_cnn", False):
            inputs = specs.unsqueeze(1) if specs.dim() == 3 else specs
        else:
            inputs = specs.squeeze(1).transpose(1, 2) if specs.dim() == 4 else specs.transpose(1, 2)

        # Evaluation loop
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return total_loss / total, (correct / total) * 100.0
