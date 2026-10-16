import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from .models import BiRNN, CNN, LSTM, RNN, GRU


def save_checkpoint(
    model: nn.Module,
    model_name: str,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    val_acc: Optional[float] = None,
    val_loss: Optional[float] = None,
    epoch: Optional[int] = None,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    """Lưu trọng số mô hình (.pt) và xuất kèm file cấu hình JSON (.json) để không cần train lại.

    Args:
        model: Đối tượng PyTorch model cần lưu.
        model_name: Tên kiến trúc ("CNN", "Vanilla RNN", "BiRNN", "LSTM",...).
        save_path: Đường dẫn lưu file (.pt hoặc .pth).
        config: Dict chứa siêu tham số khởi tạo model (input_dim, hidden_dim, num_classes,...).
        val_acc: Độ chính xác validation tốt nhất (%).
        val_loss: Mất mát validation.
        epoch: Epoch tốt nhất.
        label_to_idx: Bảng ánh xạ nhãn sang index.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = {}

    # Lưu trọng số nhị phân và siêu tham số vào file .pt
    checkpoint = {
        "model_name": model_name,
        "config": config,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "epoch": epoch,
        "label_to_idx": label_to_idx,
    }
    torch.save(checkpoint, str(path))

    # Xuất kèm file JSON để có thể đọc cấu hình mà không cần nạp PyTorch
    json_path = path.with_suffix(".json")
    json_data = {
        "model_name": model_name,
        "config": config,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "epoch": epoch,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "label_to_idx": label_to_idx,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Tự động khôi phục kiến trúc mô hình, nạp trọng số và đưa về eval mode.

    Args:
        checkpoint_path: Đường dẫn tới file checkpoint (.pt).
        device: Thiết bị tính toán (CPU hoặc CUDA).

    Returns:
        model: Mô hình PyTorch đã nạp trọng số và ở chế độ eval().
        checkpoint: Toàn bộ thông tin metadata được lưu kèm.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "CNN")
    config = checkpoint.get("config", {})

    # Tự động khởi tạo lại đúng kiến trúc từ config đã lưu
    if model_name == "CNN":
        model = CNN(num_classes=config.get("num_classes", 10))
    elif model_name in ["Vanilla RNN", "RNN"]:
        model = RNN(
            input_dim=config.get("input_dim", 40),
            hidden_dim=config.get("hidden_dim", 168),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 10),
        )
    elif model_name == "BiRNN":
        model = BiRNN(
            input_dim=config.get("input_dim", 40),
            hidden_dim=config.get("hidden_dim", 101),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 10),
        )
    elif "LSTM" in model_name:
        model = LSTM(
            input_dim=config.get("input_dim", 40),
            hidden_dim=config.get("hidden_dim", 82),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 10),
            bidirectional=config.get("bidirectional", False),
        )
    elif "GRU" in model_name:
        model = GRU(
            input_dim=config.get("input_dim", 40),
            hidden_dim=config.get("hidden_dim", 95),
            num_layers=config.get("num_layers", 2),
            num_classes=config.get("num_classes", 10),
            bidirectional=config.get("bidirectional", False),
        )
    else:
        raise ValueError(
            f"Không nhận diện được kiến trúc mô hình: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint
