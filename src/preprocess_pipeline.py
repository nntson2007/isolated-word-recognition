import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import get_data_loaders
from .feature import LogMelExtractor
from .preprocessing import SpeechDataset


def preprocess_and_cache(
    data_dir: str = "data",
    output_dir: str = "data/processed",
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> None:
    """Scan Dataset and extract Log-Mel Spectrogram"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Bắt đầu đọc danh sách file từ '{data_dir}'...")
    train_files, eval_files, test_files, label_to_idx = get_data_loaders(
        data_dir)
    splits = {
        "train": train_files,
        "eval": eval_files,
        "test": test_files,
    }

    # Save Metadata
    with open(out_path / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)

    extractor = LogMelExtractor().to(device)
    extractor.eval()

    for split_name, file_list in splits.items():
        save_file = out_path / f"{split_name}.pt"
        if save_file.exists():
            print(f"[-] Bỏ qua '{split_name}': Đã tồn tại tại {save_file}")
            continue

        print(
            f"Đang trích xuất đặc trưng cho tập '{split_name}' ({len(file_list):,} mẫu)...")
        start_time = time.time()

        dataset = SpeechDataset(file_list)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

        features_list = []
        labels_list = []

        with torch.no_grad():
            for waveforms, labels in loader:
                waveforms = waveforms.to(device, non_blocking=True)
                specs = extractor(waveforms)  # Shape: (B, 40, 98)

                # Save as float16
                features_list.append(specs.to(torch.float16).cpu())
                labels_list.append(labels.to(torch.int64).cpu())

        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        torch.save(
            {"features": all_features, "labels": all_labels},
            save_file,
        )

        size_mb = save_file.stat().st_size / (1024 * 1024)
        duration = time.time() - start_time
        print(
            f"[+] Hoàn tất '{split_name}': {len(all_features):,} mẫu -> {save_file.name} ({size_mb:.1f} MB) trong {duration:.1f}s")

        del features_list, labels_list, all_features, all_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Tiền xử lý hoàn tất.")


def get_preprocessed_loaders(
    processed_dir: str = "data/processed",
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Load Preprocessed Data"""
    path = Path(processed_dir)

    # Read label
    with open(path / "label_to_idx.json", "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    # Load Tensor
    train_data = torch.load(path / "train.pt", weights_only=True)
    eval_data = torch.load(path / "eval.pt", weights_only=True)
    test_data = torch.load(path / "test.pt", weights_only=True)

    train_ds = TensorDataset(train_data["features"], train_data["labels"])
    eval_ds = TensorDataset(eval_data["features"], eval_data["labels"])
    test_ds = TensorDataset(test_data["features"], test_data["labels"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, eval_loader, test_loader, label_to_idx
