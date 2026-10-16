from src import preprocess_and_cache
import torch
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng để trích xuất đặc trưng: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    preprocess_and_cache(
        data_dir="data",
        output_dir="data/processed",
        batch_size=256,
        device=device,
    )


if __name__ == "__main__":
    main()
