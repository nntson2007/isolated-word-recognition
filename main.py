import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src import (
    BiRNN,
    CNN,
    RNN,
    LSTM,
    GRU,
    eval,
    train,
    save_checkpoint,
    load_checkpoint,
    preprocess_and_cache,
    get_preprocessed_loaders,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processed_dir = Path("data/processed")

    # Check if prepocessed cache exist
    if not (processed_dir / "train.pt").exists():
        print("\nChưa phát hiện cache dữ liệu tiền xử lý.")
        print("Đang tiến hành trích xuất Log-Mel Spectrogram")
        preprocess_and_cache(
            data_dir="data",
            output_dir=str(processed_dir),
            batch_size=256,
            device=device,
        )

    # Load In-Memory Dataset
    print("\nNạp dữ liệu")
    batch_size = 256
    train_loader, eval_loader, test_loader, label_to_idx = get_preprocessed_loaders(
        processed_dir=str(processed_dir),
        batch_size=batch_size,
    )
    print(
        f"Dữ liệu (Batch Size = {batch_size}):\n"
        f"    - Train: {len(train_loader.dataset):,} mẫu ({len(train_loader)} batches)\n"
        f"    - Val:   {len(eval_loader.dataset):,} mẫu ({len(eval_loader)} batches)\n"
        f"    - Test:  {len(test_loader.dataset):,} mẫu ({len(test_loader)} batches)"
    )

    # Init Model architecture
    num_classes = len(label_to_idx)
    models = {
        "CNN": CNN(num_classes=num_classes),
        "Vanilla RNN": RNN(input_dim=40, hidden_dim=168, num_classes=num_classes),
        "BiRNN": BiRNN(input_dim=40, hidden_dim=101, num_classes=num_classes),
        "LSTM": LSTM(input_dim=40, hidden_dim=82, num_classes=num_classes, bidirectional=False),
        "GRU": GRU(input_dim=40, hidden_dim=95, num_classes=num_classes, bidirectional=False),
    }

    model_configs = {
        "CNN": {"num_classes": num_classes},
        "Vanilla RNN": {"input_dim": 40, "hidden_dim": 168, "num_layers": 2, "num_classes": num_classes},
        "BiRNN": {"input_dim": 40, "hidden_dim": 101, "num_layers": 2, "num_classes": num_classes},
        "LSTM": {"input_dim": 40, "hidden_dim": 82, "num_layers": 2, "num_classes": num_classes, "bidirectional": False},
        "GRU": {"input_dim": 40, "hidden_dim": 95, "num_layers": 2, "num_classes": num_classes, "bidirectional": False},
    }

    # Training config
    num_epochs = 5
    loss_function = nn.CrossEntropyLoss()

    for name, model in models.items():
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(f"Mô hình: {name} (Params: {total_params:,})")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

        best_val_acc = 0.0
        best_ckpt = f"checkpoints/best_{name.replace(' ', '_')}.pt"

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_accuracy = train(
                model=model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                extractor=None,
            )
            eval_loss, eval_accuracy = eval(
                model=model,
                device=device,
                eval_loader=eval_loader,
                loss_function=loss_function,
                extractor=None,
            )
            duration = time.time() - start

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({duration:.2f}s) | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {eval_loss:.4f} - Acc: {eval_accuracy:.2f}%"
            )

            # Save Checkpoint
            if eval_accuracy > best_val_acc:
                best_val_acc = eval_accuracy
                save_checkpoint(
                    model=model,
                    model_name=name,
                    save_path=best_ckpt,
                    config=model_configs[name],
                    val_acc=best_val_acc,
                    val_loss=eval_loss,
                    epoch=epoch,
                    label_to_idx=label_to_idx,
                )

        # Load by best config
        if Path(best_ckpt).exists():
            model, _ = load_checkpoint(best_ckpt, device=device)

        test_loss, test_accuracy = eval(
            model=model,
            device=device,
            eval_loader=test_loader,
            loss_function=loss_function,
            extractor=None,
        )
        print(
            f"-> Test Loss: {test_loss:.4f} - Test Acc: {test_accuracy:.2f}% (Best Val: {best_val_acc:.2f}%)")


if __name__ == "__main__":
    main()
