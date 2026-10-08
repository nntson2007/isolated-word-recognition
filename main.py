import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from src import BiRNN, CNN, RNN, get_data_loaders, eval, train, SpeechDataset, LogMelExtractor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, eval_files, test_files, label_to_idx = (
        get_data_loaders("data")
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        SpeechDataset(train_files),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        SpeechDataset(eval_files),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        SpeechDataset(test_files),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model architectures (balanced to ~97K-98K parameters)
    num_classes = len(label_to_idx)
    models = {
        "CNN": CNN(num_classes=num_classes),
        "Vanilla RNN": RNN(input_dim=40, hidden_dim=168, num_classes=num_classes),
        "BiRNN": BiRNN(input_dim=40, hidden_dim=101, num_classes=num_classes),
    }

    # Training setup
    num_epochs = 5
    extractor = LogMelExtractor().to(device)
    loss_function = nn.CrossEntropyLoss()

    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {name} (Params: {total_params:,})")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss, train_accuracy = train(
                model=model, device=device, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function, extractor=extractor
            )
            eval_loss, eval_accuracy = eval(
                model=model, device=device, eval_loader=eval_loader, loss_function=loss_function, extractor=extractor
            )
            duration = time.time() - start

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({duration:.2f}s) | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_accuracy:.2f}% | "
                f"Val Loss: {eval_loss:.4f} - Acc: {eval_accuracy:.2f}%"
            )

        # Benchmark on test set
        test_loss, test_accuracy = eval(
            model=model, device=device, eval_loader=test_loader, loss_function=loss_function, extractor=extractor
        )
        print(f"Test Loss: {test_loss:.4f} - Acc: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
