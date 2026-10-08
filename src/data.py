import os
from pathlib import Path
from typing import Dict, List, Tuple


def get_data_loaders(
    data_dir: str = "data",
) -> Tuple[
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    Dict[str, int],
]:
    data_path = Path(data_dir)

    # Filter out background noise directory
    classes = sorted(
        [
            d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith("_")
        ]
    )
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Read list files
    def read_list_file(file_name: str) -> set:
        list_file = data_path / file_name
        if not list_file.exists():
            return set()
        with open(list_file, "r", encoding="utf-8") as f:
            # Normalize relative path
            return {
                os.path.normpath(line.strip())
                for line in f
                if line.strip().endswith(".wav")
            }

    eval_rel_paths = read_list_file("validation_list.txt")
    test_rel_paths = read_list_file("testing_list.txt")

    train_list: List[Tuple[str, int]] = []
    eval_list: List[Tuple[str, int]] = []
    test_list: List[Tuple[str, int]] = []

    # Scan .wav files and split into sets
    for cls_name in classes:
        cls_dir = data_path / cls_name
        label_id = label_to_idx[cls_name]

        for wav_path in cls_dir.glob("*.wav"):
            rel_path = os.path.normpath(f"{cls_name}/{wav_path.name}")
            full_path = str(wav_path)

            if rel_path in test_rel_paths:
                test_list.append((full_path, label_id))
            elif rel_path in eval_rel_paths:
                eval_list.append((full_path, label_id))
            else:
                train_list.append((full_path, label_id))

    return train_list, eval_list, test_list, label_to_idx
