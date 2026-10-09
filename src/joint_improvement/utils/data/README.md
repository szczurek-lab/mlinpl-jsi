# Data Utilities

This module provides utilities for loading and processing data in NPZ (NumPy archive) format.


## Dataset Usage

```python
from joint_improvement.utils import SequenceDataset, SequenceDatasetConfig

# Load config from JSON (seed is encoded in the config path and data_path)
config = SequenceDatasetConfig.from_pretrained("configs/datasets/zinc250k/logp/seed_0/dataset_config.json")

# Create datasets for different splits (seed is already in the config)
train_dataset = SequenceDataset.from_config(config, split="train")  # Uses seed_0/train.npz
val_dataset = SequenceDataset.from_config(config, split="val")    # Uses seed_0/val.npz
test_dataset = SequenceDataset.from_config(config, split="test")   # Uses seed_0/test.npz

# With custom root directory
val_dataset = SequenceDataset.from_config(config, split="val", root="/data")  # /data/seed_0/val.npz

# Alternatively, create config programmatically
config = SequenceDatasetConfig(
    data_path="data/seed_0/{split}.npz",  # Seed encoded in path, {split} placeholder for split name
    sequence_key="sequences",
    target_key="targets"
)
train_dataset = SequenceDataset.from_config(config, split="train")  # data/seed_0/train.npz
```

The `{split}` placeholder is replaced with the split name (train/val/test). The seed should be encoded directly in the data_path (e.g., `seed_0`, `seed_1`, etc.).

## NPZ File Format

NPZ files contain optional `sequences` and `targets` arrays, plus any additional fields. Specify keys via `sequence_key` and `target_key` in config.

## Module Components

- `dataset.py`: `SequenceDataset` class
- `data_io.py`: NPZ file I/O
- `target_transforms.py`: Target transform utilities
- `transforms.py`: Sequence transform utilities
- `collator.py`: Batching utilities
- `dataloader.py`: DataLoader utilities
