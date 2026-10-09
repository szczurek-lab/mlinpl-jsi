from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .collators import BaseCollatorWithPadding
    from .dataset import SequenceDataset


class SequenceDataLoader(DataLoader):
    """Minimal PyTorch DataLoader compatible with SequenceDataset and SequenceCollator.

    Convenience wrapper for manual iteration over `SequenceDataset` batches when
    working outside of Hugging Face `Trainer`. Inherits all functionality from
    `torch.utils.data.DataLoader` while providing a simplified interface for
    sequence datasets.

    Examples
    --------
    >>> from src.utils import SequenceDataset, SequenceCollator, SequenceDataLoader
    >>> from transformers import AutoTokenizer, DataCollatorWithPadding
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    >>> dataset = SequenceDataset(sequences=["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"], targets=[0.5, 0.8, 0.3])
    >>> collator = SequenceCollator(base_collator=DataCollatorWithPadding(tokenizer), tokenizer=tokenizer)
    >>> loader = SequenceDataLoader(dataset=dataset, collator=collator, batch_size=2, shuffle=True)
    >>> for batch in loader:
    ...     print(batch["input_ids"].shape)
    ...     break
    torch.Size([2, 9])
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        collator: BaseCollatorWithPadding,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        """Initialize SequenceDataLoader.

        Parameters
        ----------
        dataset : SequenceDataset
            Dataset instance to load batches from.
        collator : BaseCollatorWithPadding
            Collator instance for processing batches. Must be compatible with
            the dataset's output format. Prefer BaseCollator subclasses from
            `collators` module for new code.
        batch_size : int, default=8
            Number of samples per batch.
        shuffle : bool, default=False
            Whether to shuffle the dataset at the beginning of each epoch.
        num_workers : int, default=0
            Number of subprocesses to use for data loading. `0` means data will
            be loaded in the main process.
        pin_memory : bool, default=False
            If `True`, the data loader will copy tensors into CUDA pinned memory
            before returning them. Useful for faster GPU transfer.
        drop_last : bool, default=False
            If `True`, drop the last incomplete batch if the dataset size is not
            divisible by `batch_size`.

        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
