"""SMILES tokenizer implementation compatible with HuggingFace transformers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from joint_improvement.utils.config import BaseConfig

SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""
)
DEFAULT_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"


@dataclass
class SMILESTokenizerConfig(BaseConfig):
    """SMILESTokenizer configuration."""

    vocab_filepath: str
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str | None = None
    mask_token: str | None = None
    task_tokens: dict[str, str] | None = None
    additional_special_tokens: list[str] | None = None
    regex_pattern: str | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path) -> SMILESTokenizerConfig:
        """Load config from file or directory."""
        config_path = Path(pretrained_model_name_or_path)
        if config_path.is_dir():
            config_path = config_path / DEFAULT_TOKENIZER_CONFIG_FILENAME
        return cls.from_json(config_path)  # type: ignore[return-value]


class SMILESTokenizer:
    """SMILES regextokenizer compatible with HuggingFace transformers interface."""

    def __init__(
        self,
        vocab_filepath: str,
        bos_token: str,
        eos_token: str,
        pad_token: str,
        unk_token: str | None = None,
        mask_token: str | None = None,
        task_tokens: dict[str, str] | None = None,
        additional_special_tokens: list[str] | None = None,
        regex_pattern: str = SMILES_REGEX_PATTERN,
    ) -> None:
        self.vocab_filepath = vocab_filepath
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.task_tokens: dict[str, str] = task_tokens or {}
        self.additional_special_tokens: list[str] = additional_special_tokens or []
        self.regex = re.compile(regex_pattern)

        self._post_init()

    def _post_init(self) -> None:
        """Initialize vocabulary after instance attributes are set."""
        special_tokens = self._build_special_tokens_list()
        self._load_vocab(special_tokens)

    def _build_special_tokens_list(self) -> list[str]:
        """Build list of all special tokens."""
        special_tokens: list[str] = [self.bos_token, self.eos_token, self.pad_token]
        if self.mask_token is not None:
            special_tokens.append(self.mask_token)
        if self.unk_token is not None:
            special_tokens.append(self.unk_token)
        special_tokens.extend(self.additional_special_tokens)
        special_tokens.extend(self.task_tokens.values())
        return special_tokens

    def _load_vocab(self, special_tokens: list[str]) -> None:
        """Load vocabulary from file and build reverse mapping."""
        self.vocab = self._load_vocab_from_file(self.vocab_filepath, special_tokens)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @property
    def bos_token_id(self) -> int:
        """BOS token ID."""
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        """EOS token ID."""
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self) -> int:
        """PAD token ID."""
        return self.vocab[self.pad_token]

    @property
    def mask_token_id(self) -> int | None:
        """MASK token ID, or None if mask_token is not set."""
        return self.vocab.get(self.mask_token) if self.mask_token else None

    @property
    def unk_token_id(self) -> int | None:
        """UNK token ID, or None if unk_token is not set."""
        return self.vocab.get(self.unk_token) if self.unk_token else None

    @property
    def task_token_ids(self) -> dict[str, int]:
        """Dictionary mapping task names to their token IDs."""
        return {task: self.vocab[token] for task, token in self.task_tokens.items()}

    def _load_vocab_from_file(self, vocab_file: str, special_tokens: list[str]) -> dict[str, int]:
        """Load vocabulary from file, ensuring special tokens are handled deterministically.

        Special tokens defined in config are always appended at the end with deterministic IDs,
        regardless of whether they appear in the vocab file. This ensures consistent token IDs
        and makes the config the single source of truth for special tokens.

        Parameters
        ----------
        vocab_file : str
            Path to vocabulary file containing regular tokens (one per line).
        special_tokens : List[str]
            List of special tokens that should be appended at the end.
            These are defined in the tokenizer config.

        Returns
        -------
        Dict[str, int]
            Vocabulary mapping token strings to token IDs.
            Regular tokens get IDs from their position in vocab file.
            Special tokens are appended at the end with deterministic IDs.
        """
        vocab: dict[str, int] = {}
        seen: set[str] = set()
        special_tokens_set = set(special_tokens)
        token_id = 0

        # Load regular tokens from file, skipping any special tokens
        with open(vocab_file, encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token and token not in seen:
                    # Skip special tokens if they appear in vocab file
                    # Config is the source of truth for special tokens
                    if token in special_tokens_set:
                        continue
                    vocab[token] = token_id
                    token_id += 1
                    seen.add(token)

        # Always append special tokens at the end with deterministic IDs
        # This ensures same special tokens always get same IDs regardless of vocab file
        for token in special_tokens:
            vocab[token] = token_id
            token_id += 1

        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return self.regex.findall(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token string to token ID."""
        token_id = self.vocab.get(token)
        if token_id is not None:
            return token_id
        if self.unk_token_id is not None:
            return self.unk_token_id
        raise KeyError(
            f"Token '{token}' not found in vocabulary and unk_token is None. "
            f"Either add the token to the vocabulary or set unk_token in tokenizer config."
        )

    def _convert_id_to_token(self, index: int) -> str:
        """Convert token ID to token string."""
        token = self.ids_to_tokens.get(index)
        if token is not None:
            return token
        if self.unk_token is not None:
            return self.unk_token
        raise KeyError(
            f"Token ID {index} not found in vocabulary and unk_token is None. "
            f"This ID is out of vocabulary range [0, {self.vocab_size - 1}]."
        )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs to a SMILES string.

        Accepts plain Python lists or PyTorch tensors (CPU/GPU). Token IDs are
        first normalized to Python ints to ensure lookups succeed when the input
        contains torch scalar types.
        """
        normalized_ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else [int(t) for t in token_ids]
        tokens = []
        special_tokens_set: set[str] = {
            self.bos_token,
            self.eos_token,
            self.pad_token,
        }
        if self.mask_token is not None:
            special_tokens_set.add(self.mask_token)
        if self.unk_token is not None:
            special_tokens_set.add(self.unk_token)
        special_tokens_set.update(self.additional_special_tokens)
        special_tokens_set.update(self.task_tokens.values())

        for token_id in normalized_ids:
            token = self._convert_id_to_token(token_id)
            if skip_special_tokens and token in special_tokens_set:
                continue
            tokens.append(token)

        return "".join(tokens)

    def batch_encode(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize text(s). Used by collators.

        Parameters
        ----------
        text : str | List[str]
            Text or list of texts to tokenize.
        return_tensors : str | None, optional
            If "pt", return PyTorch tensors.
        **kwargs : Any
            Ignored (for HF compatibility).

        Returns
        -------
        Dict[str, Any]
            Dictionary with "input_ids" and "attention_mask" keys.
        """
        texts = [text] if isinstance(text, str) else text
        encoded = self.batch_encode(texts)
        attention_mask = [[1] * len(seq) for seq in encoded]

        result: dict[str, Any] = {"input_ids": encoded, "attention_mask": attention_mask}

        if return_tensors == "pt":
            import torch

            result["input_ids"] = torch.tensor(encoded, dtype=torch.long)
            result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)

        return result

    def batch_decode(
        self,
        sequences: list[int] | list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Decode multiple sequences."""
        # Convert to list[list[int]] format
        if isinstance(sequences, torch.Tensor):
            sequences_list = sequences.tolist()
        elif isinstance(sequences, list) and sequences and isinstance(sequences[0], int):
            sequences_list = [sequences]
        else:
            sequences_list = sequences  # type: ignore[assignment]

        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences_list]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> SMILESTokenizer:
        """Load tokenizer from config file.

        Parameters
        ----------
        pretrained_model_name_or_path : str | Path
            Path to directory containing tokenizer_config.json or path to config file.
        **kwargs : Any
            Optional overrides for config values (vocab_filepath, bos_token, etc.).

        Returns
        -------
        SMILESTokenizer
            Initialized tokenizer instance.
        """
        config = SMILESTokenizerConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(
            vocab_filepath=kwargs.get("vocab_filepath", config.vocab_filepath),
            bos_token=kwargs.get("bos_token", config.bos_token),
            eos_token=kwargs.get("eos_token", config.eos_token),
            pad_token=kwargs.get("pad_token", config.pad_token),
            unk_token=kwargs.get("unk_token", config.unk_token),
            mask_token=kwargs.get("mask_token", config.mask_token),
            task_tokens=kwargs.get("task_tokens", config.task_tokens),
            additional_special_tokens=kwargs.get("additional_special_tokens", config.additional_special_tokens),
            regex_pattern=kwargs.get(
                "regex_pattern",
                config.regex_pattern if config.regex_pattern is not None else SMILES_REGEX_PATTERN,
            ),
        )


__all__ = ["SMILESTokenizer", "SMILESTokenizerConfig"]
