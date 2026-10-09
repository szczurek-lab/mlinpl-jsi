# tokenization_hyformer.py
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import BatchEncoding, PreTrainedTokenizer

# -------------------------
# Minimal base tokenizers
# -------------------------

IGNORE_TOKEN_IDX = -100

TOKEN_DICT = {
    "bos": "<s>",
    "eos": "</s>",
    "pad": "<pad>",
    "unk": "<unk>",
    "mask": "<mask>",
}

TASK_TOKEN_DICT = {
    "lm": "<lm>",
    "prediction": "<cls>",
    "mlm": "<mlm>",
}

MAX_LENGTH = 512


class BaseTokenizer(ABC):
    def __init__(
        self,
        vocabulary_path: str,
        max_length: int = MAX_LENGTH,
        bos_token: str = TOKEN_DICT["bos"],
        eos_token: str = TOKEN_DICT["eos"],
        pad_token: str = TOKEN_DICT["pad"],
        unk_token: str | None = None,
        mask_token: str | None = TOKEN_DICT["mask"],
        task_tokens: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_file = vocabulary_path
        self.max_length = max_length
        self._setup_special_tokens(bos_token, eos_token, unk_token, pad_token, mask_token, task_tokens)
        self.vocab = self._load_vocab(vocabulary_path)
        self._add_special_tokens_to_vocab()

    def _setup_special_tokens(
        self,
        bos_token: str,
        eos_token: str,
        unk_token: str | None,
        pad_token: str,
        mask_token: str | None,
        task_tokens: dict[str, str] | None,
    ) -> None:
        self.special_tokens = {"bos": bos_token, "eos": eos_token, "pad": pad_token}
        if unk_token is not None:
            self.special_tokens["unk"] = unk_token
        if mask_token is not None:
            self.special_tokens["mask"] = mask_token
        task_dict = TASK_TOKEN_DICT.copy() if task_tokens is None else task_tokens.copy()
        self.special_tokens.update(task_dict)

    @abstractmethod
    def _load_vocab(self, vocab_file: str) -> dict[str, int]: ...
    @abstractmethod
    def tokenize(self, text: str) -> list[str]: ...

    def _add_special_tokens_to_vocab(self) -> None:
        next_id = len(self.vocab)
        for _, tok in self.special_tokens.items():
            if tok is not None and tok not in self.vocab:
                self.vocab[tok] = next_id
                next_id += 1
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self._token_id_cache: dict[str, int] = {}

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.special_tokens["pad"]]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.special_tokens["bos"]]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.special_tokens["eos"]]

    @property
    def unk_token_id(self) -> int | None:
        t = self.special_tokens.get("unk")
        return None if t is None else self.vocab[t]

    @property
    def mask_token_id(self) -> int | None:
        t = self.special_tokens.get("mask")
        return None if t is None else self.vocab[t]

    def __len__(self) -> int:
        return len(self.vocab)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        out: list[int] = []
        for tok in tokens:
            if tok in self._token_id_cache:
                out.append(self._token_id_cache[tok])
            elif tok in self.vocab:
                idx = self.vocab[tok]
                self._token_id_cache[tok] = idx
                out.append(idx)
            elif "unk" in self.special_tokens and self.unk_token_id is not None:
                out.append(self.unk_token_id)
            else:
                raise KeyError(f"Unknown token '{tok}' and no UNK defined")
        return out

    def all_special_ids(self) -> list[int]:
        return self.convert_tokens_to_ids(list(self.special_tokens.values()))

    def __call__(
        self, inputs: str | list[str], task: str, padding: bool = False, truncation: bool = True, **kwargs
    ) -> dict[str, Any]:
        if isinstance(inputs, str):
            inputs = [inputs]
        batch_ids: list[list[int]] = []
        for text in inputs:
            toks = self.tokenize(text)
            toks.insert(0, self.special_tokens[task] if task in self.special_tokens else TASK_TOKEN_DICT["lm"])
            toks.insert(1, self.special_tokens["bos"])
            toks.append(self.special_tokens["eos"])
            if truncation and len(toks) > self.max_length:
                toks = toks[: self.max_length - 1] + [toks[-1]]
            ids = self.convert_tokens_to_ids(toks)
            batch_ids.append(ids)

        max_len = max(len(x) for x in batch_ids)
        if padding:
            pad = self.pad_token_id
            attn = []
            padded = []
            for ids in batch_ids:
                attn.append([1] * len(ids) + [0] * (max_len - len(ids)))
                padded.append(ids + [pad] * (max_len - len(ids)))
            batch_ids = padded
        else:
            attn = [[1] * len(ids) for ids in batch_ids]

        return {"input_ids": batch_ids, "attention_mask": attn}

    def _join_tokens(self, tokens: list[str]) -> str:
        return "".join(tokens)


SMILES_REGEX_PATTERN = (
    r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""
)


class SMILESTokenizer(BaseTokenizer):
    def __init__(self, vocabulary_path: str, regex_pattern: str = SMILES_REGEX_PATTERN, **kwargs) -> None:
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        super().__init__(vocabulary_path=vocabulary_path, **kwargs)

    def _load_vocab(self, vocab_file: str) -> dict[str, int]:
        vocab: dict[str, int] = {}
        with open(vocab_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                tok = line.strip()
                if tok:
                    vocab[tok] = i
        return vocab

    def tokenize(self, text: str) -> list[str]:
        return self.regex.findall(text)

    @classmethod
    def from_config(cls, config, **kwargs) -> SMILESTokenizer:
        init_kwargs = {
            "vocabulary_path": config.vocabulary_path,
            "max_length": getattr(config, "max_length", 512),
            "task_tokens": getattr(config, "task_tokens", None),
        }
        init_kwargs.update(getattr(config, "kwargs", {}) or {})
        init_kwargs.update(kwargs)
        return cls(**init_kwargs)


AA_REGEX_PATTERN = r"([ACDEFGHIKLMNPQRSTVWYX]|[BZO]|U|\-|\.)"


class AATokenizer(SMILESTokenizer):
    def __init__(self, vocabulary_path: str, regex_pattern: str = AA_REGEX_PATTERN, **kwargs) -> None:
        super().__init__(vocabulary_path=vocabulary_path, regex_pattern=regex_pattern, **kwargs)


# -------------------------
# HF wrapper
# -------------------------


class HyformerTokenizer(PreTrainedTokenizer):
    """HF-compatible wrapper around the above tokenizers.

    Use `mode="aa"` or `mode="smiles"`. Default 'aa'.
    """

    vocab_files_names = {"vocab_file": "aa_vocab.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        mode: str = "aa",
        max_length: int = 512,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str | None = "<unk>",
        mask_token: str | None = "<mask>",
        **kwargs,
    ):
        tok_kwargs = {"vocabulary_path": vocab_file, "max_length": max_length}
        if mode == "aa":
            self._inner = AATokenizer(**tok_kwargs)
        elif mode == "smiles":
            self._inner = SMILESTokenizer(**tok_kwargs)
        else:
            raise ValueError("mode must be 'aa' or 'smiles'")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            model_max_length=max_length,
            **kwargs,
        )
        self._vocab_file = vocab_file
        self.mode = mode

    @property
    def vocab_size(self) -> int:
        return len(self._inner)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._inner.vocab)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._inner.vocab:
            return self._inner.vocab[token]
        uid = self._inner.unk_token_id
        if uid is None:
            raise KeyError(f"Unknown token '{token}' and no <unk>")
        return uid

    def _convert_id_to_token(self, index: int) -> str:
        return self._inner.ids_to_tokens[index]

    def _tokenize(self, text: str) -> list[str]:
        return self._inner.tokenize(text)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        return [self._inner.bos_token_id] + token_ids_0 + [self._inner.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        os.makedirs(save_directory, exist_ok=True)
        out = os.path.join(save_directory, ((filename_prefix + "-") if filename_prefix else "") + "vocab.txt")
        inv = sorted(self._inner.vocab.items(), key=lambda kv: kv[1])
        with open(out, "w", encoding="utf-8") as f:
            for tok, _id in inv:
                f.write(tok + "\n")
        return (out,)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._inner._join_tokens(tokens)

    # Let HF callers pass 'task' to add task tokens via our BaseTokenizer batching
    def __call__(
        self,
        text: str | list[str],
        task: str = "lm",
        padding: bool | str = False,
        truncation: bool | str = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        out = self._inner(
            inputs=text,
            task=task,
            padding=bool(padding) or (isinstance(padding, str) and padding != "do_not_pad"),
            truncation=bool(truncation) or (isinstance(truncation, str) and truncation != "do_not_truncate"),
        )
        input_ids, attention_mask = out["input_ids"], out["attention_mask"]
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return BatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            tensor_type="pt" if return_tensors == "pt" else None,
        )
