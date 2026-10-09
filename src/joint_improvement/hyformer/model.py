# model.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import HyformerConfig
from .layers.prediction_head import PredictionHeadModule
from .layers.transformer_layer import TransformerLayer
from .losses import compute_lm_loss, compute_mlm_loss, compute_prediction_loss
from .outputs import ModelOutput
from .pretrained import PretrainedMixin


class Hyformer(PretrainedMixin, nn.Module):
    """
    Hyformer backbone model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of tokens).
    d_model : int
        Model dimension (hidden size).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of decoder blocks (transformer layers).
    max_seq_len : int
        Maximum sequence length for positional encodings and attention masks.

    Attributes
    ----------
    vocab_size : int
        Size of the vocabulary.
    num_prediction_tasks : int | None
        Number of prediction tasks (if prediction head is enabled).
    embed : nn.Embedding
        Token embedding layer of shape [vocab_size, d_model].
    blocks : nn.ModuleList
        List of decoder blocks, each containing attention and MLP sub-layers.
    final_norm : nn.LayerNorm
        Final layer normalization applied before the task heads.
    heads : nn.ModuleDict
        Task-specific heads. Contains "lm", "mlm", and optionally "prediction".
        The "lm" and "mlm" heads are weight-tied with the embedding layer.
        The "prediction" head is a PredictionHeadModule following DINOv2 best practices
        (LayerNorm, dropout, classifier) and uses the CLS token representation.

    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        attn_dropout_p: float = 0.0,
        rms_norm_eps: float = 1e-5,
        num_prediction_tasks: int | None = None,
        prediction_task_type: str | None = None,
        predictor_dropout_p: float = 0.0,
        predictor_head_depth: int = 1,
        predictor_head_act_fn: str = "gelu",
        generator_type: str | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_prediction_tasks = num_prediction_tasks
        self.prediction_task_type = prediction_task_type
        self.generator_type = generator_type

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    attn_dropout_p=attn_dropout_p,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, eps=rms_norm_eps)
        self.heads = nn.ModuleDict(
            {
                "lm": nn.Linear(d_model, vocab_size, bias=False),
                "mlm": nn.Linear(d_model, vocab_size, bias=False),
            }
        )
        # Weight tying: both heads share embedding weights
        self.heads["lm"].weight = self.embed.weight
        self.heads["mlm"].weight = self.embed.weight

        if num_prediction_tasks is not None and prediction_task_type is not None:
            self.heads["prediction"] = PredictionHeadModule(
                d_model=d_model,
                num_targets=num_prediction_tasks,
                predictor_dropout_p=predictor_dropout_p,
                predictor_head_depth=predictor_head_depth,
                predictor_head_act_fn=predictor_head_act_fn,
            )

        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        task: str,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        Forward pass through the model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [B, T] where B is batch size and
            T is sequence length.
        task : str
            Task name (e.g., "lm", "mlm", "prediction"). Used to determine
            attention type automatically.
        attention_mask : Optional[torch.Tensor], default=None
            Padding mask tensor of shape [B, T] where 1 indicates valid tokens
            and 0 indicates padding. Used to create attention mask for bidirectional tasks.
        labels : Optional[torch.Tensor], default=None
            Labels for LM/MLM tasks. Used for computing loss for language modeling tasks.
        targets : Optional[torch.Tensor], default=None
            Targets for prediction/regression tasks. Required for computing loss for prediction tasks.
        kv_caches : Optional[list[Optional[KVCache]]], default=None
            Optional list of key-value caches, one per decoder block.
            If None and use_cache=True, will be initialized as None for
            each block. If provided, should have length equal to n_layers.
        use_cache : bool, default=False
            Whether to use and update key-value caches for autoregressive
            generation.

        Returns
        -------
        ModelOutput
            Model output containing:
            - logits: Tensor of shape [B, T, vocab_size] for LM/MLM tasks,
                     or [B, num_prediction_tasks] for prediction tasks
            - loss: Optional loss tensor if labels/targets provided

        """
        if task not in self.heads:
            raise ValueError(f"Unknown task '{task}'. Available heads: {list(self.heads.keys())}")

        B, T = input_ids.shape
        is_causal = task == "lm"

        x = self.embed(input_ids)
        for _, layer in enumerate(self.layers):
            x = layer(x, is_causal=is_causal, attn_mask=None if is_causal else attention_mask)
        x = self.norm(x)

        if task == "prediction":
            pooled = x[:, 0, :]
            logits = self.heads[task](pooled)
        else:
            logits = self.heads[task](x)

        # Compute loss if labels/targets provided
        loss = None
        if task == "prediction":
            if targets is not None:
                if self.num_prediction_tasks is None:
                    raise ValueError("num_prediction_tasks must be set for prediction task")
                if self.prediction_task_type is None:
                    raise ValueError("prediction_task_type must be set for prediction task ")
                loss = compute_prediction_loss(
                    logits,
                    targets,
                    ignore_index=-1,
                    reduction="mean",
                    prediction_task_type=self.prediction_task_type,
                )
        elif labels is not None:
            if task == "lm":
                loss = compute_lm_loss(logits, labels, shift_labels=is_causal)
            elif task == "mlm":
                loss = compute_mlm_loss(logits, labels)

        extras = {
            "hidden_states": x,
        }

        return ModelOutput(
            logits=logits,
            loss=loss,
            extras=extras,
        )

    def log_likelihood(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-likelihood of the input tokens.

        Parameters
        ----------
        input_ids: torch.Tensor
            Input token IDs of shape [B, T].
        attention_mask: torch.Tensor
            Attention mask of shape [B, T].
        """
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, task="lm").logits
        log_probs = logits.log_softmax(dim=2)[:, :-1, :]  # [B, T-1, V]
        labels = input_ids[:, 1:].contiguous()  # [B, T-1]

        # Start from attention mask (exclude padding) then additionally ignore:
        # - BOS token (always first token after the prepended task token)
        # - task token id (defensive: if present in labels due to custom formatting)
        # - pad token id (defensive: if attention_mask is inconsistent)
        loss_mask = attention_mask[:, 1:].contiguous().to(dtype=log_probs.dtype)  # [B, T-1]
        if loss_mask.size(1) > 0:
            loss_mask[:, 0] = 0.0  # ignore BOS label at position 0

        task_token_ids = input_ids[:, 0:1]  # [B, 1]
        loss_mask = loss_mask * (labels != task_token_ids).to(dtype=loss_mask.dtype)

        pad_positions = attention_mask == 0
        if bool(pad_positions.any()):
            pad_token_id = int(input_ids.masked_select(pad_positions)[0].item())
            loss_mask = loss_mask * (labels != pad_token_id).to(dtype=loss_mask.dtype)

        nll_per_token = F.nll_loss(log_probs.transpose(1, 2), labels, reduction="none")  # [B, T-1]
        return -(nll_per_token * loss_mask).sum(dim=1)

    @torch.inference_mode()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the prediction head and return post-processed predictions.

        Returns
        -------
        torch.Tensor
            - regression: raw logits of shape [B, num_prediction_tasks]
            - binary_classification / multilabel_classification: sigmoid(logits)
        """
        if "prediction" not in self.heads:
            raise ValueError("Prediction head is not enabled (missing heads['prediction']).")
        if self.prediction_task_type is None:
            raise ValueError("prediction_task_type must be set to use predict().")

        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, task="prediction").logits

        if self.prediction_task_type == "regression":
            return logits
        if self.prediction_task_type in (
            "binary_classification",
            "multilabel_classification",
        ):
            return torch.sigmoid(logits)
        raise ValueError(
            "prediction_task_type must be one of 'regression', 'binary_classification', 'multilabel_classification'."
        )

    def _init_weights(self) -> None:

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        tied_heads = {self.heads["lm"], self.heads["mlm"]}
        prediction_head_classifier = None
        if hasattr(self, "heads") and "prediction" in self.heads:
            prediction_head_classifier = self.heads["prediction"].classifier

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip weight-tied heads (already initialized via embed.weight)
                if module in tied_heads:
                    continue
                # Skip prediction head classifier (initialized by PredictionHeadModule itself)
                if module is prediction_head_classifier:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model.

        Parameters
        ----------
        trainable_only : bool, default=False
            If True, only count trainable parameters. If False, count all parameters.

        Returns
        -------
        int
            Number of parameters in the model.

        Examples
        --------
        >>> model = Hyformer.from_config(config)
        >>> total_params = model.get_num_params()
        >>> trainable_params = model.get_num_params(trainable_only=True)
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config: HyformerConfig) -> Hyformer:
        """Create Hyformer model from configuration.

        Parameters
        ----------
        config : HyformerConfig
            Model configuration.

        Returns
        -------
        Hyformer
            Instantiated Hyformer model.

        Examples
        --------
        >>> config = HyformerConfig.from_pretrained("configs/hyformer/base.json")
        >>> model = Hyformer.from_config(config)
        """
        config_dict = config.to_dict()
        generator_type = config_dict["generator_type"]
        init_kwargs = {
            "vocab_size": int(config_dict["vocab_size"]),
            "d_model": int(config_dict["d_model"]),
            "n_heads": int(config_dict["n_heads"]),
            "n_layers": int(config_dict["n_layers"]),
            "max_seq_len": int(config_dict["max_seq_len"]),
            "attn_dropout_p": float(config_dict["attn_dropout_p"]),
            "rms_norm_eps": float(config_dict["rms_norm_eps"]),
            "num_prediction_tasks": (
                int(config_dict["num_prediction_tasks"])
                if config_dict.get("num_prediction_tasks") is not None
                else None
            ),
            "prediction_task_type": config_dict.get("prediction_task_type"),
            "predictor_dropout_p": float(config_dict["predictor_dropout_p"]),
            "predictor_head_depth": int(config_dict["predictor_head_depth"]),
            "predictor_head_act_fn": config_dict["predictor_head_act_fn"],
        }
        if generator_type == "unconditional":
            from joint_improvement.generators import UnconditionalGeneratorMixin

            HyformerClass = type("Hyformer", (UnconditionalGeneratorMixin, cls), {})
            return HyformerClass(**init_kwargs)

        elif generator_type == "tasar":
            from joint_improvement.generators import TasarMixin

            HyformerClass = type("Hyformer", (TasarMixin, cls), {})
            return HyformerClass(**init_kwargs)

        elif generator_type == "gumbeldore":
            from joint_improvement.generators import GumbeldoreMixin

            HyformerClass = type("Hyformer", (GumbeldoreMixin, cls), {})
            return HyformerClass(**init_kwargs)
        else:
            raise ValueError(
                f"Unknown generator_type '{generator_type}'. Available: 'unconditional', 'tasar', 'tasar_legacy'"
            )
