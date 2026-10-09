"""Multitask trainer."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from joint_improvement.hyformer.inputs import ModelInput

if TYPE_CHECKING:
    from collections.abc import Iterator
from joint_improvement.utils.config import BaseConfig

from .checkpoint import TrainerCheckpointMixin
from .device_optimizations import setup_device_optimizations
from .task_sampler import UniformTaskSampler


@dataclass
class TrainerState:
    """Trainer state containing training metadata."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    iterations_in_epoch: int = 0
    current_lr: float | None = None
    total_iters: int = 0
    warmup_iters: int = 0


@dataclass
class TrainerConfig(BaseConfig):
    """Training configuration.

    Attributes
    ----------
    tasks : dict[str, float]
        Dictionary mapping task names to their loss weights for multitask learning.
        Example: {"task_1": 1.0, "task_2": 0.5}.
    """

    batch_size: int
    learning_rate: float
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_epochs: int = 2
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    min_lr: float = 1e-5
    decay_lr: bool = True
    dtype: str = "float32"
    compile: bool = False
    patience: int | None = None  # Early stopping patience (None = no early stopping)
    log_every_n_iterations: int | None = (
        None  # Log every N iterations. If None, log only at end of each epoch
    )
    checkpoint_every_n_epochs: int | None = (
        None  # Save checkpoint every N epochs. If None, only save best checkpoint
    )
    tasks: dict[str, float] = field(default_factory=dict)
    allow_tf32: bool = False

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError(
                "TrainerConfig.tasks cannot be empty; provide at least one task with task weight."
            )


class MultiTaskTrainer(TrainerCheckpointMixin):
    """Simplified trainer compatible with HuggingFace models."""

    def __init__(
        self,
        config: TrainerConfig,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        out_dir: str | Path | None = None,
    ) -> None:
        """Initialize trainer.

        Parameters
        ----------
        config : TrainerConfig
            Training configuration.
        model : torch.nn.Module
            Model to train (compatible with HF models).
        device : torch.device | str, optional
            Device to train on. Default is "cpu".
        out_dir : str | Path, optional
            Directory to save checkpoints.
        """
        self.config = config
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.out_dir = Path(out_dir) if out_dir else None

        self.optimizer = self.create_optimizer()

        self.state = TrainerState(current_lr=config.learning_rate)
        self.task_sampler = UniformTaskSampler(tasks=config.tasks)
        self.task_iterators: dict[str, Iterator] = {}
        self.model.to(device)
        (
            self.ptdtype,
            device_type,
            self.ctx,
            self.scaler,
            compile_mode,
        ) = setup_device_optimizations(config, device)

        # Store underlying model before compilation
        self.underlying_model = self.model

        if config.compile:
            try:
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
            except TypeError:
                self.model = torch.compile(self.model, mode=compile_mode)

        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config parameters (HuggingFace pattern).

        Returns
        -------
        torch.optim.Optimizer
            AdamW optimizer configured from TrainerConfig.

        Notes
        -----
        This method can be overridden in subclasses to customize optimizer creation.
        """
        # Group parameters: those with weight decay and those without
        decay_params = []
        no_decay_params = []

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                # Parameters with fewer than 2 dimensions (bias, scale) or LayerNorm don't decay
                if (
                    p.ndim < 2
                    or "bias" in n.lower()
                    or "layernorm" in n.lower()
                    or "layer_norm" in n.lower()
                ):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

    def _get_lr(self, current_iter: int) -> float:
        """Get learning rate for current iteration using cosine decay with linear warmup.

        Parameters
        ----------
        current_iter : int
            Current training iteration.

        Returns
        -------
        float
            Learning rate for this iteration.
        """
        if not self.config.decay_lr:
            return self.config.learning_rate

        # Warmup phase
        if current_iter < self.state.warmup_iters:
            return (
                self.config.learning_rate
                * (current_iter + 1)
                / max(self.state.warmup_iters, 1)
            )

        # Cosine decay phase
        if current_iter >= self.state.total_iters:
            return self.config.min_lr

        decay_iters = self.state.total_iters - self.state.warmup_iters
        if decay_iters <= 0:
            return self.config.min_lr

        decay_ratio = (current_iter - self.state.warmup_iters) / decay_iters
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (
            self.config.learning_rate - self.config.min_lr
        )

    def _update_lr(self, current_iter: int) -> None:
        """Update learning rate for current iteration.

        Parameters
        ----------
        current_iter : int
            Current training iteration.
        """
        lr = self._get_lr(current_iter)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.state.current_lr = lr

    def train_step(self, batch: ModelInput) -> tuple[float, float, int]:
        """Perform a single training step.

        Parameters
        ----------
        batch : ModelInput
            ModelInput containing input data and task name.

        Returns
        -------
        tuple[float, float, int]
            Tuple containing (loss, gradient_norm, num_tokens).
            - loss: Loss value
            - gradient_norm: Gradient norm (clipped if grad_clip > 0, otherwise unclipped norm)
            - num_tokens: Number of tokens processed
        """
        self.optimizer.zero_grad(set_to_none=True)

        num_tokens = batch.input_ids.numel()

        with self.ctx:
            outputs = self.model(**batch.to(self.device))
            loss = outputs.loss
            if loss is None:
                raise ValueError(
                    "Model returned None loss. Ensure targets/labels are provided in the input."
                )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        max_norm = self.config.grad_clip if self.config.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return (
            loss.item(),
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            num_tokens,
        )

    def _get_next_batch(
        self,
        task_name: str,
        train_loaders: dict[str, DataLoader],
    ) -> ModelInput:
        """Get next batch for a task, cycling iterator if exhausted."""
        try:
            return next(self.task_iterators[task_name])
        except StopIteration:
            # Iterator exhausted, recreate to cycle through dataset
            self.task_iterators[task_name] = iter(train_loaders[task_name])
            return next(self.task_iterators[task_name])

    @torch.inference_mode()
    def evaluate_step(self, batch: ModelInput) -> float:
        """Perform a single evaluation step."""
        with self.ctx:
            outputs = self.model(**batch.to(self.device))
            if outputs.loss is None:
                raise ValueError(
                    "Model output loss is None; ensure labels/targets are provided."
                )
            return outputs.loss.item()

    def _save_evaluation_losses_json(
        self, task_losses: dict[str, float], weighted_loss: float
    ) -> None:
        """Save evaluation losses to JSON file.

        Parameters
        ----------
        task_losses : dict[str, float]
            Dictionary mapping task names to their loss values.
        weighted_loss : float
            Weighted validation loss value.
        """
        if not self.out_dir:
            return

        eval_file = self.out_dir / "evaluation_losses.json"

        # Load existing evaluation results if file exists
        evaluation_history: dict[str, dict[str, float | dict[str, float]]] = {}
        if eval_file.exists():
            try:
                with eval_file.open() as f:
                    evaluation_history = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to load existing evaluation losses from {eval_file}: {e}"
                )

        # Create entry for current epoch
        epoch_key = f"epoch_{self.state.epoch}"
        evaluation_history[epoch_key] = {
            "weighted_loss": weighted_loss,
            "per_task": task_losses,
        }

        # Save updated evaluation history
        try:
            with eval_file.open("w") as f:
                json.dump(evaluation_history, f, indent=2)
            logger.debug(f"Saved evaluation losses to {eval_file}")
        except OSError as e:
            logger.warning(f"Failed to save evaluation losses to {eval_file}: {e}")

    @torch.inference_mode()
    def evaluate(
        self, val_loaders: dict[str, DataLoader]
    ) -> tuple[dict[str, float], float]:
        """Evaluate model."""
        was_training = self.model.training
        self.model.eval()

        task_names = list(val_loaders.keys())
        if set(task_names) != set(self.config.tasks.keys()):
            raise ValueError(
                f"Task mismatch: Config tasks: {set(self.config.tasks.keys())}, Loader tasks: {set(task_names)}"
            )

        task_probs = self.task_sampler.get_task_probs(task_names)
        task_losses = {}

        for task_name, val_loader in val_loaders.items():
            losses = [self.evaluate_step(batch) for batch in val_loader]
            losses = [loss for loss in losses if loss is not None]
            task_losses[task_name] = (
                sum(losses) / len(losses) if losses else float("inf")
            )

        weighted_val_loss = (
            sum(task_probs.get(task, 0.0) * loss for task, loss in task_losses.items())
            if task_losses
            else float("inf")
        )

        per_task_str = ", ".join(
            f"{task}: {loss:.4f}" for task, loss in task_losses.items()
        )
        logger.info(
            f"Eval epoch {self.state.epoch}: loss {weighted_val_loss:.4f}, per_task [{per_task_str}]"
        )

        # Save evaluation losses to JSON
        self._save_evaluation_losses_json(task_losses, weighted_val_loss)

        if was_training:
            self.model.train()
        return task_losses, weighted_val_loss

    @torch.inference_mode()
    def test(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model over a single-task dataloader and return concatenated logits and true values."""
        was_training = self.model.training
        self.model.eval()

        logits_chunks: list[torch.Tensor] = []
        true_chunks: list[torch.Tensor] = []
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            batch_device = batch.to(self.device)
            with self.ctx:
                outputs = self.model(
                    input_ids=batch_device.input_ids,
                    attention_mask=batch_device.attention_mask,
                    task=batch_device.task,
                )
            logits_chunks.append(outputs.logits.detach().float().to(device="cpu"))

            # Collect true values (prefer targets over labels)
            true_values = (
                batch_device.targets
                if batch_device.targets is not None
                else batch_device.labels
            )
            if true_values is not None:
                true_chunks.append(true_values.detach().float().to(device="cpu"))

        if was_training:
            self.model.train()

        if not logits_chunks:
            return torch.empty((0,), dtype=torch.float32), torch.empty(
                (0,), dtype=torch.float32
            )

        predicted = torch.cat(logits_chunks, dim=0)
        true = (
            torch.cat(true_chunks, dim=0)
            if true_chunks
            else torch.empty((0,), dtype=torch.float32)
        )
        return {"predicted": predicted, "true": true}

    def _infer_total_num_iters(
        self, train_loaders: dict[str, DataLoader]
    ) -> tuple[int, int]:
        """Calculate total iterations: batches_per_epoch = min(len(loader_i))."""
        batches_per_epoch = max(
            1,
            min(len(loader) for loader in train_loaders.values())
        )
        return self.config.max_epochs * batches_per_epoch, batches_per_epoch

    def train(
        self,
        train_loaders: dict[str, DataLoader],
        val_loaders: dict[str, DataLoader] | None = None,
    ) -> None:
        """Training loop for multitask learning.

        Tasks are sampled according to config.tasks weights.
        One epoch = Σ(prob_task_i × length_task_i) batches.
        """
        # Early return if already completed training
        if self.state.epoch >= self.config.max_epochs:
            return

        # Validate that all config tasks have corresponding loaders
        task_names = list(train_loaders.keys())
        if set(task_names) != set(self.config.tasks.keys()):
            raise ValueError(
                f"Task mismatch: Config tasks: {set(self.config.tasks.keys())}, Loader tasks: {set(task_names)}"
            )

        # Infer total iterations using sampler's normalized probabilities
        self.state.total_iters, batches_per_epoch = self._infer_total_num_iters(
            train_loaders
        )
        self.state.warmup_iters = int(self.config.warmup_ratio * self.state.total_iters)

        # Initialize training state
        not_improved = 0
        self.model.train()
        torch.cuda.empty_cache()

        # Initialize logging
        global_iter = 0
        log_start_time = time.time()
        log_tokens = 0

        # Main training loop over epochs
        while self.state.epoch < self.config.max_epochs:
            # Initialize epoch-level tracking
            batches_in_epoch = 0

            # Initialize iterators for each task's loader (reset at start of each epoch)
            self.task_iterators = {
                task: iter(train_loaders[task]) for task in task_names
            }

            # Sample tasks and train for one epoch
            while batches_in_epoch < batches_per_epoch:
                # Sample a task name according to task probabilities
                task_name = self.task_sampler.sample()

                # Get next batch for the sampled task (explicitly handles iterator cycling)
                batch = self._get_next_batch(task_name, train_loaders)

                self.state.iterations_in_epoch = batches_in_epoch
                self._update_lr(global_iter)

                # Perform training step
                loss_val, grad_norm, num_tokens = self.train_step(batch)
                last_loss = loss_val
                last_grad_norm = grad_norm

                # Track metrics for throughput
                log_tokens += num_tokens
                global_iter += 1
                batches_in_epoch += 1

                # Logging
                should_log = (
                    self.config.log_every_n_iterations is not None
                    and global_iter % self.config.log_every_n_iterations == 0
                ) or (
                    self.config.log_every_n_iterations is None
                    and batches_in_epoch == batches_per_epoch
                )
                if should_log:
                    elapsed = time.time() - log_start_time
                    tokens_per_sec = log_tokens / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        f"Step {global_iter}: "
                        f"task {task_name}, "
                        f"loss {last_loss:.4f}, "
                        f"grad_norm {last_grad_norm:.4f}, "
                        f"lr {self.state.current_lr:.6f}, "
                        f"tokens/s {tokens_per_sec:.0f}"
                    )

                    log_start_time = time.time()
                    log_tokens = 0

            # Validation
            if val_loaders:
                task_val_losses, weighted_val_loss = self.evaluate(val_loaders)

                if weighted_val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = weighted_val_loss
                    self.state.best_epoch = self.state.epoch
                    if self.out_dir:
                        # Save trainer checkpoint (optimizer + trainer state)
                        best_trainer_path = (
                            self.out_dir / self.BEST_CHECKPOINT_FILE_NAME
                        )
                        self.save_trainer_checkpoint(best_trainer_path)
                        # Update best model checkpoint (only when it's actually the best)
                        self.model.save_pretrained(
                            self.out_dir / self.BEST_MODEL_CHECKPOINT_FILE_NAME
                        )
                    not_improved = 0
                    logger.info(
                        f"New best weighted validation loss: {self.state.best_val_loss:.4f}"
                    )
                else:
                    not_improved += 1

                # Early stopping
                if self.config.patience and not_improved >= self.config.patience:
                    logger.info(f"Early stopping at epoch {self.state.epoch}")
                    break

            # Periodic checkpoint (in addition to best checkpoint)
            if (
                self.out_dir
                and self.config.checkpoint_every_n_epochs is not None
                and self.state.epoch % self.config.checkpoint_every_n_epochs == 0
            ):
                # Save trainer checkpoint
                ckpt_path = self.out_dir / self.CHECKPOINT_EPOCH_FILE_PATTERN.format(
                    epoch=self.state.epoch
                )
                self.save_trainer_checkpoint(ckpt_path)
                # Save model checkpoint
                model_path = self.out_dir / self.MODEL_EPOCH_FILE_PATTERN.format(
                    epoch=self.state.epoch
                )
                self.model.save_pretrained(model_path)

            self.state.epoch += 1

        # optionally reload best model weights at end
        if val_loaders and self.out_dir:
            best_model_path = self.out_dir / self.BEST_MODEL_CHECKPOINT_FILE_NAME
            if best_model_path.exists():
                self.model.load_pretrained(
                    best_model_path, device=self.device, strict=True
                )
                
