import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax


class GeneratorMixin:
    """Mixin for generator classes."""

    def _transform_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        logits = self._scale_logits(logits, temperature)
        logits = self._apply_top_k(logits, top_k)
        logits = self._apply_top_p(logits, top_p)
        return logits

    def _compute_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return log_softmax(logits, dim=-1)

    def _compute_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return softmax(logits, dim=-1)

    def _scale_logits(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 1.0:
            return logits
        return logits / temperature

    def _apply_top_k(self, logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
        if top_k is None or top_k <= 0 or top_k > logits.size(-1):
            return logits
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.clone()
        # Threshold against the kth logit along the last dimension (not sequence dim)
        logits[logits < v[..., -1:]] = -float("Inf")
        return logits

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p is None or top_p <= 0.0 or top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = softmax(sorted_logits, dim=-1)

        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumprobs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

        return logits

    def _to_numpy(self, logits: torch.Tensor) -> np.ndarray:
        if logits.is_cuda:
            logits = logits.detach()
        return logits.cpu().numpy()
