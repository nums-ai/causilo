"""Learn query gains from context length and projected content."""

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import Tensor, nn

from ...kernels import HAS_TRITON
from ..frequency import FrequencyWeights

LENGTH_GAIN_LIMIT = 8.0
CONTENT_GAIN_LIMIT = 2.0

# A fitted runner supplies its own cache only while executing on its CUDA stream.
_length_gains = ContextVar("causilo_length_gains", default=None)


@contextmanager
def reuse_length_gains(cache):
    """Reuse at most one length per layer within the current fitted runner."""
    token = _length_gains.set(cache)
    try:
        yield
    finally:
        _length_gains.reset(token)


def bounded_multiplier(value: Tensor, limit: float) -> Tensor:
    """Map unconstrained values into (1 / limit, limit) in log space."""
    radius = math.log(limit)
    return (radius * (value.float() / radius).tanh()).exp().to(value.dtype)


class QueryScale(nn.Module):
    """Scale each head using its log context length and query content."""

    def __init__(self, heads: int, head_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        self.length = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, heads * head_width))
        self.content = nn.Sequential(nn.Linear(head_width, 64), nn.GELU(), nn.Linear(64, head_width))

    def _length_gain(self, query: Tensor, key_count: int | Tensor | FrequencyWeights) -> Tensor:
        cache = None
        if query.is_cuda and isinstance(key_count, (int, FrequencyWeights)) and not torch.compiler.is_compiling():
            cache = _length_gains.get()
        if cache is not None:
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
            key = (key_count, query.dtype, dtype)
            cached = cache.get(self)
            if cached is not None and cached[0] == key:
                return cached[1]

        if isinstance(key_count, FrequencyWeights):
            key_count = key_count.total
        if isinstance(key_count, Tensor):
            log_count = key_count.to(device=query.device, dtype=torch.float32).clamp_min(1).log().to(query.dtype)
        else:
            log_count = query.new_tensor(math.log(max(key_count, 1)))
        log_count = log_count.reshape(-1, 1)
        if log_count.shape[0] not in (1, query.shape[0]):
            raise ValueError(f"Dense key lengths contain {log_count.shape[0]} values, expected one or {query.shape[0]}.")

        gain = bounded_multiplier(self.length(log_count), LENGTH_GAIN_LIMIT).view(-1, self.heads, 1, self.head_width)
        if cache is not None:
            cache[self] = (key, gain)
        return gain

    def forward(self, query: Tensor, key_count: int | Tensor | FrequencyWeights) -> Tensor:
        """Scale packed queries (batch, heads, queries, head_width)."""
        length_gain = self._length_gain(query, key_count)
        content = self.content(query)
        if (
            HAS_TRITON and query.is_cuda and torch.version.hip is None and not torch.is_grad_enabled()
            and query.dtype == length_gain.dtype == content.dtype == torch.float16 and query.numel()
            and query.ndim == 4 and query.stride(-1) == 1 and content.is_contiguous() and length_gain.is_contiguous()
            and (query.is_contiguous() or query.transpose(1, 2).is_contiguous())
        ):
            from ...kernels.triton.query_gain import apply_query_gain

            return apply_query_gain(query, length_gain, content, CONTENT_GAIN_LIMIT)
        return query * (length_gain * bounded_multiplier(content, CONTENT_GAIN_LIMIT))
