"""Learn query gains from context length and content within fixed multiplicative bounds."""

import math

from torch import Tensor, nn


def bounded_multiplier(value: Tensor, limit: float) -> Tensor:
    """Map unconstrained values into (1/limit, limit) through bounded log-space gains."""
    radius = math.log(limit)
    return (radius * (value.float() / radius).tanh()).exp().to(value.dtype)


class QueryScale(nn.Module):
    """Scale each query head using log context length and its own projected content."""

    def __init__(self, heads: int, head_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        self.length = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, heads * head_width))
        self.correction = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, heads * head_width))
        self.content = nn.Sequential(nn.Linear(head_width, 64), nn.GELU(), nn.Linear(64, head_width))

    def forward(self, query: Tensor, key_count: int | Tensor) -> Tensor:
        """Preserve (*batch, heads, queries, head_width) using checkpoint-defined gains."""
        if isinstance(key_count, Tensor):
            # Compute the logarithm in FP32 before mixed-precision conversion.
            log_count = key_count.float().clamp_min(1).log().to(query).unsqueeze(-1)
            gain_shape = (*key_count.shape, self.heads, 1, self.head_width)
        else:
            log_count = query.new_tensor([[math.log(max(key_count, 1))]])
            gain_shape = (self.heads, 1, self.head_width)
        # These bounds belong to the model computation, not memory/precision
        # tuning knobs. Changing them changes the pretrained model's outputs.
        length_gain = bounded_multiplier(self.length(log_count), 8.0)
        length_gain = length_gain * bounded_multiplier(self.correction(log_count), 2.0)
        content_gain = bounded_multiplier(self.content(query), 2.0)
        return query * (length_gain.reshape(gain_shape) * content_gain)
