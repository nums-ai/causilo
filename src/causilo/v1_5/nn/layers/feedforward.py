"""Gated feedforward transformation shared by attention and row layers."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...kernels import HAS_TRITON


def _bf16_gated_product(value: Tensor, gate: Tensor) -> Tensor:
    if (
        HAS_TRITON and value.is_cuda and torch.version.hip is None and not torch.is_grad_enabled()
        and value.dtype == gate.dtype == torch.float16 and value.is_contiguous() and gate.is_contiguous()
        and value.numel()
    ):
        from ...kernels.triton.swiglu import gated_product

        return gated_product(value, gate)
    return (F.silu(gate.float()) * value.float()).bfloat16()


class FeedForward(nn.Module):
    """Expand and contract the final hidden-width axis with SwiGLU."""

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.value = nn.Linear(width, width * expansion)
        self.gate = nn.Linear(width, width * expansion)
        self.output = nn.Linear(width * expansion, width)
        self.register_buffer("_output_weight_bf16", None, persistent=False)
        self.register_buffer("_output_bias_bf16", None, persistent=False)

    def prepare_bf16_output(self) -> None:
        """Prepare the fixed output projection once per CUDA replica."""
        self._output_weight_bf16 = self.output.weight.detach().to(torch.bfloat16)
        self._output_bias_bf16 = self.output.bias.detach().to(torch.bfloat16)

    def forward(self, rows: Tensor) -> Tensor:
        if (
            rows.dtype == torch.float32
            and rows.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        ):
            # Share the autocast conversion between the two projections.
            rows = rows.to(torch.float16)
        value = self.value(rows)
        gate = self.gate(rows)
        if value.is_cuda and value.dtype == torch.float16:
            # Keep the gated product in FP32 so it can exceed the FP16 range.
            with torch.autocast("cuda", enabled=False):
                if self._output_weight_bf16 is not None:
                    product = _bf16_gated_product(value, gate)
                    return F.linear(product, self._output_weight_bf16, self._output_bias_bf16).float()
                return self.output(F.silu(gate.float()) * value.float())
        return self.output(F.silu(gate) * value)
