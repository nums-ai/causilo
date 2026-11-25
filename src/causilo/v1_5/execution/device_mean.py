"""Reduce trimmed regression means before copying them to the host."""

import numpy as np
import torch

from ..kernels import HAS_TRITON
from .memory import is_allocation_error
from .outputs import _MEAN_MASS, _MEAN_WEIGHTS, MEAN_LEVELS


def _resident(cache, device):
    tensors = (cache.column_ecdf_context, cache.column_ecdf_cumulative)
    return all(tensor is None or tensor.device == device for tensor in tensors) and all(
        pair.key.device == pair.value.device == device
        for block in (cache.columns, cache.prediction)
        for pair in block.layers.values()
    )


def trimmed_mean_reducer(runner, dataset, caches, alignment):
    """Select a single full CUDA group and prepare its small reduction inputs lazily."""
    if (
        not HAS_TRITON or runner.device.type != "cuda" or torch.version.hip is not None
        or torch.cuda.get_device_capability(runner.device)[0] != 9
    ):
        return None
    signs = dataset.ensemble.target_signs
    if len(signs) != 8 or dataset.ensemble.trim_count != 2:
        return None
    if caches is not None and (len(caches) != 1 or not _resident(caches[0], runner.device)):
        return None
    try:
        from ..kernels.triton.regression_mean import trimmed_mean
    except ImportError:
        return None

    scaler = dataset.target_preprocessor.encoder_
    scale, offset = scaler.scale_[0], scaler.mean_[0]
    # Stay with host integration when rounding a large offset could consume a
    # float32 step of the fitted target scale.
    if abs(offset) * np.finfo(np.float64).eps * len(MEAN_LEVELS) > scale * np.finfo(np.float32).eps:
        return None
    indices = np.arange(len(MEAN_LEVELS), dtype=np.int32)
    forward, reverse = (indices[index] if isinstance(index, slice) else index for index in alignment)
    parameters = (
        scale, offset, *dataset.target_limits,
        _MEAN_MASS[0] * 0.5, _MEAN_MASS[-1] * 0.5,
    )
    inputs = None

    def reduce(outputs):
        nonlocal inputs
        if (
            outputs.device != runner.device or outputs.dtype != torch.float32 or outputs.ndim != 3
            or outputs.shape[0] != 8 or outputs.shape[2] != len(MEAN_LEVELS)
        ):
            return None
        rows = outputs.shape[1]
        if rows == 0 or rows * 8 + 32 * len(MEAN_LEVELS) > runner._working_memory():
            return None
        try:
            if inputs is None:
                inputs = (
                    torch.tensor(forward.copy(), device=outputs.device, dtype=torch.int32),
                    torch.tensor(reverse.copy(), device=outputs.device, dtype=torch.int32),
                    torch.tensor(signs, device=outputs.device),
                    torch.tensor(_MEAN_WEIGHTS, device=outputs.device),
                    torch.tensor(parameters, device=outputs.device, dtype=torch.float64),
                )
            result = torch.empty(rows, dtype=torch.float64, device=outputs.device)
            trimmed_mean[(rows,)](
                outputs, *inputs, result, *outputs.stride(),
                len(MEAN_LEVELS), 1024, num_warps=8, enable_fp_fusion=False,
            )
            return result
        except (MemoryError, RuntimeError) as error:
            if not is_allocation_error(error):
                raise
            inputs = None
            return None

    return reduce
