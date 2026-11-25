"""Low-level numerical kernels, loaded explicitly by their callers."""

from importlib.util import find_spec

HAS_TRITON = find_spec("triton") is not None
