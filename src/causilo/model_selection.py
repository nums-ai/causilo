"""Resolve checkpoint identity and runtime before fitting an estimator."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_VERSION = "1.5"
LEGACY_REPOSITORY = "nums-ai/causilo"
LEGACY_REVISION = "94f2bd91db0737d4da59f347910662905ecb5a09"
LEGACY_FORMAT = "causilo-v1.0"
RUNTIME15_FORMAT = "causilo-v1.5"
RUNTIME15_REPOSITORY = "nums-ai/causilo-1.5"
RUNTIME15_REVISION = "4ca2275bae85014a19422e8c18fcf8f24c5c14e1"


@dataclass(frozen=True)
class ModelSource:
    repository: str
    revision: str
    architecture: str
    folder: str = ""


MODEL_SOURCES = {
    ("1.0", task): ModelSource(LEGACY_REPOSITORY, LEGACY_REVISION, LEGACY_FORMAT, folder)
    for task, folder in (("classification", "classifier"), ("regression", "regressor"))
}
MODEL_SOURCES.update({
    ("1.5", task): ModelSource(RUNTIME15_REPOSITORY, RUNTIME15_REVISION, RUNTIME15_FORMAT, folder)
    for task, folder in (("classification", "classifier"), ("regression", "regressor"))
})


@dataclass(frozen=True)
class LocalCheckpoint:
    weights: Path
    config_bytes: bytes
    config: dict

    @property
    def architecture(self) -> str | None:
        return self.config.get("architecture")


CheckpointSource = ModelSource | LocalCheckpoint


def checkpoint_files(source: CheckpointSource) -> tuple[Path, ...]:
    """Find the fitted source files without downloading anything."""
    if isinstance(source, LocalCheckpoint):
        return source.weights, source.weights.with_name("config.json")

    from huggingface_hub import try_to_load_from_cache

    prefix = f"{source.folder}/" if source.folder else ""
    paths = (
        try_to_load_from_cache(source.repository, f"{prefix}{name}", revision=source.revision)
        for name in ("config.json", "model.safetensors")
    )
    return tuple(Path(path) for path in paths if isinstance(path, str))


def read_local_checkpoint(path):
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("checkpoint_path must be None or a path")
    weights = Path(path).expanduser().absolute()
    if weights.is_dir():
        weights = weights / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {weights}")
    config = weights.with_name("config.json")
    if not config.is_file():
        raise FileNotFoundError(f"Checkpoint configuration not found beside the weights: {config}")
    encoded = config.read_bytes()
    return LocalCheckpoint(weights, encoded, json.loads(encoded))


def local_checkpoint_identity(checkpoint: LocalCheckpoint, *, expected_digest=None) -> dict[str, str]:
    """Fingerprint the configuration and weights used for this fit."""
    digest = hashlib.sha256(checkpoint.config_bytes)
    with checkpoint.weights.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if expected_digest is not None and observed != expected_digest:
        raise ValueError("Local checkpoint differs from the weights used to fit the saved estimator")
    return {"path": str(checkpoint.weights), "digest": observed}


def select_checkpoint(version, task, path=None) -> CheckpointSource:
    if not isinstance(version, str) or version not in ("1.0", "1.5"):
        raise ValueError("model_version must be '1.0' or '1.5'")
    if path is None:
        return MODEL_SOURCES[version, task]
    checkpoint = read_local_checkpoint(path)
    if checkpoint.architecture not in (LEGACY_FORMAT, RUNTIME15_FORMAT):
        raise ValueError("Unsupported model configuration")
    return checkpoint


def runtime_for(architecture):
    if architecture == LEGACY_FORMAT:
        from .v1_0 import runtime
    elif architecture == RUNTIME15_FORMAT:
        from .v1_5 import runtime
    else:
        raise ValueError("Unsupported model configuration")
    return runtime
