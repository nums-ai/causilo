"""Save current fitted estimators and restore them with matching weights."""

import os
import pickle
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

from sklearn.exceptions import NotFittedError

from .model_selection import (
    checkpoint_files,
    read_local_checkpoint,
    runtime_for,
    select_checkpoint,
)

STATE_SCHEMA = "causilo-fitted-estimator-4"
_REFIT_MESSAGE = "Unsupported fitted-state format. Refit with the current Causilo installation and save again."


def runtime_versions() -> dict[str, str]:
    """Record the versions that reconstruct fitted Python and tensor objects."""
    dependencies = ("numpy", "pandas", "scipy", "scikit-learn", "torch")
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        **{name: version(name) for name in dependencies},
    }


def fitted_contexts(fitted, context_type):
    """Validate the current single-table or batch record before loading weights."""
    if isinstance(fitted, dict):
        if not fitted or any(
            not isinstance(key, str) or not key or not isinstance(context, context_type)
            for key, context in fitted.items()
        ):
            raise ValueError("Saved batch must contain string IDs and fitted contexts")
        return fitted.values()
    if not isinstance(fitted, context_type):
        raise ValueError("Saved context does not match the checkpoint architecture")
    return (fitted,)


def export_estimator(estimator):
    from . import __version__

    saved = {
        "schema": STATE_SCHEMA,
        "version": __version__,
        "dependencies": runtime_versions(),
        "parameters": estimator.get_params(deep=False),
        "fitted": None,
    }
    engine = getattr(estimator, "_engine", None)
    if engine is None or (engine.state is None and engine.batch_states is None):
        return saved

    saved.update(runtime_for(engine.source.architecture).export_fitted(engine))
    saved.update(
        task=engine.task,
        checkpoint_request=estimator._checkpoint_request,
    )
    return saved


def _restore_source(request, task, checkpoint):
    model_version, path = request
    if path is None:
        source = select_checkpoint(model_version, task)
        if checkpoint != source.revision:
            raise ValueError("Saved state belongs to a different checkpoint. Refit and save again.")
        return source, None

    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("path"), str)
        or not isinstance(checkpoint.get("digest"), str)
        or not checkpoint["digest"]
    ):
        raise ValueError("Saved fitted state is missing its fitted weight identity")
    return read_local_checkpoint(checkpoint["path"]), checkpoint["digest"]


def import_estimator(estimator, saved, **placement):
    from . import __version__
    from .estimators import device_key, invalidate_fit, memory_budget_key

    if not isinstance(saved, dict) or saved.get("schema") != STATE_SCHEMA:
        raise ValueError(_REFIT_MESSAGE)
    if saved["version"] != __version__:
        raise ValueError("Restore requires the same Causilo version")
    if saved["dependencies"] != runtime_versions():
        raise ValueError("Restore requires matching Python and dependency versions")

    invalidate_fit(estimator)
    estimator.__dict__.pop("_engine", None)
    estimator.__dict__.pop("_checkpoint_request", None)
    parameters = saved["parameters"] | placement
    estimator.__dict__.update(parameters)
    if saved["fitted"] is None:
        return

    checkpoint_request = saved["checkpoint_request"]
    source, digest = _restore_source(checkpoint_request, saved["task"], saved.get("checkpoint"))
    runtime = runtime_for(source.architecture)
    engine = runtime.make_engine(
        saved["task"], device_key(parameters["device"]), source,
        memory_budgets=memory_budget_key(parameters["device_memory_budget_bytes"]),
    )
    runtime.restore_fitted(engine, saved, digest)
    estimator._engine = engine
    estimator._checkpoint_request = checkpoint_request
    runtime.publish_fitted_attributes(estimator)


def write_context(estimator, path):
    """Save a fitted context and its checkpoint identity for later prediction."""
    engine = getattr(estimator, "_engine", None)
    if engine is None or (engine.state is None and engine.batch_states is None):
        raise NotFittedError("Call fit successfully before saving a context")
    path = Path(path)
    destination = path.resolve()
    for file in checkpoint_files(engine.source):
        if destination == file.resolve() or (
            destination.exists() and file.exists() and destination.samefile(file)
        ):
            raise ValueError(f"Context output must differ from the fitted checkpoint files: {path}")
    saved = export_estimator(estimator)
    with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
        source = Path(temporary.name)
        try:
            pickle.dump(saved, temporary, protocol=5)
            temporary.close()
            os.replace(source, path)
        finally:
            source.unlink(missing_ok=True)


def read_context(estimator_type, path, *, task, device, device_memory_budget_bytes):
    """Restore a trusted fitted-state file on the requested execution device."""
    try:
        with Path(path).open("rb") as stream:
            saved = pickle.load(stream)
    except (ImportError, AttributeError) as error:
        raise ValueError(_REFIT_MESSAGE) from error
    if not isinstance(saved, dict) or saved.get("schema") != STATE_SCHEMA or saved.get("fitted") is None:
        raise ValueError(_REFIT_MESSAGE)
    if saved.get("task") != task:
        raise TypeError(f"Saved context is for {saved.get('task')!r}, expected {task!r}")
    estimator = estimator_type.__new__(estimator_type)
    import_estimator(estimator, saved, device=device, device_memory_budget_bytes=device_memory_budget_bytes)
    return estimator
