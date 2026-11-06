"""Optional MLflow.

Training was tracked with MLflow on the cluster.  Reproducing the models does
not need it, so importing it is made optional: without the package installed
every call becomes a no-op and training runs unchanged.
"""
from __future__ import annotations

import contextlib


class _NullMlflow:
    """Absorbs every mlflow call, including `start_run` used as a context manager."""

    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return contextlib.nullcontext()
        return _noop


try:  # pragma: no cover - depends on the environment
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = _NullMlflow()
