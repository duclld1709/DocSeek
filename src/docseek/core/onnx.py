"""ONNX Runtime execution providers for fastembed models (embedders, rerankers, CLIP).

``DOCSEEK_ONNX_PROVIDERS``: ``auto`` (default: DirectML GPU when the installed onnxruntime
has it, else CPU), ``cpu``, or ``dml``. Windows installs ``onnxruntime-directml``; Linux/CI use
the CPU build, where ``auto`` resolves to CPU.
"""

from __future__ import annotations

import os
from functools import cache


@cache
def onnx_providers() -> tuple[str, ...]:
    import onnxruntime as ort

    choice = os.environ.get("DOCSEEK_ONNX_PROVIDERS", "auto").lower()
    available = set(ort.get_available_providers())
    if choice == "cpu":
        return ("CPUExecutionProvider",)
    if choice == "dml" and "DmlExecutionProvider" not in available:
        raise RuntimeError("DOCSEEK_ONNX_PROVIDERS=dml but onnxruntime has no DirectML provider")
    if "DmlExecutionProvider" in available and choice in ("auto", "dml"):
        return ("DmlExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def device_label() -> str:
    return "gpu-directml" if onnx_providers()[0] == "DmlExecutionProvider" else "cpu"
