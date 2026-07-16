from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Some sparse reductions are not strictly deterministic on every CUDA
        # build. warn_only keeps the run usable while surfacing any violation.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def configure_torch_threads(num_threads: int) -> None:
    """Set conservative CPU thread counts for small sparse graph workloads."""
    import torch

    if num_threads <= 0:
        return
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(num_threads, 4)))
    except RuntimeError:
        # Inter-op threads can only be set before parallel work starts.
        pass


def resolve_device(name: str = "auto") -> "torch.device":
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.lower().startswith("cuda") and not torch.cuda.is_available():
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print("GPU name: unavailable")
        raise RuntimeError("CUDA was explicitly requested but is not available; CPU fallback is disabled")
    return torch.device(name)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _atomic_text_write(text: str, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def save_yaml(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    _atomic_text_write(text, path)


def save_json(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    _atomic_text_write(text, path)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def count_parameters(module: "torch.nn.Module") -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def stable_hash(data: Any, length: int = 12) -> str:
    """Return a stable short hash for nested JSON-compatible configuration."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
