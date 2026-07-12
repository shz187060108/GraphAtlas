#!/usr/bin/env python
from __future__ import annotations

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import PROJECT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path, field: str) -> list[str]:
    values: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if field not in row:
            raise KeyError(f"{path}:{line_number} is missing field {field!r}")
        values.append(str(row[field]))
    return values


def _extract_npz_text(raw: np.lib.npyio.NpzFile) -> list[str]:
    key = next((name for name in ("texts", "raw_text", "text") if name in raw.files), None)
    if key is None:
        raise ValueError("NPZ contains no texts, raw_text, or text array")
    values = np.asarray(raw[key], dtype=object).reshape(-1)
    return ["" if value is None else str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed raw node text and cache text_features in a GraphAtlas NPZ.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<name>/raw/<name>.npz")
    parser.add_argument("--npz", default=None, help="Override the standardized NPZ path")
    parser.add_argument("--jsonl", default=None, help="Optional external JSONL with one text record per node")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    dataset = args.dataset.lower().replace("-", "_")
    path = Path(args.npz) if args.npz else PROJECT_ROOT / "data" / dataset / "raw" / f"{dataset}.npz"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as raw:
        payload = {key: np.asarray(raw[key]) for key in raw.files}
        texts = _read_jsonl(Path(args.jsonl), args.text_field) if args.jsonl else _extract_npz_text(raw)
        num_nodes = int(np.asarray(raw["node_features"]).shape[0])
    if len(texts) != num_nodes:
        raise ValueError(f"Text count {len(texts)} does not match node count {num_nodes}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install text support with `pip install -e .[text]`.") from error

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    ).astype(np.float32, copy=False)
    payload["text_features"] = embeddings

    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)

    manifest = {
        "dataset": dataset,
        "model": args.model,
        "batch_size": args.batch_size,
        "device": args.device,
        "normalize_embeddings": args.normalize,
        "num_nodes": num_nodes,
        "embedding_dim": int(embeddings.shape[1]),
        "source_text_sha256": _hash_strings(texts),
        "embedding_sha256": _sha256_array(embeddings),
        "npz": str(path),
    }
    manifest_path = path.parent / "text_embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
