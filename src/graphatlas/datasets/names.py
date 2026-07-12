from __future__ import annotations


def is_atlas_het_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized == "atlas_het" or normalized.startswith("atlas_het_")
