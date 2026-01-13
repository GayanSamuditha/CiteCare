"""
Checksum utilities for incremental indexing.
"""

import hashlib
from pathlib import Path
from typing import Dict


def file_sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_checksums(directory: Path) -> Dict[str, str]:
    """Compute checksums for all files in a directory (recursive)."""
    checksums: Dict[str, str] = {}
    for file in directory.rglob("*"):
        if file.is_file():
            checksums[str(file.relative_to(directory))] = file_sha256(file)
    return checksums
