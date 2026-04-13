import sys
from pathlib import Path


def ensure_project_root() -> str:
    current_file = Path(__file__).resolve()

    for candidate in [current_file.parent] + list(current_file.parents):
        src_dir = candidate / "src"
        inference_dir = src_dir / "inference"
        data_dir = candidate / "data"

        if src_dir.is_dir() and inference_dir.is_dir() and data_dir.is_dir():
            project_root = str(candidate)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            return project_root

    fallback_root = str(current_file.resolve().parents[1])
    if fallback_root not in sys.path:
        sys.path.insert(0, fallback_root)
    return fallback_root
