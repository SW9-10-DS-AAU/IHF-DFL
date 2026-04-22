from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found — not in a repo")


# Robust handling of pathing across the codebase, using repo_root as the base for all paths