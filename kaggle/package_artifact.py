from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable Kaggle upload ZIP.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "agent-routing-recsys-artifact-kaggle.zip"),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            archive_name = (Path(root.name) / path.relative_to(root)).as_posix()
            archive.write(path, archive_name)

    print(f"Created {output} with {len(files)} files ({output.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
