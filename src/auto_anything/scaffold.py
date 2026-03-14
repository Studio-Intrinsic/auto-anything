from __future__ import annotations

import shutil
from pathlib import Path


def materialize_scaffold(
    *,
    scaffold_dir: Path,
    destination_dir: Path,
    replacements: dict[str, str] | None = None,
) -> None:
    replacements = replacements or {}
    scaffold_dir = scaffold_dir.expanduser().resolve()
    destination_dir = destination_dir.expanduser().resolve()
    if not scaffold_dir.is_dir():
        raise FileNotFoundError(f"Scaffold directory does not exist: {scaffold_dir}")

    for source_path in scaffold_dir.rglob("*"):
        relative = source_path.relative_to(scaffold_dir)
        target_path = destination_dir / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix in {".py", ".md", ".json", ".txt"}:
            content = source_path.read_text(encoding="utf-8")
            for key, value in replacements.items():
                content = content.replace(key, value)
            target_path.write_text(content, encoding="utf-8")
        else:
            shutil.copy2(source_path, target_path)
