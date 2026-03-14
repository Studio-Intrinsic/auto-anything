from __future__ import annotations

import base64
import subprocess
from pathlib import Path


def extract_text(pdf_path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def render_pdf_pages(pdf_path: Path, output_dir: Path, *, max_pages: int = 4) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "page"
    subprocess.run(
        [
            "pdftoppm",
            "-jpeg",
            "-r",
            "144",
            "-f",
            "1",
            "-l",
            str(max_pages),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(output_dir.glob("page-*.jpg"))


def image_to_data_url(image_path: Path) -> str:
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
