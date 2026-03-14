from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.models import WorkspaceLayout
from auto_anything.workspace import resolve_workspace_paths


class WorkspaceTests(unittest.TestCase):
    def test_resolves_expected_agent_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            layout = WorkspaceLayout(
                root_dir="work",
                candidate_dir="pipeline",
                evaluator_dir="eval",
                artifacts_dir="artifacts",
                replay_dir="replay",
            )

            paths = resolve_workspace_paths(base, layout)
            paths.ensure_exists()

            self.assertEqual(paths.root_dir, (base / "work").resolve())
            self.assertEqual(paths.candidate_dir, (base / "work" / "pipeline").resolve())
            self.assertTrue(paths.replay_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
