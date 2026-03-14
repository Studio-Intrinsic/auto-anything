from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_anything.execution import run_task_command
from auto_anything.models import (
    AgentRuntimeConfig,
    ExecutionBackendConfig,
    ExecutionBackendKind,
    ObjectiveBrief,
    RunCommand,
    WorkspaceLayout,
)
from auto_anything.compiler import DefaultObjectiveCompiler


class ExecutionTests(unittest.TestCase):
    def test_isolated_workspace_backend_syncs_artifacts_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir) / "task"
            task_root.mkdir(parents=True, exist_ok=True)
            (task_root / "artifacts").mkdir(parents=True, exist_ok=True)
            script = task_root / "write_eval.py"
            script.write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import json",
                        "root = Path('.')",
                        "payload = {'ok': True, 'cwd': str(root.resolve())}",
                        "(root / 'artifacts' / 'eval_summary.json').write_text(json.dumps(payload), encoding='utf-8')",
                        "print('done')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            brief = ObjectiveBrief(
                title="Execution test",
                objective_statement="Run eval in isolation.",
                run_commands=(RunCommand(name="evaluate", command=("python3", "write_eval.py")),),
                workspace_layout=WorkspaceLayout(),
                execution_backend=ExecutionBackendConfig(
                    kind=ExecutionBackendKind.ISOLATED_WORKSPACE,
                    sync_back_paths=("artifacts",),
                ),
                agent_runtime=AgentRuntimeConfig(),
            )
            charter = DefaultObjectiveCompiler().compile(brief)

            result = run_task_command(task_root=task_root, charter=charter, command_name="evaluate")

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.backend_kind, ExecutionBackendKind.ISOLATED_WORKSPACE)
            self.assertEqual(result.synced_paths, ("artifacts",))
            summary = json.loads((task_root / "artifacts" / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["ok"])
            self.assertNotEqual(summary["cwd"], str(task_root.resolve()))


if __name__ == "__main__":
    unittest.main()
