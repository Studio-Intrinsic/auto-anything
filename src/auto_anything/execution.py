from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .models import ExecutionBackendConfig, ExecutionBackendKind, ExecutionResult, RunCommand, TaskCharter


def _resolve_env(config: ExecutionBackendConfig, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    if config.inherit_env:
        env.update({key: value for key, value in os.environ.items() if key in config.env_allowlist})
    if extra_env:
        env.update(extra_env)
    return env


def _resolve_working_dir(task_root: Path, command: RunCommand) -> Path:
    return (task_root / command.working_dir).resolve()


def _copy_workspace(source_root: Path, destination_root: Path) -> None:
    shutil.copytree(
        source_root,
        destination_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def _sync_back_paths(source_root: Path, destination_root: Path, sync_back_paths: tuple[str, ...]) -> tuple[str, ...]:
    synced: list[str] = []
    for relative in sync_back_paths:
        rel = relative.strip().strip("/")
        if not rel:
            continue
        source_path = source_root / rel
        if not source_path.exists():
            continue
        destination_path = destination_root / rel
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.is_file():
            destination_path.unlink()
        elif destination_path.is_dir():
            shutil.rmtree(destination_path)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        synced.append(rel)
    return tuple(synced)


class ExecutionBackend:
    def run(
        self,
        *,
        task_root: Path,
        command: RunCommand,
        config: ExecutionBackendConfig,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        raise NotImplementedError


class DirectSubprocessBackend(ExecutionBackend):
    def run(
        self,
        *,
        task_root: Path,
        command: RunCommand,
        config: ExecutionBackendConfig,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        env = _resolve_env(config, extra_env)
        working_dir = _resolve_working_dir(task_root, command)
        t0 = time.perf_counter()
        completed = subprocess.run(
            list(command.command),
            cwd=str(working_dir),
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=command.timeout_seconds,
        )
        duration = time.perf_counter() - t0
        return ExecutionResult(
            command_name=command.name,
            command=command.command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            backend_kind=ExecutionBackendKind.DIRECT_SUBPROCESS,
            working_dir=str(working_dir),
            notes=config.notes,
        )


class IsolatedWorkspaceBackend(ExecutionBackend):
    def run(
        self,
        *,
        task_root: Path,
        command: RunCommand,
        config: ExecutionBackendConfig,
        extra_env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        env = _resolve_env(config, extra_env)
        with tempfile.TemporaryDirectory(prefix="auto-anything-exec-") as tmpdir:
            isolated_root = Path(tmpdir) / "workspace"
            _copy_workspace(task_root, isolated_root)
            working_dir = _resolve_working_dir(isolated_root, command)
            t0 = time.perf_counter()
            completed = subprocess.run(
                list(command.command),
                cwd=str(working_dir),
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=command.timeout_seconds,
            )
            duration = time.perf_counter() - t0
            synced_paths = _sync_back_paths(isolated_root, task_root, config.sync_back_paths)
            return ExecutionResult(
                command_name=command.name,
                command=command.command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_seconds=duration,
                backend_kind=ExecutionBackendKind.ISOLATED_WORKSPACE,
                working_dir=str(working_dir),
                isolated_workspace=str(isolated_root),
                synced_paths=synced_paths,
                notes=config.notes,
            )


def create_execution_backend(config: ExecutionBackendConfig) -> ExecutionBackend:
    if config.kind == ExecutionBackendKind.ISOLATED_WORKSPACE:
        return IsolatedWorkspaceBackend()
    return DirectSubprocessBackend()


def select_run_command(charter: TaskCharter, *, command_name: str | None = None) -> RunCommand:
    if not charter.run_commands:
        raise ValueError("TaskCharter.run_commands must not be empty.")
    if command_name is None:
        return charter.run_commands[0]
    for command in charter.run_commands:
        if command.name == command_name:
            return command
    raise ValueError(f"Unknown run command: {command_name}")


def run_task_command(
    *,
    task_root: Path,
    charter: TaskCharter,
    command_name: str | None = None,
    extra_env: dict[str, str] | None = None,
    force_backend: ExecutionBackendKind | None = None,
) -> ExecutionResult:
    command = select_run_command(charter, command_name=command_name)
    backend_config = charter.execution_backend
    if force_backend is not None:
        backend_config = replace(backend_config, kind=force_backend)
    backend = create_execution_backend(backend_config)
    return backend.run(
        task_root=task_root.expanduser().resolve(),
        command=command,
        config=backend_config,
        extra_env=extra_env,
    )
