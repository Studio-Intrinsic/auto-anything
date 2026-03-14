from __future__ import annotations

from .models import TaskCharter


def _render_list(items: tuple[str, ...], *, empty: str = "- none") -> str:
    if not items:
        return empty
    return "\n".join(f"- `{item}`" for item in items)


def render_task_agents_md(*, charter: TaskCharter, task_name: str, iteration_command: str = "") -> str:
    run_commands = tuple(" ".join(command.command) for command in charter.run_commands)
    mutable_paths = charter.search_surface.mutable_paths
    protected_paths = charter.search_surface.protected_paths
    anti_goals = charter.anti_goals
    hard_constraints = tuple(
        constraint.statement for constraint in charter.hard_constraints if constraint.statement not in set(anti_goals)
    )
    artifact_paths = (
        f"{charter.workspace_layout.artifacts_dir}/eval_summary.json",
        f"{charter.workspace_layout.artifacts_dir}/experiment_history.json",
        f"{charter.workspace_layout.artifacts_dir}/knowledge_base.md",
        f"{charter.workspace_layout.artifacts_dir}/experiments/",
        f"{charter.workspace_layout.artifacts_dir}/progress_curve.svg",
    )
    subsystem_lines = []
    for subsystem in charter.search_surface.subsystems:
        subsystem_lines.append(f"- `{subsystem.subsystem_id}`: {subsystem.summary}")
        if subsystem.owned_paths:
            subsystem_lines.append(f"  owned paths: {', '.join(f'`{path}`' for path in subsystem.owned_paths)}")
        if subsystem.primary_signals:
            subsystem_lines.append(f"  primary signals: {', '.join(f'`{name}`' for name in subsystem.primary_signals)}")
        if subsystem.guardrail_signals:
            subsystem_lines.append(f"  guardrails: {', '.join(f'`{name}`' for name in subsystem.guardrail_signals)}")
    subsystem_block = "\n".join(subsystem_lines) if subsystem_lines else "- none declared"

    role_lines = "\n".join(f"- `{role.role.value}`: {role.goal}" for role in charter.agent_loop.passes)
    soft_constraints = tuple(constraint.statement for constraint in charter.soft_constraints)
    focus_subsystems = charter.focus_subsystems or ("none declared",)
    iteration_command = iteration_command or "python3 <path-to-run_task_iteration.py> --task-root <task_root> --hypothesis ... --change-summary ..."

    return (
        f"# AGENTS\n\n"
        f"## Task\n"
        f"- task: `{task_name}`\n"
        f"- charter id: `{charter.charter_id}`\n"
        f"- objective: {charter.objective_statement}\n"
        f"- default model: `{charter.agent_runtime.default_model or 'unspecified'}`\n"
        f"- allowed models:\n{_render_list(charter.allowed_models, empty='- use charter default')}\n\n"
        f"## Start Here\n"
        f"- read `task_charter.json`\n"
        f"- inspect the current experiment context in `{charter.workspace_layout.artifacts_dir}/knowledge_base.md`\n"
        f"- inspect prior runs in `{charter.workspace_layout.artifacts_dir}/experiment_history.json` and local git tags `aa-exp-*`\n"
        f"- work inside the mutable surface first\n"
        f"- keep protected paths stable unless the charter clearly requires otherwise\n\n"
        f"## Mutable Surface\n"
        f"{_render_list(mutable_paths)}\n\n"
        f"## Protected Paths\n"
        f"{_render_list(protected_paths)}\n\n"
        f"## Run Commands\n"
        f"{_render_list(run_commands)}\n\n"
        f"## Agent Loop\n"
        f"{role_lines}\n\n"
        f"## Counterbalance\n"
        f"- mode: `{charter.counterbalance.mode.value}`\n"
        f"- block on severity: `{charter.counterbalance.block_on_severity.value}`\n"
        f"- do not accept a local metric gain if the critic finds gaming, brittle assumptions, or architecture regressions\n\n"
        f"## Focus\n"
        f"{_render_list(focus_subsystems)}\n\n"
        f"## Subsystems\n"
        f"{subsystem_block}\n\n"
        f"## Hard Constraints\n"
        f"{_render_list(hard_constraints)}\n\n"
        f"## Soft Constraints\n"
        f"{_render_list(soft_constraints)}\n\n"
        f"## Anti-Goals\n"
        f"{_render_list(charter.anti_goals)}\n\n"
        f"## Decomposition Hints\n"
        f"{_render_list(charter.decomposition_hints)}\n\n"
        f"## Artifacts To Check After Each Run\n"
        f"{_render_list(artifact_paths)}\n\n"
        f"## Workflow\n"
        f"- make a focused change\n"
        f"- run the main eval command\n"
        f"- inspect metrics, critic findings, and diff against prior accepted work\n"
        f"- prefer modular changes over monolithic rewrites\n"
        f"- record the next authoritative iteration with `{iteration_command}`\n"
    )


__all__ = ["render_task_agents_md"]
