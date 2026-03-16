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
    iteration_command = iteration_command or "auto-anything iterate --task-root <task_root> --hypothesis ... --change-summary ..."
    notes = tuple(note for note in charter.notes if note.strip())
    artifact_lines = []
    for artifact in charter.optimizable_artifacts:
        artifact_lines.append(
            f"- `{artifact.artifact_id}`: kind=`{artifact.kind.value}` location=`{artifact.location or '.'}`"
        )
        if artifact.mutable_paths:
            artifact_lines.append(
                f"  mutable paths: {', '.join(f'`{path}`' for path in artifact.mutable_paths)}"
            )
        if artifact.description:
            artifact_lines.append(f"  description: {artifact.description}")
        if artifact.serialization_hint:
            artifact_lines.append(f"  serialization: `{artifact.serialization_hint}`")
    artifact_block = "\n".join(artifact_lines) if artifact_lines else "- none declared"

    return (
        f"# AGENTS\n\n"
        f"## Task\n"
        f"- task: `{task_name}`\n"
        f"- charter id: `{charter.charter_id}`\n"
        f"- objective: {charter.objective_statement}\n"
        f"- optimization mode: `{charter.optimization_mode.value}`\n"
        f"- search strategy: `{charter.search_strategy.kind.value}` (beam_width=`{charter.search_strategy.beam_width}`)\n"
        f"- default model: `{charter.agent_runtime.default_model or 'unspecified'}`\n"
        f"- allowed models:\n{_render_list(charter.allowed_models, empty='- use charter default')}\n\n"
        f"## What The Framework Gave You\n"
        f"- a typed charter with your objective, constraints, anti-goals, and signals\n"
        f"- a workspace layout with mutable and protected paths\n"
        f"- placeholder pipeline and evaluator files in the mutable surface\n"
        f"- experiment tracking (history, knowledge base, progress curve, git tags)\n"
        f"- a self-critic that checks for protected path violations, monolith growth, and focus drift\n"
        f"- an acceptance engine that compares baseline vs candidate and blocks regressions\n\n"
        f"## What You Must Do (the framework cannot do this for you)\n"
        f"The scaffolded files are generic placeholders. You own making them real.\n\n"
        f"1. **Define your domain subsystems.** The default subsystems are generic. Rename and reshape them to match your actual problem decomposition. Update `task_charter.json` with real subsystem IDs, owned paths, and primary signals.\n"
        f"2. **Get real data into the workspace.** Copy, symlink, or download your corpus into the workspace so the evaluator has something meaningful to run against.\n"
        f"3. **Build a real evaluator.** The placeholder eval script does not measure anything useful. Replace it with one that produces honest signals for your domain. The eval must write `{charter.workspace_layout.artifacts_dir}/eval_summary.json` with shape: `{{\"candidate_id\": \"...\", \"signals\": {{\"signal_name\": numeric_value}}}}`.\n"
        f"4. **Define real signals.** Edit the charter's `evaluation_plan.signals` to reflect what actually matters: quality metrics, cost, speed, whatever your objective demands. Signals can be hard-gated, regression-limited, or weighted.\n"
        f"5. **Build the pipeline.** Replace the placeholder pipeline files with real implementation. This is the mutable surface the experiment loop will iterate on.\n"
        f"6. **Run the first honest baseline.** `auto-anything baseline --task-root .` — this locks in a real starting point.\n"
        f"7. **Then iterate.** Make changes, run `auto-anything iterate --hypothesis '...' --change-summary '...'`, check accepted/rejected, repeat.\n\n"
        f"Do not trust any metrics until steps 1-5 are done. The experiment loop is only useful when the evaluator and signals are honest.\n\n"
        f"## Start Here\n"
        f"- read `task_charter.json` to understand the full contract\n"
        f"- inspect the mutable surface and decide what the real pipeline needs to look like\n"
        f"- inspect the data assets and get real corpus data into the workspace\n"
        f"- choose deterministic or AI-backed approaches based on the task, not by default habit\n"
        f"- use live provider and benchmark APIs instead of stale prior model knowledge when model choice matters\n\n"
        f"## Mutable Surface\n"
        f"{_render_list(mutable_paths)}\n\n"
        f"## Optimizable Artifacts\n"
        f"{artifact_block}\n\n"
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
        f"## Notes\n"
        f"{_render_list(notes)}\n\n"
        f"## Artifacts To Check After Each Run\n"
        f"{_render_list(artifact_paths)}\n\n"
        f"## Live Model Selection\n"
        f"- only add AI or multi-agent structure when the task actually benefits from it; deterministic code is fine when it is simpler, cheaper, and good enough\n"
        f"- start with `auto_anything.recommend_openrouter_models_for_task(...)` to combine Artificial Analysis benchmark/speed data with OpenRouter availability, modality support, and provider pricing\n"
        f"- bias toward current releases when the benchmark and cost tradeoff is close, but do not trust recency alone\n"
        f"- use `auto_anything.list_artificial_analysis_llms()` to fetch live benchmark, speed, and price-per-1M-token data from `https://artificialanalysis.ai/api/v2/data/llms/models`\n"
        f"- use `auto_anything.shortlist_artificial_analysis_llms()` to shortlist candidates by benchmark score, blended price, and output speed before locking a model family\n"
        f"- use `auto_anything.list_openrouter_models()` to fetch live OpenRouter availability, modalities, supported parameters, and pricing from `https://openrouter.ai/api/v1/models`\n"
        f"- use `auto_anything.list_openrouter_models(available_only=True)` when you want the authenticated account-aware catalog\n"
        f"- for exact OpenRouter run cost, prefer `auto_anything.extract_openrouter_usage(response_json)` and `auto_anything.fetch_openrouter_generation(generation_id)` over hand-rolled token estimates\n"
        f"- before locking the default model, run a tiny empirical probe on 1-3 real task examples across the top 2-4 candidates and keep the winner that best balances quality, speed, and cost\n"
        f"- OpenRouter docs: `https://openrouter.ai/docs/api-reference/overview`\n"
        f"- Artificial Analysis docs: `https://artificialanalysis.ai/api`\n"
        f"- required env vars when used: `OPENROUTER_API_KEY`, `ARTIFICIAL_ANALYSIS_API_KEY`\n\n"
        f"## Workflow\n"
        f"- make a focused change\n"
        f"- run the main eval command\n"
        f"- inspect metrics, critic findings, and diff against prior accepted work\n"
        f"- prefer modular changes over monolithic rewrites\n"
        f"- record the next authoritative iteration with `{iteration_command}`\n"
    )


__all__ = ["render_task_agents_md"]
