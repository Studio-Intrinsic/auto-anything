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
        f"4. **Define real signals.** Edit the charter's `evaluation_plan.signals` to reflect what actually matters: quality metrics, cost, speed, whatever your objective demands. Signals can be hard-gated, regression-limited, or weighted. See **How The Engine Works** below for critical guidance on setting weights correctly.\n"
        f"5. **Choose models using live data (if your pipeline uses AI).** Do not hardcode a model from memory. Your knowledge of model capabilities, pricing, and availability is stale. Instead:\n"
        f"   - Run `auto_anything.recommend_openrouter_models_for_task(task_description, modalities=[...])` to get ranked candidates combining benchmark scores, speed, cost, and availability.\n"
        f"   - Or manually: `auto_anything.shortlist_artificial_analysis_llms()` for benchmark/speed data, then `auto_anything.list_openrouter_models()` for pricing and availability.\n"
        f"   - Check modality support (vision, structured output, tool use) against what your pipeline actually needs.\n"
        f"   - Run `auto_anything.probe_candidates(...)` on 2-3 real examples across your top candidates before committing.\n"
        f"   - Track model cost as a signal in your evaluator using `auto_anything.extract_openrouter_usage(response)` for exact per-call cost.\n"
        f"   - The right model for classification may differ from the right model for judging. Choose per-call-site, not globally.\n"
        f"   - Revisit model selection when results plateau — a better model may unblock quality more than code changes.\n"
        f"6. **Build the pipeline.** Replace the placeholder pipeline files with real implementation. This is the mutable surface the experiment loop will iterate on.\n"
        f"7. **Run the first honest baseline.** `auto-anything baseline --task-root .` — this locks in a real starting point.\n"
        f"8. **Then iterate.** Make changes, run `auto-anything iterate --hypothesis '...' --change-summary '...'`, check accepted/rejected, repeat.\n\n"
        f"Do not trust any metrics until steps 1-6 are done. The experiment loop is only useful when the evaluator, signals, and model choices are honest.\n\n"
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
        f"## How The Engine Works\n"
        f"The acceptance engine computes a single utility score to decide accept/reject:\n\n"
        f"```\n"
        f"utility = sum(signal.weight * (candidate_value - baseline_value)) for each signal\n"
        f"```\n\n"
        f"The candidate is accepted when utility > 0 and no hard gates or regression limits are violated.\n\n"
        f"**Critical: signal scale matters.** The engine uses raw deltas. A signal ranging 0-1 (like a rate) produces deltas of ~0.01. A signal ranging 0-500 (like line count) produces deltas of ~50. If you give both weight=1.0, the large-scale signal will dominate every decision regardless of the small-scale signal.\n\n"
        f"To fix this, you have two options:\n"
        f"- **Normalize your signal values in the evaluator** so all signals are on comparable scales (e.g. 0-1). For example, emit `pipeline_lines / 500` instead of raw line count.\n"
        f"- **Set weights to compensate for scale.** If signal A ranges 0-1 and signal B ranges 0-500, give B a weight ~500x smaller than A to make their contributions comparable.\n\n"
        f"Example: if `compliance_rate` (0-1, weight 1.0) and `pipeline_lines` (0-500, weight 0.1), a 50-line increase creates utility -5.0 which drowns out a 0.01 compliance gain (+0.01). Fix: either emit lines as a fraction (`lines/500`), or set weight to 0.002.\n\n"
        f"The engine also checks:\n"
        f"- `hard_gate`: candidate must meet `target_value` or it is blocked unconditionally\n"
        f"- `max_regression`: candidate must not regress more than this amount from baseline (raw delta)\n"
        f"- counterbalance findings above `block_on_severity` block acceptance\n"
        f"- counterbalance findings below threshold apply a small penalty (`penalty_per_finding`)\n\n"
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
        f"## Model Selection API Reference\n"
        f"See step 5 above for when and how to use these. Do not hardcode models from memory.\n\n"
        f"- `auto_anything.recommend_openrouter_models_for_task(task, modalities=[...])` — ranked candidates combining benchmarks, speed, cost, availability\n"
        f"- `auto_anything.shortlist_artificial_analysis_llms()` — benchmark/speed/price shortlist\n"
        f"- `auto_anything.list_openrouter_models()` — full catalog with pricing and modality support\n"
        f"- `auto_anything.probe_candidates(candidates, examples, runner, scorer)` — empirical head-to-head on real examples\n"
        f"- `auto_anything.extract_openrouter_usage(response)` — exact per-call cost from OpenRouter response\n"
        f"- `auto_anything.fetch_openrouter_generation(generation_id)` — cost lookup by generation ID\n"
        f"- Env vars: `OPENROUTER_API_KEY`, `ARTIFICIAL_ANALYSIS_API_KEY`\n"
        f"- Docs: OpenRouter `https://openrouter.ai/docs/api-reference/overview`, Artificial Analysis `https://artificialanalysis.ai/api`\n\n"
        f"## Workflow\n"
        f"1. **Diagnose first.** Run `auto-anything diagnose --task-root .` to see failure modes, doc tiers, and open hypotheses. Do not iterate without understanding what's failing and why.\n"
        f"2. **Form a hypothesis.** Run `auto-anything hypothesize --task-root . --hypothesis '...' --targets '...'` to record what you believe and what you're targeting. This builds the hypothesis ledger.\n"
        f"3. **Make a focused change** that tests your hypothesis.\n"
        f"4. **Iterate.** Run `{iteration_command}` to evaluate, compare against baseline, and accept/reject.\n"
        f"5. **Review the signal breakdown** to understand what drove the decision.\n"
        f"6. **Diagnose again** to update failure modes and close/open hypotheses. Then repeat from step 2.\n\n"
        f"The cycle is: diagnose → hypothesize → change → iterate → review → diagnose. Not: change → iterate → change → iterate.\n"
    )


__all__ = ["render_task_agents_md"]
