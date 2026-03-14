# auto-anything

`auto-anything` is a general library for turning high-level goals into agent-optimizable engineering loops.

The user should be able to show up with:

- task data
- a goal statement
- anti-goals and constraints
- optionally an explicit metric
- optionally a domain skill pack

and the system should turn that into an explicit optimization program:

- what can change
- what must not change
- how success is measured
- what counts as regression
- how candidates are accepted or rejected

The expected output is not limited to a trained model. It can be a Python pipeline, a remediation runtime, an evaluator-backed system, or any bounded implementation surface that can be improved experimentally.

The repo is organized in two layers:

- `src/auto_anything/` contains the generic library: compiler, engine, experiment memory, scaffold materialization, and open-ended bootstrap primitives.
- `examples/` contains small worked examples that exercise the generic library on concrete tasks.

## Quick Start

1. Create a local env file:

```bash
cp .env.example .env.local
```

2. Edit `.env.local` and add your provider and benchmark API keys:

```bash
OPENROUTER_API_KEY=...
ARTIFICIAL_ANALYSIS_API_KEY=...
```

`OPENROUTER_API_KEY` is used when the task actually needs model calls or live OpenRouter model discovery. `ARTIFICIAL_ANALYSIS_API_KEY` is used for live benchmark, speed, and pricing lookups from Artificial Analysis so the agent can make current model-selection decisions instead of relying on stale priors.

3. Bootstrap a starter workspace from plain-English intent plus referenced files:

```bash
python3 examples/bootstrap_from_request.py \
  --objective "Extract invoice fields accurately while staying scalable, fast, and cheap."
```

If you do not pass `--path`, the script defaults to the bundled sample invoice at [examples/sample_data/sample_invoice.pdf](/Users/gmiller/code/gregm711/studio-instrinsic-env/open-source/auto-anything-env/auto-anything/examples/sample_data/sample_invoice.pdf). You can also point it at your own files:

```bash
python3 examples/bootstrap_from_request.py \
  --objective "Extract invoice fields accurately while staying scalable, fast, and cheap." \
  --path /absolute/path/to/invoices
```

This creates a self-contained task workspace under `work/<derived-task-name>/`, writes a generic starter pipeline and eval harness, writes a workspace `AGENTS.md`, and runs an initial baseline. For unknown tasks, that baseline is only a placeholder until the agent replaces the evaluator with a real one.

4. Point me at that workspace and tell me to iterate:

```text
Use auto-anything. I need a pipeline with these PDFs to have data extracted. Get to work in the workspace you just bootstrapped.
```

The important property is that the front door is now plain text plus file references. The user should not need to hand-assemble a charter before the agent can start.

The bootstrapped `AGENTS.md` is the agent handoff document inside the task workspace. It is generated from the charter and is meant to keep the implementation loop grounded in:

- the actual mutable surface
- the protected paths
- the run commands
- the subsystem map
- the builder/critic/judge loop
- the artifacts and history to inspect after each run

It also now includes a live model-selection section so the agent can query current provider and benchmark APIs instead of relying on stale model knowledge:

- OpenRouter model catalog and pricing: `https://openrouter.ai/docs/api-reference/overview`
- Artificial Analysis free benchmark API: `https://artificialanalysis.ai/api`

The corresponding library helpers are:

- `list_openrouter_models(...)`
- `get_openrouter_model(...)`
- `extract_openrouter_usage(...)`
- `fetch_openrouter_generation(...)`
- `list_artificial_analysis_llms(...)`
- `shortlist_artificial_analysis_llms(...)`

The intent is not “always use AI.” Many tasks should stay deterministic. The point is to give the agent reliable, current information for model selection, pricing comparison, and exact cost accounting when AI or multi-agent structure is actually warranted.

### Live Model Research

When model choice matters, the intended workflow is:

1. Use `list_artificial_analysis_llms(...)` to pull current quality, coding, math, speed, and price-per-1M-token data from Artificial Analysis.
2. Use `shortlist_artificial_analysis_llms(...)` to narrow the field by your actual constraints.
3. Use `list_openrouter_models(...)` or `get_openrouter_model(...)` to confirm the shortlisted models are actually available through OpenRouter and support the modalities/parameters you need.
4. During real eval runs, use `extract_openrouter_usage(...)` or `fetch_openrouter_generation(...)` to capture exact OpenRouter cost instead of guessing from token counts.

Artificial Analysis attribution is required when using their free API. See their docs at `https://artificialanalysis.ai/api`.

## Motivation

We want the autonomous experiment loop from projects like `autoresearch`, but generalized into something that is useful outside a single research setup.

Inspiration: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

`autoresearch` demonstrates an important pattern: keep the evaluator fixed, keep the mutable surface small, let an agent iterate, and accept only measurable improvements. That repo applies the pattern to a narrow training setup. `auto-anything` is our attempt to lift the pattern into a reusable library for open-ended but still evaluable tasks.

Our motivation is broader:

- use the same core loop for LLM training, RL tasks, invoice extraction, document remediation, and future domains
- let users start from plain-English objectives, not only hand-built reward functions
- package domain knowledge as reusable skill packs instead of hidden prompt folklore
- keep the optimization loop generic while keeping each task explicit and auditable

The design rule is:

- generic in the loop
- concrete in the charter

## What This Library Is

`auto-anything` is the abstract system layer behind task-specific autonomous optimization.

It is built around explicit boundaries:

- immutable evaluation and acceptance logic
- mutable candidate implementation surface
- typed objective charter
- auditable skill contributions
- replayable experiment records

The user-facing flow is:

1. Bring data.
2. Describe the goal in plain English, or provide explicit metrics.
3. Add optional anti-goals and constraints.
4. Add optional skill packs.
5. Compile the request into a typed task charter.
6. Let an agent iterate on the allowed surface.
7. Run a built-in critical pass from the same CLI agent.
8. Keep only candidates that improve the charter without violating hard gates.

Concrete benchmarks and domains should usually live outside the core package as examples or adapters. The core library should supply the loop, memory, scaffold, and bootstrap machinery needed to let the calling agent construct whatever task world is required.

## Agent-First Workflow

This library is being designed for direct in-repo use by an implementation agent such as Codex.

The expected workflow is not:

- send the task to a remote optimizer
- wait for a hidden system to solve it elsewhere

The expected workflow is:

1. The agent reads the repo and objective.
2. The compiler turns the request into a `TaskCharter`.
3. The bootstrap flow materializes a local workspace.
4. The agent writes the candidate Python pipeline directly in that workspace.
5. The agent runs the declared commands locally.
6. The adapter collects an `EvaluationReport`.
7. The same agent shifts into a critic role and tries to break, question, or game-check the candidate.
8. The engine decides keep or discard using both the primary report and the counterbalance report.
9. The agent iterates again.

Because of that, the library should optimize for:

- explicit mutable paths
- explicit protected paths
- explicit subsystem ownership
- explicit run commands
- explicit workspace layout
- inspectable artifacts and replay slices

Nothing important should be hidden in agent prompt state if it can be captured in typed workspace or charter data.

## Plain-Text Front Door

The intended usage model is:

```text
Use auto-anything. Optimize a pipeline for these PDFs. Keep it accurate, fast, and cheap.
```

plus referenced paths.

The generic request path for that is now:

- `PlainTextTaskRequest`
- `build_brief_from_request(...)`
- `build_bootstrap_plan_from_request(...)`
- `bootstrap_task_from_request(...)`
- `run_task_baseline(...)`
- `run_task_iteration(...)`

That means the library can accept loose user intent, synthesize a real workspace, run an initial baseline, and hand off a concrete task root for iterative improvement.

The important design choice is that bootstrap does not need to already know the task. It only needs to create a useful world with:

- a task charter
- a mutable pipeline surface
- a runnable eval command
- an `AGENTS.md` handoff
- experiment memory and history

After that, the calling agent is expected to finish shaping the pipeline and evaluator for the actual task.

## Execution Boundary

The optimizer and the candidate should not be the same process boundary.

`auto-anything` now has a pluggable execution backend layer:

- `direct_subprocess`
  - runs declared commands directly in the task workspace
- `isolated_workspace`
  - copies the workspace to a temporary location
  - runs the command there
  - syncs configured paths such as `artifacts/` back into the live workspace

This is the first step toward Agentica-style execution discipline without importing a full server/runtime stack. The orchestrator stays simple and local, while candidate and eval commands can run behind a clearer host/guest boundary.

## Subsystem-Scoped Iteration

Whole-pipeline optimization is too blunt for real work. The agent needs to be able to focus on one subsystem, improve it quickly, and still keep whole-system guardrails in view.

`auto-anything` now treats subsystems as typed charter data:

- each subsystem owns specific paths
- each subsystem names its primary signals
- each subsystem names guardrail signals that should stay visible during focused work

That makes loops like this possible:

1. focus on `field-extraction`
2. edit only the owned paths for that subsystem
3. rerun the full eval harness
4. let the critic attack both extraction quality and architecture quality
5. accept only if the subsystem improved without breaking global guardrails

This also gives the critic a second job beyond metric gaming. It should actively flag codebase failure modes such as:

- giant monolithic files
- hidden coupling between stages
- duplicated logic across subsystems
- shortcuts that improve one slice while making future iteration harder

The goal is targeted iteration without monolithic sprawl.

## Experiment Lineage

Experiments need memory. The agent should be able to inspect previous attempts, diff them, revive useful ideas, and re-architect from concrete prior states instead of working from a vague summary.

`auto-anything` now treats experiment lineage as a first-class artifact:

- each eval run appends to an experiment ledger
- each eval run snapshots the workspace into local git
- each experiment gets a stable git tag such as `aa-exp-0007`
- each experiment gets its own JSON and Markdown report
- each workspace maintains a rolling `knowledge_base.md`
- each workspace writes a progress curve SVG from the accumulated history

That means the agent can use normal tools like:

- `git log --oneline --decorate`
- `git show aa-exp-0003`
- `git diff aa-exp-0004..aa-exp-0009`

and combine that with:

- `artifacts/experiment_history.json`
- `artifacts/knowledge_base.md`
- `artifacts/experiments/aa-exp-0007.md`
- `artifacts/progress_curve.svg`

to understand what changed, what improved, and where a promising branch of ideas was abandoned too early.

## Built-In Counterbalance

This system should assume that the optimizer can game the metric unless something pushes back.

But the pushback should not require a second paid model call or a separate hidden agent service.

The intended pattern is:

- the same CLI agent first works as the builder
- then the same CLI agent shifts into a critic role
- the critic tries to find shortcuts, blind spots, missing cases, and metric gaming
- the engine uses those findings to block or penalize acceptance

This is closer to self-play or an internal GAN-style discipline than to a second external evaluator.

The important property is not "multiple models."
The important property is "multiple adversarial stances inside one loop."

## Core Concepts

### `ObjectiveBrief`

The raw request.

This captures:

- objective statement
- anti-goals
- constraints
- data assets
- optional explicit signals
- optional mutable/protected path hints
- model/tool allowlists

### `SkillPack`

A reusable domain knowledge pack.

A skill contributes:

- suggested evaluation signals
- suggested constraints
- decomposition hints
- suggested subsystem ownership
- safe mutable surface defaults

Examples:

- invoice extraction
- RL environment invention
- code generation pipeline optimization

### `TaskCharter`

The compiled and explicit contract for optimization.

This is the central object the rest of the system should optimize against. It turns vague goals into inspectable structure:

- hard constraints
- soft constraints
- evaluation plan
- search surface
- applied skills
- decomposition hints

### `ObjectiveSignal`

A measurable thing the engine can score.

Signals can be:

- scalar
- rubric-based
- binary

Signals may be:

- weighted
- hard-gated
- regression-limited

### `ExperimentEngine`

The acceptance layer.

It compares a baseline report against a candidate report, and may also consume a self-critic report, using:

- hard gates
- regression limits
- weighted utility gain
- counterbalance findings

### `TaskAdapter`

The bridge to a real domain.

Each concrete problem should define a task adapter that knows how to:

- set up the task
- materialize a local candidate workspace
- identify the baseline candidate
- run a candidate
- collect evaluation reports

### `WorkspaceLayout` and `RunCommand`

These exist because the agent is expected to work directly inside the repo.

The charter should be able to say:

- where the candidate code lives
- where evaluator code lives
- where artifacts and replay cases go
- which commands the agent is supposed to run

That gives the system an explicit operational surface instead of relying on tribal knowledge.

## Example: Invoice Extraction

The long-term user story looks like this:

1. User supplies a corpus of invoices and expected extraction outputs.
2. User says:
   - "Extract invoice fields accurately"
   - "Stay scalable"
   - "Use only these models"
   - "Do not explode token cost"
3. An invoice extraction skill pack contributes:
   - likely signals such as field accuracy, document pass rate, latency, token cost, schema validity
   - decomposition hints such as OCR cleanup, table line items, vendor normalization, confidence calibration
   - subsystem ownership such as document ingestion, field extraction, normalization, and schema validation
   - safe defaults for the mutable pipeline boundary
4. The compiler produces a `TaskCharter`.
5. An agent improves the bounded pipeline.
6. The same agent performs a critic pass and tries to find gaming behavior, brittle assumptions, hidden failure cases, or architecture sprawl.
7. The engine keeps only candidates that actually improve the charter.

The key point is that the user should not be forced to hand-author a reward function before the system can start.

The bundled sample invoice lives at [examples/sample_data/sample_invoice.pdf](/Users/gmiller/code/gregm711/studio-instrinsic-env/open-source/auto-anything-env/auto-anything/examples/sample_data/sample_invoice.pdf), so the quickstart works without any extra data setup.

The expected output for that sample lives at [examples/sample_data/sample_invoice.expected.json](/Users/gmiller/code/gregm711/studio-instrinsic-env/open-source/auto-anything-env/auto-anything/examples/sample_data/sample_invoice.expected.json), which is what the starter eval harness uses for its first baseline score.

You can also bootstrap with an initial subsystem focus:

```bash
python3 examples/bootstrap_invoice_task.py \
  --objective "Extract invoice fields accurately while staying scalable, fast, and cheap." \
  --focus-subsystem field-extraction
```

The starter workspace is intentionally modular so focused iteration has somewhere clean to land:

- `src/invoice_pipeline/document_io.py`
- `src/invoice_pipeline/field_extractors.py`
- `src/invoice_pipeline/normalization.py`
- `src/invoice_pipeline/schema.py`
- `src/invoice_pipeline/extract.py`

Every eval in that workspace also updates:

- `artifacts/eval_summary.json`
- `artifacts/experiment_history.json`
- `artifacts/knowledge_base.md`
- `artifacts/experiments/*.md`
- `artifacts/progress_curve.svg`

and snapshots the current workspace into local git so later experiments can build on explicit prior states.

To record a real iteration after you edit the pipeline, run:

```bash
python3 examples/run_task_iteration.py \
  --task-root work/invoice_extraction_demo \
  --hypothesis "Improving normalization should reduce date parsing brittleness." \
  --change-summary "Refined normalization logic for invoice dates." \
  --focus-subsystem normalization
```

That path uses the engine decision plus the self-critic pass to write an authoritative experiment record, instead of relying only on the raw eval summary.

## Design Principles

- Be generic in the optimization loop, not vague in task definition.
- Plain text must become a typed charter before optimization begins.
- Skills should be explicit reusable modules, not hidden prompt state.
- The optimizer should always face a built-in counterbalance.
- The mutable surface should stay narrow.
- Acceptance logic should stay outside the candidate surface.
- If the task cannot be evaluated, the system should say so instead of pretending.

## Package Layout

- `src/auto_anything/models.py`
  - typed task, signal, report, and experiment models
- `src/auto_anything/interfaces.py`
  - protocol boundaries for skills, adapters, compilers, and agents
- `src/auto_anything/compiler.py`
  - default objective compiler and skill contribution merge logic
- `src/auto_anything/engine.py`
  - baseline-vs-candidate acceptance logic and experiment bookkeeping
- `src/auto_anything/skills.py`
  - simple skill registry helpers
- `src/auto_anything/workspace.py`
  - workspace path resolution helpers for in-repo agent execution

## Current Scope

This repo currently contains the abstract foundation only.

It now includes:

- the abstract core library
- an invoice extraction skill
- a bootstrap path that scaffolds a starter invoice task workspace

It does not yet include:

- a fully general task-family inference layer
- a mature end-to-end autonomous optimization loop
- more than one practical task family

Those are the next layers to build on top of the current scaffold.
