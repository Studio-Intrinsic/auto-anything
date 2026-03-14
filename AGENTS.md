# AGENTS.md

This repo is for building the generic `auto-anything` library.

## Primary Intent

Design the system for direct use by an in-repo coding agent.

Assume the normal mode of operation is:

1. the user provides data and goals
2. the library compiles a typed charter
3. the agent writes a Python pipeline locally
4. the agent runs local commands
5. the same agent swaps into a critic role and tries to break the candidate
6. the library evaluates baseline vs candidate with that counterbalance in view
7. the agent iterates

Do not optimize first for:

- hidden orchestration
- remote execution assumptions
- implicit workspace conventions

Prefer:

- typed workspace layout
- explicit mutable and protected paths
- explicit run commands
- explicit artifacts and replay locations
- acceptance logic outside the mutable candidate surface
- explicit self-critic / counterbalance data rather than hidden evaluator magic

## Design Rule

Generic in the optimization loop.
Concrete in the task charter.

## Current Direction

The abstract library should stay small and sharp.

Good additions:

- core typed models
- compiler logic
- acceptance logic
- workspace helpers
- adapter protocols

Bad additions right now:

- domain-specific logic in the core package
- speculative orchestration layers
- hidden prompt-only behavior that could be expressed as typed data
- broad frameworks before the first practical task adapter exists

## Immediate Next Step

After the abstract layer is stable, keep pushing the cold-start bootstrap path:

- clone repo
- add `.env.local`
- point at PDFs
- supply plain-English objective
- let the agent work inside the generated workspace

Invoice extraction is the current proving ground for that workflow.
