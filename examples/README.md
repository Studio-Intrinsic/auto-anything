# Examples

These examples are intentionally small.

They are meant to show the user contract for `auto-anything`, not to be a complete task runtime.

Current example:

- `invoice_extraction_quickstart.py`
  - compiles a plain-English invoice extraction objective into a typed charter
  - assumes the implementation agent will then write the pipeline inside the declared workspace
  - defaults to the bundled sample invoice in `examples/sample_data/`
- `bootstrap_invoice_task.py`
  - creates a starter task workspace in `work/invoice_extraction_demo/`
  - copies fixtures and goldens
  - writes a baseline pipeline and eval harness
  - runs the baseline eval so the agent can immediately start iterating
