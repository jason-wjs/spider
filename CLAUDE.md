# Repository Guide

SPIDER is a Python 3.12+ framework for physics-informed motion retargeting across
dexterous-hand and humanoid workflows.

## Sources of Truth

- `README.md` and `docs/`: installation, supported workflows, and user guides.
- `pyproject.toml`: dependencies, Python version, Ruff rules, and entry points.
- `docs/tasks/g1_wbc_mjx_full_rollout/criteria.md`: current G1 WBC MJX
  acceptance contract.
- `docs/HISTORY.md`: historical decisions only; do not copy thresholds from it.
- The nearest task handoff and `AGENTS.md`: active task and machine constraints.

## Common Commands

```bash
uv sync --extra dev
uv run pytest <test-path>
uv run ruff check <changed-paths>
uv run ruff format --check <changed-paths>
```
Run the narrowest relevant test first, then expand verification in proportion to
the change. GPU simulator scripts are integration or diagnostic runs, not the
default unit-test entry point.

## Repository Map

- `examples/config/`: Hydra configuration and dataset-specific overrides.
- `examples/`: runnable workflow entry points.
- `spider/`: core configuration, preprocessing, simulators, optimizers, viewers,
  postprocessing, and reusable task modules.
- `scripts/`: benchmark, evaluation, and experiment runners.
- `tests/`: unit and integration tests, including task-specific suites.
- `docs/tasks/`: current experiment contracts, handoffs, and diagnostics.

## Working Rules

- Prefer `uv run` and repository configuration over machine-specific commands.
- Keep Hydra defaults in configuration files; avoid duplicating them in scripts
  or documentation.
- Do not commit generated artifacts, local output paths, caches, or checkpoints.
- Preserve unrelated worktree changes and keep edits scoped to the request.
- For formal G1 WBC MJX GPU runs, follow the current criteria and the GPU
  visibility guidance in `AGENTS.md`.
