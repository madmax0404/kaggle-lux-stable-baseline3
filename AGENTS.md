# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `modified_packages/`, shadowing upstream libraries. Extend the `_modified.py` modules and treat the originals as references. `GreedyLRScheduler/GreedyLRScheduler.py` hosts the custom scheduler. `Notebooks/Agent_Development` contains training notebooks, `Notebooks/EDA` covers analysis, and `images/` stores presentation assets. Keep heavy artifacts outside the repo and point to them from notebooks.

## Build, Test, and Development Commands
Set up Python 3.11+ via `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Export the local forks when running anything: `PYTHONPATH=modified_packages:$PYTHONPATH`. Use `jupyter lab Notebooks/Agent_Development` for experiments and `tensorboard --logdir runs` to review PPO metrics. For environment smoke checks run `python3 -m luxai_s3.env_modified --episodes 1` after exporting the `PYTHONPATH`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and a 100-character soft limit. Add type hints for public functions and note tensor shapes in docstrings. Modules stay snake_case, `_modified` suffixes mark local forks, classes use `CamelCase`, and constants stay uppercase. Keep notebooks light on side effects and move reusable code into modules.

## Testing Guidelines
Ship new code with `pytest` cases under a mirrored tree such as `tests/luxai_s3/test_wrappers.py`. Name tests `test_<behavior>`, assert numerical outcomes and tensor shapes, and target ≥80% coverage on touched files. Run `pytest --maxfail=1 --disable-warnings` before pushing, capturing random seeds or logs whenever failures depend on stochastic episodes.

## Commit & Pull Request Guidelines
Commits stay small, imperative, and scoped (`add`, `fix`, `refactor`), reflecting the existing history (`added english readme`). Mention the affected subsystem in the subject when possible. Pull requests must explain motivation, summarize environment or policy changes, and attach validation evidence (reward curves, win rates, or replay links). Link related Kaggle discussions or issues and flag notebooks or configs reviewers must rerun.

## Agent-Specific Notes
Log major hyper-parameter adjustments in the leading markdown cell of the relevant notebook and reference external checkpoints there. When altering environment logic, update paired wrappers and regenerate any cached observation statistics. Keep experimental scheduler variants inside `GreedyLRScheduler/` and describe expected usage in the change description.
