# Scope artifact role

`conditions.json` is retained as a historical compatibility artifact and is
used by `training/train.py` through the final trainer helpers for condition and
task-scope lookup. In particular, the final `PS_JOINT_LMV` recipe resolves its
scope from this file.

The file is intentionally not deleted or rewritten. Some of its embedded
Phase 1 metadata is historical and is not the authority for the final recipe.
The final learning rates, GPU/world-size settings, schedules, and checkpoint
policy are defined by `training/train.py` and the final configuration files
under `training/configs/`.
