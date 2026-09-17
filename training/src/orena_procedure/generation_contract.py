"""Shared generation defaults for new VLM inference experiments."""

from __future__ import annotations

# New active inference experiments use a cap large enough for the longest
# audited answer plus EOT. Historical experiments keep their explicitly
# registered caps for reproducibility.
DEFAULT_MAX_NEW_TOKENS = 128
