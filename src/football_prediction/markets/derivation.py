"""Small market-derivation utilities used by notebooks."""

from __future__ import annotations


def line_column(prefix, line):
    """Create a stable column name for a half-unit betting line."""
    return f"{prefix}_{str(line).replace('.', '_')}"

