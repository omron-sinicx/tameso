# Copyright (c) 2026 OMRON SINIC X Corporation

"""MAT^3: MAsked Tactile Trajectory Transformer + vicinity retrieval."""

# In conda / pixi environments, PyPI torch loads the system libstdc++ first, which can be older than
# the one required by conda-forge's sqlite/ICU (used by trackio). Loading sqlite3 before torch makes
# the environment's libstdc++ win. Harmless elsewhere.
try:
    import sqlite3  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__version__ = "0.1.0"
