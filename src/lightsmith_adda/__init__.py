"""ADDA backend for lightsmith (GPL-3.0-or-later).

Registers the backend ``"adda"`` (method ``"dda"``) through the
``lightsmith.backends`` entry-point group. ADDA itself is an external
executable: put ``adda`` on PATH or point ``LIGHTSMITH_ADDA`` at it.
"""

from .backend import ENV_VAR, AddaBackend, AddaOptions, find_executable

__version__ = "0.1.0"

__all__ = ["ENV_VAR", "AddaBackend", "AddaOptions", "find_executable"]
