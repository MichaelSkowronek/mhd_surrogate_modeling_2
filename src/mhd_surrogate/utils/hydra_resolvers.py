"""Custom OmegaConf resolvers used by the Hydra config tree (configs/).

Registered on import: an entry point imports this module before its
`@hydra.main` runs, so the resolvers exist when Hydra resolves the config
(including `hydra.run.dir`).
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def if_null(value: Any, default: Any) -> Any:
    """`value`, or `default` if it is null: `${if_null:${resume},outputs/...}`.
    (OmegaConf's own `oc.select` falls back only for a missing key, not for
    one set to null.)"""
    return default if value is None else value


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("if_null", if_null, replace=True)


register_resolvers()
