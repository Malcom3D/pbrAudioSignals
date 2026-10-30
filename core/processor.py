# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .groups import GENERIC, KNOWN_GROUPS
from typing import Any, Callable, Dict, Optional, Iterable, List, Set

_REGISTRY: Dict[str, type] = {}
_REGISTRY_GROUPS: Dict[str, Set[str]] = {}

class BaseProcessor:
    """
    Base class for all DAG-compatible signal processors.

    Contract
    --------
    * Input `x` is always a C-contiguous float32 array of shape (C, S).
    * Output must be a newly allocated float32 array of shape (C, S).
    * Processors must not mutate `x` in place.
    * Parameters are taken from `self.params` (construction time) with
      optional per-evaluation overrides read from `props.custom`.

    Track groups
    ------------
    ``track_group``       : the *primary* group of this processor.
    ``also_valid_for``    : optional set of additional groups this
                            processor is safe to run on (e.g. a normaliser
                            can be ``generic`` but also explicitly allowed
                            in ``audio-force``).
    ``accepts_any_group`` : skip matching entirely (rare; use sparingly).
    """

    name: str = "base_processor"
    track_group: str = GENERIC
    also_valid_for: Set[str] = set()
    accepts_any_group: bool = False
    required_side: Set[str] = set()

    def __init__(self, **params: Any):
        self.params = dict(params)

    # ---- sidechannel helpers --------------------------------------------------

    def check_sides(self, props: Any) -> Optional[str]:
        """Return the name of a missing required side channel, or None."""
        if not self.required_side:
            return None
        custom = getattr(props, "custom", {}) if props is not None else {}
        for key in self.required_side:
            if key not in custom:
                return key
        return None

    # ---- group helpers --------------------------------------------------

    @classmethod
    def supports_group(cls, group: str) -> bool:
        return (
            cls.accepts_any_group
            or group == cls.track_group
            or group in cls.also_valid_for
        )

    # ---- public entry point -------------------------------------------

    def process(self, x: np.ndarray, props: Any) -> np.ndarray:
        if x is None:
            return x
        x = _as_float32_2d(x)
        if x.shape[1] == 0:
            return x

        out = self._process(x, props)
        if out is None:
            return x
        if out is x:
            out = out.copy()
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32, copy=False)

    # ---- override point -----------------------------------------------

    def _process(self, x: np.ndarray, props: Any) -> np.ndarray:
        return x

    # ---- parameter resolution -----------------------------------------

    def p(self, props: Any, key: str, default: Any = None, cast: Optional[Callable[[Any], Any]] = None) -> Any:
        """
        Resolve parameter `key` with (in order of priority):
          1. self.params[key] (set at construction time)
          2. props.custom["params"][key] (per-evaluation override)
          3. props.custom[key]
          4. default
        """
        if key in self.params:
            v = self.params[key]
        elif props is not None and isinstance(getattr(props, "custom", None), dict):
            custom = props.custom
            sub = custom.get("params", {})
            if isinstance(sub, dict) and key in sub:
                v = sub[key]
            elif key in custom:
                v = custom[key]
            else:
                v = default
        else:
            v = default
        return cast(v) if (cast is not None and v is not None) else v

    # ---- side-channel helpers ----------------------------------------

    def side(self, props: Any, key: str, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
        """Return a side signal stored in props.custom[key], padded/truncated."""
        if props is None:
            return None
        custom = getattr(props, "custom", None)
        if not isinstance(custom, dict):
            return None
        sig = custom.get(key)
        if sig is None:
            return None
        sig = np.ascontiguousarray(sig, dtype=np.float32).ravel()
        if expected_len is None or sig.shape[0] == expected_len:
            return sig
        if sig.shape[0] > expected_len:
            return sig[:expected_len]
        padded = np.zeros(expected_len, dtype=np.float32)
        padded[: sig.shape[0]] = sig
        return padded


# ---- module-level helpers ---------------------------------------------

def _as_float32_2d(x: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def register_processor(name: str, track_group: Optional[str] = None,
                       also_valid_for: Optional[Iterable[str]] = None):
    """
    Class decorator: register a processor.

    Parameters
    ----------
    name                 : lookup name used in chain JSON.
    track_group          : primary group. Defaults to the class attribute
                           (which defaults to ``generic``).
    also_valid_for       : extra groups this processor may run on.
                           Merged with the class attribute.
    """
    def _wrap(cls):
        cls.name = name

        group = track_group if track_group is not None else \
                getattr(cls, "track_group", GENERIC)
        cls.track_group = group

        extra = set(getattr(cls, "also_valid_for", set()))
        if also_valid_for:
            extra |= set(also_valid_for)
        cls.also_valid_for = extra

        _REGISTRY[name] = cls
        _REGISTRY_GROUPS[name] = {group} | extra
        return cls
    return _wrap

def get_processor(name: str, track_group: Optional[str] = None) -> type:
    """
    Look up a processor class.

    Raises
    ------
    KeyError : unknown name.
    ValueError : ``track_group`` was supplied and the processor does
                 not support it.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown processor '{name}'. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]

    if track_group is not None and not cls.supports_group(track_group):
        raise ValueError(
            f"Processor '{name}' (group={cls.track_group}) "
            f"is not valid for track_group='{track_group}'. "
            f"Also valid for: {sorted(cls.also_valid_for)}"
        )
    return cls

def list_processors(track_group: Optional[str] = None) -> List[str]:
    """
    List registered processor names, optionally filtered by group.

    When ``track_group`` is provided, returns processors whose
    ``supports_group(track_group)`` is True (including cross-group
    allowances).
    """
    if track_group is None:
        return sorted(_REGISTRY)
    return sorted(
        name for name, cls in _REGISTRY.items()
        if cls.supports_group(track_group)
    )

def processor_groups(name: str) -> Set[str]:
    """Return the set of groups a processor is registered for."""
    return set(_REGISTRY_GROUPS.get(name, set()))


#def register_processor(name: str):
#    """Class decorator: register processor under `name` for DAG (de)serialisation."""
#    def _wrap(cls):
#        cls.name = name
#        _REGISTRY[name] = cls
#        return cls
#    return _wrap
#
#
#def get_processor(name: str) -> type:
#    if name not in _REGISTRY:
#        raise KeyError(f"Unknown processor '{name}'. Registered: {sorted(_REGISTRY)}")
#    return _REGISTRY[name]
#
