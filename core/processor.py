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

from typing import Any, Callable, Dict, Optional
import numpy as np

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
    """

    name: str = "base_processor"

    def __init__(self, **params: Any):
        self.params = dict(params)

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


_REGISTRY: Dict[str, type] = {}


def register_processor(name: str):
    """Class decorator: register processor under `name` for DAG (de)serialisation."""
    def _wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_processor(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown processor '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]

