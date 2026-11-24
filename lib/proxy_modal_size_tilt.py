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

# pbrAudioSignals/lib/proxy_modal_size_tilt.py
"""
Size-driven spectral pivot for ProxySynth modal displacement tracks.

In `ProxySynth`, the IR table is indexed by:
    (material_key, proxy_type, size_scale)

where `size_scale` is the object's bounding-box diagonal normalized
into [0, 1] over the size range for that (material, shape) group.

This processor reconstructs `size_scale` from `obj_config.size`
(if it was written by the exporter) and applies the same *tone* shift
implied by ProxyIRTable: smaller objects are brighter, larger ones
are darker.

This is the static counterpart of `proxy_modal_force_dynamic_eq`
(which handles the *dynamic* brightness of an actual impact). Both
compose: the size tilt sets the perceptual "weight" of the object
and the force-DEQ shades its impact.

Params
-----
size_min              : float, default 0.02   # metres
size_max              : float, default 2.0    # metres
tilt_db_range         : float, default 4.0    # +/- dB at neutral size
neutral_size          : float, default 0.2    # "reference" object
low_shelf_hz          : float, default 250.0
high_shelf_hz         : float, default 3000.0
sample_rate           : int,   default 48000
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT


@nb.njit(cache=True, nogil=True, fastmath=True)
def _one_pole_lp_1d(x: np.ndarray, fc_norm: np.float32) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    a = np.float32(1.0) - np.exp(
        np.float32(-2.0) * np.float32(np.pi) * fc_norm
    )
    if a < np.float32(1e-6):
        a = np.float32(1e-6)
    if a > np.float32(1.0):
        a = np.float32(1.0)
    y = np.float32(0.0)
    for i in range(n):
        y = y + a * (x[i] - y)
        out[i] = y
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _apply_shelves_2d(x: np.ndarray,
                      low_db: np.float32,
                      high_db: np.float32,
                      fc_low_norm: np.float32,
                      fc_high_norm: np.float32) -> np.ndarray:
    """Same 2-shelf topology used by every other family."""
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        low = _one_pole_lp_1d(x[ch], fc_low_norm)
        high_lp = _one_pole_lp_1d(x[ch], fc_high_norm)
        gl = np.float32(10.0) ** (low_db / np.float32(20.0))
        gh = np.float32(10.0) ** (high_db / np.float32(20.0))
        for i in range(S):
            mid = x[ch, i] - high_lp[i]
            high = high_lp[i] - low[i]
            out[ch, i] = low[i] * gl + mid + high * gh
    return out


@register_processor("proxy_modal_size_tilt", track_group=MODAL_DISPLACEMENT)
class ProxyModalSizeTilt(BaseProcessor):
    """
    Static size-driven shelving for ProxySynth-like content.

    Uses only `obj_config.size` from `props.custom`. If no object
    configuration is available, the processor returns its input
    unchanged (no-op).

    The output tilt is *static* (per-stream), which is exactly what
    the IR table would apply by picking a different slice of the size
    axis. Therefore this processor is idempotent and safe to place
    near the head of the chain.
    """

    required_side = {"force"}   # for optional obj_config lookup

    def __init__(self,
                 size_min: float = 0.02,
                 size_max: float = 2.0,
                 tilt_db_range: float = 4.0,
                 neutral_size: float = 0.2,
                 low_shelf_hz: float = 250.0,
                 high_shelf_hz: float = 3000.0,
                 sample_rate: int = 48000,
                 **kw):
        super().__init__(
            size_min=size_min,
            size_max=size_max,
            tilt_db_range=tilt_db_range,
            neutral_size=neutral_size,
            low_shelf_hz=low_shelf_hz,
            high_shelf_hz=high_shelf_hz,
            sample_rate=sample_rate,
            **kw,
        )

    # ------------------------------------------------------------------
    def _resolve_size(self, props):
        """Read `obj_config.size`; return None if unavailable."""
        if props is None or not isinstance(getattr(props, "custom", None), dict):
            return None
        cfg = props.custom.get("obj_config")
        if cfg is None:
            return None
        size = getattr(cfg, "size", None)
        try:
            size = float(size)
        except (TypeError, ValueError):
            return None
        if size <= 0.0:
            return None
        return size

    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        size = self._resolve_size(props)
        if size is None:
            return x

        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        size_min = float(self.p(props, "size_min", self.params["size_min"]))
        size_max = float(self.p(props, "size_max", self.params["size_max"]))
        neutral = float(self.p(props, "neutral_size", self.params["neutral_size"]))
        tilt_range = float(self.p(props, "tilt_db_range", self.params["tilt_db_range"]))
        low_hz = float(self.p(props, "low_shelf_hz", self.params["low_shelf_hz"]))
        high_hz = float(self.p(props, "high_shelf_hz", self.params["high_shelf_hz"]))

        # Sign of tilt: small objects -> positive high-shelf.
        # Use log(size) so 1cm, 10cm, 1m are perceptually equal steps.
        #   tilt_db = -tilt_range * log2(size / neutral) / log2(size_ratio)
        size_ratio = max(size_max / max(size_min, 1e-6), 1.0)
        log2_ratio = float(np.log2(size_ratio)) or 1.0
        tilt_db = -tilt_range * float(np.log2(size / neutral)) / log2_ratio
        # Clamp to the configured range.
        tilt_db = float(np.clip(tilt_db, -tilt_range, tilt_range))

        # Additive-inverse on the low side so DC stays neutral.
        low_db = np.float32(-0.25 * tilt_db)
        high_db = np.float32(tilt_db)

        nyq = 0.5 * sr
        fc_low_hz = float(np.clip(low_hz, 20.0, nyq * 0.9))
        fc_high_hz = float(np.clip(high_hz, 200.0, nyq * 0.9))

        return _apply_shelves_2d(
            x, low_db, high_db,
            np.float32(fc_low_hz / sr),
            np.float32(fc_high_hz / sr),
        )
