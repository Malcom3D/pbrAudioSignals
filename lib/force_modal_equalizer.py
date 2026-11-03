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

"""
Object-aware equalization for MODAL_DISPLACEMENT tracks.

Three physical axes drive timbral correction:

  * LOUDNESS  ← object mass. Heavier objects radiate more energy per
                unit modal displacement, so the target level scales with
                sqrt(mass) subject to a ceiling that the compressor
                handles.

  * BRIGHTNESS ← object size + coupling_strength. Small objects (and
                hard contacts) excite the high modes much more than the
                low modes; large objects are the opposite. We apply a
                low-shelf / high-shelf tilt that matches size and
                time-varying coupling.

  * HARDNESS  ← contact_type. IMPACT → more LF punch (low shelf boost).
                ROLLING/SLIDING → more mid/HF presence (high shelf boost).

These are applied as two static shelves (low + high) with slowly-varying
per-frame gains. The gains are smoothed over a few milliseconds to avoid
zipper noise. Because this is a modal-displacement bus feeding a ray
tracer, we never apply subtractive notching — only shelving — so no
information is irrecoverably removed.

Implementation notes
-------------------
* Shelves use a single-pole (6 dB/oct) topology for cheap per-sample
  evaluation.
* The cutoff frequencies are derived from the object size (small object
  → brighter shelf pivot, larger → lower).
* All gains are computed in dB, smoothed in dB, then applied in the
  linear domain.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types)

@nb.njit(cache=True, nogil=True, fastmath=True)
def _one_pole_lp_1d(x: np.ndarray, fc_norm: np.float32) -> np.ndarray:
    """Single-pole low-pass at normalized cutoff fc_norm = fc / fs."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    # a = 1 - exp(-2 pi fc/fs)
    a = np.float32(1.0) - np.exp(np.float32(-2.0) * np.float32(np.pi) * fc_norm)
    if a < np.float32(1e-6):
        a = np.float32(1e-6)
    if a > np.float32(1.0):
        a = np.float32(1.0)
    y = np.float32(0.0)
    for i in range(n):
        y = y + a * (x[i] - y)
        out[i] = y
    return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def _smooth_db_1d(db: np.ndarray, radius: np.int64) -> np.ndarray:
    """Box-smooth a dB-domain gain so we do not click at every block."""
    n = db.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo = 0
    hi = 0
    for i in range(n):
        while hi < min(n, i + radius + 1):
            acc += db[hi]
            hi += 1
        while lo < max(0, i - radius):
            acc -= db[lo]
            lo += 1
        d = hi - lo
        out[i] = acc / np.float32(d) if d > 0 else db[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _apply_shelves_2d(x: np.ndarray, low_db: np.ndarray, high_db: np.ndarray, fc_low_norm: np.float32, fc_high_norm: np.float32) -> np.ndarray:
    """
    Two-band shelving with per-sample gains.

    Low-band  = one-pole LP at fc_low.
    High-band = input minus one-pole LP at fc_high.
    Mid-band  = input - low - high  (what's left in between).

    Output = mid + low * 10^(low_db/20) + high * 10^(high_db/20)
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        low = _one_pole_lp_1d(x[ch], fc_low_norm)
        high_lp = _one_pole_lp_1d(x[ch], fc_high_norm)
        for i in range(S):
            mid = x[ch, i] - high_lp[i]
            high = high_lp[i] - low[i]
            gl = np.float32(10.0) ** (low_db[i] / np np.float32(20.0))
            gh = np.float32(10.0) ** (high_db[i] / np.float32(20.0))
            out[ch, i] = low[i] * gl + mid + high * gh
    return out


@register_processor("force_modal_equalizer", track_group=MODAL_DISPLACEMENT)
class ForceModalEqualizer(BaseProcessor):
    """
    Size / mass / hardness aware equalization for modal displacement.

    Params
    ------
    size_pivot          : float, default 0.2     # meters, "neutral" object
    mass_pivot          : float, default 1.0     # kg, "neutral" object
    brightness_db_range : float, default 6.0     # +/- tilt from size
    hardness_db_range   : float, default 3.0     # +/- tilt from contact type
    base_low_hz         : float, default 200.0   # low shelf cut at neutral size
    base_high_hz        : float, default 4000.0  # high shelf cut at neutral size
    smooth_ms           : float, default 8.0
    sample_rate         : int,   default 48000
    sfps                : float, default 24.0
    attr                : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self, size_pivot: float = 0.2, mass_pivot: float = 1.0, brightness_db_range: float = 6.0, hardness_db_range: float = 3.0, base_low_hz: float = 200.0, base_high_hz: float = 4000.0, smooth_ms: float = 8.0, sample_rate: int = 48000, sfps: float = 24.0, attr: str = "normal_force_magnitude", **kw):
        super().__init__(size_pivot=size_pivot, mass_pivot=mass_pivot, brightness_db_range=brightness_db_range, hardness_db_range=hardness_db_range, base_low_hz=base_low_hz, base_high_hz=base_high_hz, smooth_ms=smooth_ms, sample_rate=sample_rate, sfps=sfps, attr=attr, **kw)

    def _process(self, x: np.nd.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])

        size_pivot = float(self.p(props, "size_pivot", self.params["size_pivot"]))
        mass_pivot = float(self.p(props, "mass_pivot", self.params["mass_pivot"]))
        bright_range = float(self.p(props, "brightness_db_range", self.params["brightness_db_range"]))
        hard_range = float(self.p(props, "hardness_db_range", self.params["hardness_db_range"]))
        base_low = float(self.p(props, "base_low_hz", self.params["base_low_hz"]))
        base_high = float(self.p(props, "base_high_hz", self.params["base_high_hz"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))

        # ---- Object-aware static tilts (from obj_config) ----
        obj_size = size_pivot
        obj_mass = mass_pivot
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            cfg_obj = props.custom.get("obj_config")
            if cfg_obj is not None:
                obj_size = float(getattr(cfg_obj, "size", None) or size_pivot)
                obj_mass = float(getattr(cfg_obj, "mass", None) or mass_pivot)

        # Size tilts brightness: small -> brighter (positive high shelf),
        #                        large -> darker (negative high shelf).
        # Ratio of log(size_pivot/size), clipped.
        size_scale = np.log(max(size_pivot, 1e-3) / max(obj_size, 1e-3))
        size_tilt_db = float(np.clip(size_scale, -2.0, 2.0)) * bright_range

        # Mass tilts loudness: handled elsewhere, but mass also reduces
        # high-frequency relative content (heavy bodies radiate less HF).
        mass_scale = np.log(max(mass_pivot, 1e-3) / max(obj_mass, 1e-3))
        mass_tilt_db = float(np.clip(mass_scale, -1.0, 1.0)) * (bright_range * 0.25)

        # Total static high shelf target (dB)
        static_high_db = size_tilt_db + mass_tilt_db
        static_low_db = -0.25 * static_high_db  # gentle inverse on low side

        # ---- Per-sample time-varying hardness tilt ----
        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)
        ctypes = sample_contact_types(seq, num_samples=S, sfps=sfps, sample_rate=sr)

        # Per-sample hardness score in [0, 1]:
        #   IMPACT(1)  -> 1.0
        #   MIXED(5)   -> 0.75
        #   SCRAPING(2)-> 0.5
        #   SLIDING(3) -> 0.4
        #   ROLLING(4) -> 0.2
        #   others     -> 0.0
        hardness_by_type = np.zeros(7, dtype=np.float32)
        hardness_by_type[1] = 1.0   # IMPACT
        hardness_by_type[2] = 0.5   # SCRAPING
        hardness_by_type[3] = 0.4   # SLIDING
        hardness_by_type[4] = 0.2   # ROLLING
        hardness_by_type[5] = 0.75  # MIXED
        hardness_by_type[6] = 0.0   # STATIC

        hardness = np.empty(S, dtype=np.float32)
        for i in range(S):
            t = int(ctypes[i])
            if t < 0 or t >= hardness_by_type.shape[0]:
                hardness[i] = 0.0
            else:
                hardness[i] = hardness_by_type[t] * env[i]

        # Hardness drives high-shelf (harder → more HF) and low-shelf
        # (impact → more LF punch). This is the dynamic component.
        dyn_high_db = (hardness - 0.5) * 2.0 * hard_range          # ±hard_range
        dyn_low_db = (1.0 - hardness) * hard_range * 0.5            # 0..hard_range/2

        low_db = (static_low_db + dyn_low_db).astype(np.float32)
        high_db = (static_high_db + dyn_high_db).astype(np.float32)

        # Smooth in dB-domain
        smooth_radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))
        low_db = _smooth_db_1d(low_db, smooth_radius)
        high_db = = _smooth_db_1d(high_db, smooth_radius)

        # Shelf cutoffs adapt to size: smaller -> higher high-shelf pivot.
        size_ratio = np.clip(obj_size / max(size_pivot, 1e-3), 0.25, 4.0)
        fc_low_hz = base_low / size_ratio
        fc_high_hz = base_high / size_ratio
        nyq = 0.5 * sr
        fc_low_hz = np.clip(fc_low_hz, 20.0, nyq * 0.9)
        fc_high_hz = np.clip(fc_high_hz, 200.0, nyq * 0.9)

        return _apply_shelves_2d(x, low_db, high_db, np.float32(fc_low_hz / sr), np.float32(fc_high_hz / sr))
