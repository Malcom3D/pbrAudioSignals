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

# pbrAudioSignals/lib/proxy_modal_force_dynamic_eq.py
"""
Force-envelope-driven dynamic dynamic high-frequency boost for ProxySynth.

Direct port of `ellipsoidalProxy.ProxyEqualizer._process_with_dynamic_eq`
into the StorageEngine DAG. The original algorithm was:

    block_force = mean(force_env[block])
    if block_force > 0.1:
        dynamic_curve[f > 2000] *= 1 + 0.5 * block_force

The DAG native form is *per sample*, so we can avoid the FFT
block-boundary fades entirely. Since the shelf topology is already
provided by `_shelf_tilt_2d` (used by `force_modal_contact_tilt`),
we express the same intent as a smooth high-shelf ramp whose gain
is `20 * log10(1 + boost * force)`.

Notes
-----
* The DB-domain ramp is lowpassed to prevent zipper noise.
* On top of the shelf, we add a *very* subtle "air" component
  above 10 kHz (from the ProxyEqualizer `scraping_curve` logic)
  that is also force-gated.
* Unlike `force_modal_contact_tilt` (which uses contact-type tables),
  this processor is purely force-driven. It does not read
  `contact_type`. It's the natural *continuation* of an impact's
  brightness tail.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope


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
def _dynamic_bright_2d(x: np.ndarray,
                       gain_db: np.ndarray,
                       air_db: np.ndarray,
                       fc_low_norm: np.float32,
                       fc_high_norm: np.float32,
                       fc_air_norm: np.float32) -> np.ndarray:
    """
    3-shelf dynamic EQ:
        low-band   = unity (unchanged)
        mid-band   = input - low - high
        high-band  = input - air_lp (between fc_high and fc_air)
        air-band   = air_lp (above fc_air)

    mid-band is NOT scaled — it's the neutral reference.
    high-band and air-band get +gain_db[i] and +air_db[i] respectively.
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        low = _one_pole_lp_1d(x[ch], fc_low_norm)
        high_lp = _one_pole_lp_1d(x[ch], fc_high_norm)
        air_lp = _one_pole_lp_1d(x[ch], fc_air_norm)
        for i in range(S):
            mid = x[ch, i] - high_lp[i]
            high = high_lp[i] - air_lp[i]
            air = air_lp[i]
            gh = np.float32(10.0) ** (gain_db[i] / np.float32(20.0))
            ga = np.float32(10.0) ** (air_db[i] / np.float32(20.0))
            out[ch, i] = low[i] + mid + high * gh + air * ga
    return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def _box_smooth_1d(x: np.ndarray, radius: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo, hi = 0, 0
    for i in range(n):
        while hi < min(n, i + radius + 1):
            acc += x[hi]
            hi += 1
        while lo < max(0, i - radius):
            acc -= x[lo]
            lo += 1
        d = hi - lo
        out[i] = acc / np.float32(d) if d > 0 else x[i]
    return out


@register_processor("proxy_modal_force_dynamic_eq", track_group=MODAL_DISPLACEMENT)
class ProxyModalForceDynamicEq(BaseProcessor):
    """
    ProxySynth force-driven dynamic brightness.

    Replaces the block-averaged FFT path in
    `ProxyEqualizer._process_with_dynamic_eq` with a per-sample shelf
    whose gain tracks the force envelope continuously.

    Params
    ------
    boost_per_unit_force : float, default 0.5
        High-shelf gain in dB per unit force (matches the original
        `0.5 * block_force` in dB-ish terms after the log mapping).
    max_boost_db         : float, default 6.0
        Ceiling, prevents roaring at saturated impacts.
    air_per_unit_force   : float, default 1.5
        Additional air-band (> 10 kHz) boost, dB per unit force.
    low_shelf_hz         : float, default 250.0
    high_shelf_hz        : float, default 2000.0    # matches original f > 2000
    air_shelf_hz         : float, default 10000.0
    smooth_ms            : float, default 6.0
    sample_rate          : int,   default 48000
    sfps                 : float, default 24.0
    attr                 : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self,
                 boost_per_unit_force: float = 0.5,
                 max_boost_db: float = 6.0,
                 air_per_unit_force: float = 1.5,
                 low_shelf_hz: float = 250.0,
                 high_shelf_hz: float = 2000.0,
                 air_shelf_hz: float = 10000.0,
                 smooth_ms: float = 6.0,
                 sample_rate: int = 48000,
                 sfps: float = 24.0,
                 attr: str = "normal_force_magnitude",
                 **kw):
        super().__init__(
            boost_per_unit_force=boost_per_unit_force,
            max_boost_db=max_boost_db,
            air_per_unit_force=air_per_unit_force,
            low_shelf_hz=low_shelf_hz,
            high_shelf_hz=high_shelf_hz,
            air_shelf_hz=air_shelf_hz,
            smooth_ms=smooth_ms,
            sample_rate=sample_rate,
            sfps=sfps,
            attr=attr,
            **kw,
        )

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        C, S = x.shape
        if S <= 1:
            return x

        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        boost = float(self.p(props, "boost_per_unit_force", self.params["boost_per_unit_force"]))
        boost_max = float(self.p(props, "max_boost_db", self.params["max_boost_db"]))
        air_boost = float(self.p(props, "air_per_unit_force", self.params["air_per_unit_force"]))
        low_hz = float(self.p(props, "low_shelf_hz", self.params["low_shelf_hz"]))
        high_hz = float(self.p(props, "high_shelf_hz", self.params["high_shelf_hz"]))
        air_hz = float(self.p(props, "air_shelf_hz", self.params["air_shelf_hz"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))

        env = sample_envelope(
            seq, num_samples=S, sfps=sfps, sample_rate=sr,
            attr=attr, normalize=True, clip_positive=True,
        )

        # dB-domain shelf gains.
        # Original: 1 + 0.5 * f -> dB. In log terms: 20*log10(1+0.5*f).
        gain_db = np.float32(20.0) * np.log10(np.float32(1.0) + np.float32(boost) * env + np.float32(1e-9))
        air_db = np.float32(air_boost) * env  # simple linear dB

        # Clamp and smooth.
        gain_db = np.clip(gain_db, 0.0, boost_max).astype(np.float32)
        air_db = np.clip(air_db, 0.0, boost_max).astype(np.float32)

        radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))
        gain_db = _box_smooth_1d(gain_db, radius)
        air_db = _box_smooth_1d(air_db, radius)

        nyq = 0.5 * sr
        fc_low_hz = float(np.clip(low_hz, 20.0, nyq * 0.9))
        fc_high_hz = float(np.clip(high_hz, 200.0, nyq * 0.9))
        fc_air_hz = float(np.clip(air_hz, 500.0, nyq * 0.9))

        return _dynamic_bright_2d(
            x, gain_db, air_db,
            np.float32(fc_low_hz / sr),
            np.float32(fc_high_hz / sr),
            np.float32(fc_air_hz / sr),
        )
