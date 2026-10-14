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
import numba as nb

from ..core.processor import BaseProcessor, register_processor


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _force_weighted_gain_2d(x: np.ndarray, force: np.ndarray,
                            base_gain: np.float32,
                            gain_low: np.float32,
                            gain_high: np.float32,
                            compression: np.float32,
                            smooth_radius: np.int64) -> np.ndarray:
    C, S = x.shape
    # smooth the force envelope once
    force_s = _box_smooth_1d_local(force, smooth_radius)

    gain_env = np.empty(S, dtype=np.float32)
    for i in range(S):
        f = force_s[i]
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        # linear interpolation between gain_low and gain_high
        g = gain_low + (gain_high - gain_low) * f
        # apply compression around the mean gain
        mean_g = np.float32(0.5) * (gain_low + gain_high)
        g = mean_g + (g - mean_g) * (np.float32(1.0) - compression)
        gain_env[i] = g

    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * base_gain * gain_env[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def _box_smooth_1d_local(x: np.ndarray, half: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo = 0
    hi = 0
    for i in range(n):
        while hi < min(n, i + half + 1):
            acc += x[hi]
            hi += 1
        while lo < max(0, i - half):
            acc -= x[lo]
            lo += 1
        denom = hi - lo
        out[i] = acc / np.float32(denom) if denom > 0 else x[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _force_weighted_blend_2d(dry: np.ndarray, wet: np.ndarray,
                             force: np.ndarray,
                             dry_wet: np.float32) -> np.ndarray:
    C, S = dry.shape
    out = np.empty_like(dry)
    for ch in nb.prange(C):
        for i in range(S):
            f = force[i]
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            # high force -> more wet, low force -> more dry
            w = dry_wet * (np.float32(0.5) + np.float32(0.5) * f)
            out[ch, i] = dry[ch, i] * (np.float32(1.0) - w) + wet[ch, i] * w
    return out


@register_processor("force_weighted_gain")
class ForceWeightedGain(BaseProcessor):
    """
    Time-varying gain modulated by the force side-channel.

    Params
    ------
    target_rms     : float, default 0.15   # base gain is auto-computed to hit this
    gain_low       : float, default 0.3    # gain at force = 0
    gain_high      : float, default 1.0    # gain at force = 1
    max_gain_db    : float, default 20.0
    compression    : float, default 0.5    # 0 = none, 1 = fully flat
    smooth_ms      : float, default 5.0
    sample_rate    : int,   default 48000
    """

    def __init__(self, target_rms: float = 0.15,
                 gain_low: float = 0.3,
                 gain_high: float = 1.0,
                 max_gain_db: float = 20.0,
                 compression: float = 0.5,
                 smooth_ms: float = 5.0,
                 sample_rate: int = 48000, **kw):
        super().__init__(target_rms=target_rms,
                         gain_low=gain_low,
                         gain_high=gain_high,
                         max_gain_db=max_gain_db,
                         compression=compression,
                         smooth_ms=smooth_ms,
                         sample_rate=sample_rate, **kw)

    def _process(self, x, props):
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x

        target_rms = float(self.p(props, "target_rms", self.params["target_rms"]))
        gain_low = float(self.p(props, "gain_low", self.params["gain_low"]))
        gain_high = float(self.p(props, "gain_high", self.params["gain_high"]))
        max_gain_db = float(self.p(props, "max_gain_db", self.params["max_gain_db"]))
        compression = float(self.p(props, "compression", self.params["compression"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        base_gain = target_rms / rms if rms > 0 else 1.0
        max_lin = 10.0 ** (max_gain_db / 20.0)
        base_gain = max(0.1, min(base_gain, max_lin))

        mx = force.max() if force.max() > 0 else 1.0
        force = (force / mx).astype(np.float32, copy=False)

        smooth_radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))
        return _force_weighted_gain_2d(
            x, force,
            np.float32(base_gain),
            np.float32(gain_low),
            np.float32(gain_high),
            np.float32(compression),
            smooth_radius,
        )


@register_processor("force_weighted_blend")
class ForceWeightedBlend(BaseProcessor):
    """
    Blend the incoming signal with a target signal based on force.

    Params
    ------
    dry_wet          : float, default 0.85
    target_key       : str,   default "wet"   # props.custom[target_key]
    """

    def __init__(self, dry_wet: float = 0.85, target_key: str = "wet", **kw):
        super().__init__(dry_wet=dry_wet, target_key=target_key, **kw)

    def _process(self, x, props):
        dry_wet = np.float32(self.p(props, "dry_wet", self.params["dry_wet"]))
        target_key = self.p(props, "target_key", self.params["target_key"])

        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        wet = self.side(props, target_key, expected_len=S)
        if force is None or wet is None:
            return x

        mx = force.max() if force.max() > 0 else 1.0
        force = (force / mx).astype(np.float32, copy=False)
        # broadcast wet (1-D) to (C, S)
        wet_2d = np.broadcast_to(wet.reshape(1, -1), x.shape).copy()
        return _force_weighted_blend_2d(x, wet_2d, force, dry_wet)

