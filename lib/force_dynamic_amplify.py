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

# pbrAudioSignals/lib/force_dynamic_amplify.py

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT, AUDIO_FORCE


@nb.njit(cache=True, nogil=True, fastmath=True)
def _box_smooth_1d(x: np.ndarray, half: np.int64) -> np.ndarray:
    """Centred box smoother, symmetric in the interior."""
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
def _dynamic_amplify_2d(x: np.ndarray,
                        force: np.ndarray,
                        base_gain: np.float32,
                        max_gain_lin: np.float32,
                        compression: np.float32,
                        smooth_radius: np.int64) -> np.ndarray:
    """
    Per-sample gain = base_gain * force_gain_compressed, exactly as in
    old SynthPostProcess._dynamic_amplify.
    """
    C, S = x.shape

    # smooth the force envelope once
    fe = np.empty(S, dtype=np.float32)
    for i in range(S):
        v = force[i]
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        fe[i] = v
    fe = _box_smooth_1d(fe, smooth_radius)

    # dynamic gain envelope
    gain_env = np.empty(S, dtype=np.float32)
    mean_db = np.float32(0.0)
    # First pass: raw dB of force_gain = 0.3 + 0.7 * fe
    for i in range(S):
        fg = np.float32(0.3) + np.float32(0.7) * fe[i]
        # 20 * log10(fg)
        gain_env[i] = np.float32(20.0) * np.log10(fg + np.float32(1e-10))
        mean_db += gain_env[i]
    mean_db /= np.float32(S)

    for i in range(S):
        # compress around the mean
        gain_db_c = gain_env[i] * (np.float32(1.0) - compression) + mean_db * compression
        # back to linear
        gain_env[i] = np.power(np.float32(10.0), gain_db_c / np.float32(20.0))

    # smear the combined gain envelope to prevent zipper noise
    gain_env = _box_smooth_1d(gain_env, np.int64(3))

    # clamp base gain
    if base_gain < np.float32(0.1):
        base_gain = np.float32(0.1)
    if base_gain > max_gain_lin:
        base_gain = max_gain_lin

    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * base_gain * gain_env[i]
    return out

@register_processor("force_dynamic_amplify",
                    track_group=MODAL_DISPLACEMENT,
                    also_valid_for={AUDIO_FORCE})
class ForceDynamicAmplify(BaseProcessor):
    """
    Force-envelope-driven dynamic amplification with compression.

    Mirrors the old SynthPostProcess._dynamic_amplify():
      * computes base_gain from a target RMS,
      * multiplies by a force-driven gain envelope (0.3 .. 1.0),
      * compresses that envelope in dB around its mean,
      * smooths both envelopes to avoid zipper noise.

    Params
    ------
    target_rms                : float, default 0.15
    max_gain_db               : float, default 20.0
    dynamic_range_compression : float, default 0.5   (0 = none, 1 = fully flat)
    smooth_ms                 : float, default 5.0
    sample_rate               : int,   default 48000
    """

    required_side = {"force"}

    def __init__(self,
                 target_rms: float = 0.15,
                 max_gain_db: float = 20.0,
                 dynamic_range_compression: float = 0.5,
                 smooth_ms: float = 5.0,
                 sample_rate: int = 48000, **kw):
        super().__init__(target_rms=target_rms,
                                                 max_gain_db=max_gain_db,
                         dynamic_range_compression=dynamic_range_compression,
                         smooth_ms=smooth_ms,
                         sample_rate=sample_rate, **kw)

    def _process(self, x, props):
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x

        target_rms = float(self.p(props, "target_rms", self.params["target_rms"]))
        max_gain_db = float(self.p(props, "max_gain_db", self.params["max_gain_db"]))
        comp = float(self.p(props, "dynamic_range_compression",
                            self.params["dynamic_range_compression"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))

        # base gain from current RMS
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        base_gain = (target_rms / rms) if rms > 0.0 else 1.0
        max_lin = 10.0 ** (max_gain_db / 20.0)
        base_gain = float(max(0.1, min(base_gain, max_lin)))

        # normalize force to 0..1
        mx = float(force.max()) if force.max() > 0 else 1.0
        force = (force / mx).astype(np.float32, copy=False)

        smooth_radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))

        return _dynamic_amplify_2d(
            x, force,
            np.float32(base_gain),
            np.float32(max_lin),
            np.float32(comp),
            smooth_radius,
        )
