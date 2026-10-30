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
from ..core.groups import MODAL_DISPLACEMENT, AUDIO_FORCE

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _force_ema_2d(x: np.ndarray,
                  force: np.ndarray,
                  alpha_min: np.float32,
                  alpha_max: np.float32) -> np.ndarray:
    """
    Time-varying EMA per channel:
        alpha[i] = alpha_min + (alpha_max - alpha_min) * (1 - force[i])
        y[i]     = (1 - alpha[i]) * x[i] + alpha[i] * y[i-1]

    Force is expected in 0..1.
    """
    C, S = x.shape
    out = np.empty_like(x)

    # smoothing factor: high when force is low
    sf = np.empty(S, dtype=np.float32)
    for i in range(S):
        f = force[i]
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        sf[i] = np.float32(1.0) - f

    for ch in nb.prange(C):
        out[ch, 0] = x[ch, 0]
        for i in range(1, S):
            alpha = alpha_min + (alpha_max - alpha_min) * sf[i]
            out[ch, i] = (np.float32(1.0) - alpha) * x[ch, i] + alpha * out[ch, i - 1]
    return out

@register_processor("force_adaptive_smoothing", track_group=MODAL_DISPLACEMENT, also_valid_for={AUDIO_FORCE})
class ForceAdaptiveSmoothing(BaseProcessor):
    """
    Adaptive EMA smoothing modulated by the force envelope.

    Mirrors SynthPostProcess._adaptive_smoothing():
      * when force is high  -> alpha -> alpha_min (fast, preserves transients)
      * when force is low   -> alpha -> alpha_max (slow, smooths quiet tails)

    Params
    ------
    alpha_min        : float, default 0.1
    alpha_max        : float, default 0.9
    smooth_ms        : float, default 0.0   # >0 pre-smooths the force envelope
    sample_rate      : int,   default 48000
    """

    required_side = {"force"}

    def __init__(self,
                 alpha_min: float = 0.1,
                 alpha_max: float = 0.9,
                 smooth_ms: float = 0.0,
                 sample_rate: int = 48000, **kw):
        super().__init__(alpha_min=alpha_min,
                         alpha_max=alpha_max,
                         smooth_ms=smooth_ms,
                         sample_rate=sample_rate, **kw)

    def _process(self, x, props):
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x

        a_min = float(self.p(props, "alpha_min", self.params["alphaalpha_min"]))
        a_max = float(self.p(props, "alpha_max", self.params["alpha_max"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))

        # normalize force
        f = np.abs(force).astype(np.float32)
        mx = float(f.max()) if f.max() > 0 else 1.0
        f = f / mx

        # optional pre-smoothing of the force envelope
        if smooth_ms > 0.0:
            from scipy.ndimage import gaussian_filter1d
            sigma = max(1.0, smooth_ms * sr / 1000.0 / 2.0)
            f = gaussian_filter1d(f, sigma=sigma).astype(np.float32)

        return _force_ema_2d(x, f, np.float32(a_min), np.float32(a_max))
