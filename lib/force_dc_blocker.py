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
from ..core.groups import AUDIO_FORCE
from .force_sidechannel import get_force_sequence, sample_envelope, eval_at_frame


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _adaptive_dc_block_2d(
    x: np.ndarray,
    force: np.ndarray,
    alpha_quiet: np.float32,
    alpha_active: np.float32,
) -> np.ndarray:
    """
    Per-sample DC-blocker whose pole moves with the force envelope:

        force ~ 0   ->  alpha -> alpha_quiet  (tight DC removal)
        force ~ 1   ->  alpha -> alpha_active (loose, preserves transients)

    y[n] = x[n] - x[n-1] + alpha[n] * y[n-1]
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        xp = np.float32(0.0)
        yp = np.float32(0.0)
        for i in range(S):
            f = force[i]
            a = alpha_quiet + (alpha_active - alpha_quiet) * f
            xi = x[ch, i]
            yi = xi - xp + a * yp
            out[ch, i] = yi
            xp = xi
            yp = yi
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _subtract_local_mean_2d(
    x: np.ndarray,
    force: np.ndarrayarray,
    half_window: np.int64,
    min_force: np.float32,
) -> np.ndarray:
    """
    Subtract a local mean computed only over force-quiet samples that lie
    within ±half_window of each sample. Because we only average over
    force-quiet samples, an isolated transient in the audio-force signal
    (a real impact) does *not* get subtracted.
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            lo = i - half_window
            if lo < 0:
                lo = 0
            hi = i + half_window
            if hi > S:
                hi = S
            acc = np.float32(0.0)
            n = 0
            for j in range(lo, hi):
                if force[j] <= min_force:
                    acc += x[ch, j]
                    n += 1
            if n > 0:
                out[ch, i] = x[ch, i] - acc / np.float32(n)
            else:
                out[ch, i] = x[ch, i]
    return out


@register_processor("force_dc_blocker", track_group=AUDIO_FORCE)
class DCBlockerForce(BaseProcessor):
    """
    Remove DC / envelope-shaped bias from audio-force tracks, using the
    ForceDataSequence as the reference for *where* the DC lives.

    Steps
    -----
    1. Build a dense per-sample force envelope from ``ForceDataSequence``
       (default: normal_force_magnitude, normalized 0..1).
    2. Optionally subtract a *local* mean computed only over force-quiet
       samples in a ±window, so real impacts are preserved.
    3. Run a per-sample adaptive DC blocker whose pole curves with the
       force envelope: tight cutoff when quiet, loose when active.

    Params
    ------
    alpha_quiet      : float = 0.999    # pole when force == 0 (tightest)
    alpha_active     : float = 0.9999   # pole when force == 1 (loosest)
    local_mean_ms    : float = 20.0     # window for the envelope-shaped mean
    min_force        : float = 0.02     # force < this => "quiet sample"
    attr             : str   = "normal_force_magnitude"
    sample_rate      : int   = 48000
    sfps             : float = 24.0     # sub-frames per second (from config)
    enable_local_mean: bool  = True
    """

    required_side = {"force"}

    def __init__(self, alpha_quiet: float = 0.999, alpha_active: float = 0.9999, local_mean_ms: float = 20.0, min_force: float = 0.02, attr: str = "normal_force_magnitude", sample_rate: int = 48000, sfps: float = 24.0, enable_local_mean: bool = True, **kw):
        super().__init__(alpha_quiet=alpha_quiet, alpha_active=alpha_active, local_mean_ms=local_mean_ms, min_force=min_force, attr=attr, sample_rate=sample_rate, sfps=sfps, enable_local_mean=enable_local_mean, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        aq = np.float32(self.p(props, "alpha_quiet", self.params["alpha_quiet"]))
        aa = np.float32(self.p(props, "alpha_active", self.params["alpha_active"]))
        min_f = np.float32(self.p(props, "min_force", self.params["min_force"]))
        lm_ms = float(self.p(props, "local_mean_ms", self.params["local_mean_ms"]))
        use_lm = bool(self.p(props, "enable_local_mean", self.params["enable_local_mean"]))

        # Dense per-sample force envelope in [0,1].
        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        y = x
        if use_lm:
            half = np.int64(max(1, int(lm_ms * sr / 1000.0)))
            y = _subtract_local_mean_2d(y, env, half, min_f)

        return _adaptive_dc_block_2d(y, env, aq, aa)
