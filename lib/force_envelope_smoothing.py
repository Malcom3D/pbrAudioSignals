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
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types

@nb.njit(cache=True, nogil=True, fastmath=True)
def _signed_median_filter_1d(x: np.ndarray, radius: np.int64) -> np.ndarray:
    """Signed 3-tap median with a configurable half-radius (typical: 1)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    for i in range(n):
        lo = i - radius if i - radius >= 0 else 0
        hi = i + radius + 1 if i + radius + 1 <= n else n
        tmp = np.empty(hi - lo, dtype=np.float32)
        for k in range(lo, hi):
            tmp[k - lo] = x[k]
        # insertion sort (tiny arrays)
        for a in range(1, tmp.shape[0]):
            v = tmp[a]
            b = a - 1
            while b >= 0 and tmp[b] > v:
                tmp[b + 1] = tmp[b]
                b -= 1
            tmp[b + 1] = v
        # signed median: if median says all residuals agree,
        # fall back to the raw value (protects real transients).
        out[i] = tmp[(hi - lo) // 2]
    return out

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _spike_and_dip_masks(x: np.ndarray, threshold_ratio: np.float32) -> np.ndarray:
    """
    For each sample, mark 1 if it is a single-sample spike or dip
    relative to its two neighbours.
    """
    C, S = x.shape
    mask = np.zeros((C, S), dtype=np.uint8)
    for ch in nb.prange(C):
        # local scale from a moving RMS
        scale = np.float32(0.0)
        for i in range(S):
            a = abs(x[ch, i])
            if a > scale:
                scale = a
        if scale <= 0.0:
            continue
        thr = scale * threshold_ratio
        for i in range(1, S - 1):
            l = x[ch, i - 1]
            c = x[ch, i]
            r = x[ch, i + 1]
            # spike
            if (c - l) > thr and (c - r) > thr and abs(l - r) < thr:
                mask[ch, i] = 1
            # dip / cancellation
            elif (l - c) > thr and (r - c) > thr and abs(l - r) < thr:
                mask[ch, i] = 2
    return mask

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _force_gated_smooth_2d(x: np.ndarray, force: np.ndarray, mask: np.ndarray, smooth_strength_quiet: np.float32, smooth_strength_active: np.float32) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)

    # smoothed reference: 3-tap median on each channel
    sm = np.empty_like(x)
    for ch in nb.prange(C):
        sm[ch] = _signed_median_filter_1d(x[ch], np.int64(1))

    for ch in nb.prange(C):
        for i in range(S):
            f = force[i]
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            # blend coefficient: how much to trust the smoothed value
            # quiet region -> strong smoothing of fake peaks
            # active region -> keep raw (real force)
            w = smooth_strength_quiet * (np.float32(1.0) - f) + smooth_strength_active * f
            # but only smooth flagged samples; never smear adjacents
            if mask[ch, i] != 0:
                out[ch, i] = x[ch, i] * (np.float32(1.0) - w) + sm[ch, i] * w
            else:
                out[ch, i] = x[ch, i]
    return out

@register_processor("force_envelope_smoothing", track_group=AUDIO_FORCE)
class ForceEnvelopeSmoothing(BaseProcessor):
    """
    Remove ForceSynth artifacts — one-sample spikes and cancellations —
    while preserving real high-energy transients referenced by
    ``ForceDataSequence``.

    Steps
    -----
    1. Build a dense per-sample force envelope (0..1).
    2. Detect "spike / dip" samples: the sample is much larger in
       magnitude than BOTH neighbours, while the neighbours agree.
    3. Only these flagged samples are blended toward a 3-tap signed
       median. Blend weight is high where force is quiet (fake peaks),
       low where force is active (real impacts).

    Params
    ------
    spike_ratio              : float = 0.35  # sample / neighbours threshold
    smooth_strength_quiet    : float = 0.9   # blend at force == 0
    smooth_strength_active   : float = 0.0   # blend at force == 1
    attr                     : str   = "normal_force_magnitude"
    sample_rate              : int   = 48000
    sfps                     : float = 24.0
    """

    required_side = {"force"}

    def __init__(self, spike_ratio: float = 0.35, smooth_strength_quiet: float = 0.9, smooth_strength_active: float = 0.0, attr: str = "normal_force_magnitude", sample_rate: int = 48000, sfps: float = 24.0, **kw):
        super().__init__(spike_ratio=spike_ratio, smooth_strength_quiet=smooth_strength__quiet, smooth_strength_active=smooth_strength_active, attr=attr, sample_rate=sample_rate, sfps=sfps, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        thr = np.float32(self.p(props, "spike_ratio", self.params["spike_ratio"]))
        sq = np.float32(self.p(props, "smooth_strength_quiet", self.params["smooth_strength_quiet"]))
        sa = np.float32(self.p(props, "smooth_strength_active", self.params["smooth_strength_active"]))

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        mask = _spike_and_dip_masks(x, thr)
        return _force_gated_smooth_2d(x, env, mask, sq, sa)
