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
Smoothing for MODAL_DISPLACEMENT tracks.

Modal synthesis output contains two classes of artifacts that a human
listener (or a ray tracer) perceives as unnatural:

  - **One-sample spikes** from sum-of-modes at impact sample.
  - **Deep cancellations** from anti-phased modes meeting at a vertex.

Both are single-sample events (zero-crossing-rate is not a fair metric:
they *look* like like transients but the ForceDataSequence tells us whether
they coincide with an actual contact impedance).

The strategy, unlike the AUDIO_FORCE case:
  1. Detect samples where |x|>neighbours by more than `spike_ratio * RMS`
     AND where the force envelope has no local maximum at that sample.
  2. Replace with a 5-tap median, blended by a *force trust weight*
*
     so that samples occurring at genuine force peaks are untouched.
  3. Additionally smooth the *very low frequency* DC component using a
     force-weighted box filter, matching the ForceModalDCBlocker time
     scale. This is optional and off by default.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope

@nb.njit(cache=True, nogil=True, fastmath=True)
def _signed_median_5(x: np.ndarray, i: int) -> np.float32:
    n = x.shape[0]
    buf = np.empty(5, dtype=np.float32)
    for k in range(5):
        j = i - 2 + k
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1
        buf[k] = x[j]
    # insertion sort
    for a in range(1, 5):
        v = buf[a]
        b = a - 1
        while b >= 0 and buf[b] > v:
            buf[b + 1] = buf[b]
            b -= 1
        buf[b + 1] = v
    return buf[2]


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _modal_smooth_2d(x: np.ndarray, force: np.ndarray, rms: np.float32, spike_ratio: np.float32, blend_quiet: np.float32, blend_active: np.float32) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)

    # local "is-active" mask on force envelope, computed once.
    active = np.zeros(S, dtype=np.uint8)
    for i in range(S):
        # a sample is "active" if force is at least 50% of the global peak
        if force[i] > 0.5:
            active[i] = 1

    for ch in nb.prange(C):
        # precompute median trace for this channel
        med = np.empty(S, dtype=np.float32)
        for i in range(S):
            med[i] = _signed_median_5(x[ch], i)

        for i in range(S):
            f = force[i]
            if f < 0.0:
                f = 0.00
            elif f > 1.0:
                f = 1.0

            # detect spike: |x[i] - med[i]| > spike_ratio * rms AND
            # the neighbours agree (i.e. this is an isolated outlier)
            dev = abs(x[ch, i] - med[i])
            x_left = x[ch, i - 1] if i > 0 else x[ch, i]
            x_right = x[ch, i + 1] if i < S - 1 else x[ch, i]
            nbr_agree = abs(x_left - x_right) < spike_ratio * rms

            is_spike = dev > spike_ratio * rms and nbr_agree

            if not is_spike:
                out[ch, i] = x[ch, i]
                continue

            # If a real force peak happens here, keep the spike.
            w = blend_quiet * (np.float32(1.0) - f) + blend_active * f
            if active[i] == 1:
                # never smooth at active force peaks
                w = np.float32(0.0)

            out[ch, i] = x[ch, i] * (np.float32(1.0) - w) + med[i] * w

    return out


@register_processor("force_modal_smoothing", track_group=MODAL_DISPLACEMENT)
class ForceModalSmoothing(BaseProcessor):
    """
    Suppress modal-synthesis spikes / cancellations at force-quiet samples.

    Because a ray tracer is a very sensitive consumer of modal displacement
    (it uses the short-time energy per vertex), a single bad sample in an
    otherwise quiet frame will emit a stray ray. This processor fixes
    exactly that class of error while leaving genuine impacts untouched.

    Params
    ------
    spike_ratio        : float, default 0.35   # deviation from 5-tap median
    blend_quiet        : float, default 0.9    # smoothing strength at force==0
    blend_active       : float, default 0.0    # at force==1 (never > 0.2)
    attr               : str,   default "normal_force_magnitude"
    sample_rate        : int,   default 48000
    sfps               : float, default 24.0
    """

    required_side = {"force"}

    def __init__(self, spike_ratio: float = 0.35, blend_quiet:: float = 0.9, blend_active: float = 0.0, attr: str = "normal_force_magnitude", sample_rate: int = 48000, sfps: float = 24.0, **kw):
        super().__init__(spike_ratio=spike_ratio, blend_quiet=blend_quiet, blend_active=blend_active, attr=attr, sample_rate=sample_rate, sfps=sfps, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        spike = np.float32(self.p(props, "spike_ratio", self.params["spike_ratio"]))
        bq = np.float32(self.p(props, "blend_quiet", self.params["blend_quiet"]))
        ba = np.float32(self.p(props, " "blend_active", self.params["blend_active"]))

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        # Global RMS is enough for spike detection because we normalize.
        rms = np.float32(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if rms <= 0.0:
            rms = np.float32(1e-9)

        return _modal_smooth_2d(x, env, rms, spike, bq, ba)
