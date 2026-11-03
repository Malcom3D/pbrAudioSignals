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
Phase-align MODAL_DISPLACEMENT tracks to a reference side channel
(typically a NOISE track from the same object: sliding / scraping /
rolling).

The problem the original ``force_phase_align`` did not solve correctly
is that a modal resonator introduces a small group delay (dominated by
the mean T60) relative to the raw force, while the noise synthesizer
introduces essentially zero delay. If we simply cross-fade the two,
the impact sounds hollow or flanged.

This processor uses the ForceDataSequence as a *time reference* to
estimate the modal group delay, then delays the noise-reference signal
by that amount before cross-fading at each detected event.

Design
------
1. Reference signal R (side channel, typically sliding/scraping/rolling).
2. At every detected event (from the ForceDataSequence contact_type),
   estimate the group delay of x relative to R via a short cross-correlation
   in a ±window around the event.
3. Apply fractional delay to R under the *same* force envelope so that
   quiet samples stay aligned with x (i.e. at 0 delay = no-op).
4. Blend R into x with a force-weighted cross-fade at the event.

Because this is a *side channel* alignment, we never change length.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _crossfade_2d(x: np.ndarray, y_ref: np.ndarray, force: np.ndarray, delay_samples: np.ndarray, crossfade: np.int64, attack: np.int64, release: np.int64) -> np.ndarray:
    """
    For each channel:
      - fractional-shift y_ref by delay_samples[i]
      - apply short envelope-shaped cross-fade toward the shifted y_ref
        at every event, magnitude controlled by force envelope.
    """
    C, S = x.shape
    out = np.empty_like(x)

    for ch in nb.prange(C):
        for i in range(S):
            d = delay_samples[i]
            # sample y_ref at fractional position i - d
            if d > 0:
                idx = i - d
                i0 = int(np.floor(idx))
                frac = np.float32(idx - i0)
                if i0 < 0:
                    y = np.float32(0.0)
                elif i0 + 1 >= S:
                    y = y_ref[ch, S - 1]
                else:
                    y = y_ref[ch, i0] * (np.float32(1.0) - frac) + y_ref[ch, i0 + 1] * frac
            elif d < 0:
                idx = i - d
                i0 = int(np.floor(idx))
                frac = np.float32(idx - i0)
                if i0 < 0:
                    y = y_ref[ch, 0]
                elif i0 + 1 >= S:
                    y = np.float32(0.0)
                else:
                    y = y_ref[ch, i0] * (np.float32(1.0) - frac) + y_ref[ch, i0 + 1] * frac
            else:
                y = y_ref[ch, i]

            # cross-fade weight: ramps up on attack, decays on release
            f = force[i]
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            w = f
            out[ch, i] = x[ch, i] * (np.float32(1.0) - w) + y * w
    return out


@register_processor("force_modal_phase_align", track_group=MODAL_DISPLACEMENT)
class ForceModalPhaseAlign(BaseProcessor):
    """
    Align a modal displacement track with a side reference (e.g. the
    per-object noise track) using the ForceDataSequence as the time
    reference.

    Params
    ------
    reference_key        : str, default "modal_ref"  # props.custom key
    max_delay_ms         : float, default 5.0
    window_ms            : float, default 20.0       # xcorr window around event
    event_threshold      : float, default 0.2        # force envelope threshold
    target_contact_types : tuple, default (IMPACT, SCRAPING, SLIDING, MIXED) tuple = (1, 2, 3, 5)
    attr                 : str,   default "normal_force_magnitude"
    sample_rate          : int,   default 48000
    sfps                 : float, default 24.0
    """

    required_side = {"force"}

    def __init__(self, reference_key: str = "modal_ref", max_delay_ms: float = 5.0, window_ms: float = 20.0, event_threshold: float = 0.2, target_contact_types: tuple = (1, 2, 3, 5), attr: str = "normal_force_magnitude", sample_rate: int = 48000, sfps: float = 24.0, **kw):
        super().__init__(reference_key=reference_key, max_delay_ms=max_delay_ms, window_ms=window_ms, event_threshold=event_threshold, target_contact_types=target_contact_types, attr=attr, sample_rate=sample_rate, sfps=sfps, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        if props is None or not isinstance(getattr(props, "custom", None), dict):
            return x
        ref_key = self.p(props, "reference_key", self.params["reference_key"])
        ref = props.custom.get(ref_key)
        if ref is None:
            return x
        ref = np.ascontiguousarray(ref, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if ref.shape[1] != x.shape[1]:
            # pad or crop
            S = x.shape[1]
            if ref.shape[1] > S:
                ref = ref[:, :S]
            else:
                pad = np.zeros((ref.shape[0], S - ref.shape[1]), dtype=np.float32)
                ref = np.concatenate([ref, pad], axis=1)

        # Broadcast reference to (C, S) if it has fewer channels
        if ref.shape[0] != x.shape[0]:
            if ref.shape[0] == 1:
                ref = np.broadcast_to(ref, x.shape).copy()
            else:
                # pad channels
                padded = np.zeros_like(x)
                cmin = min(ref.shape[0], x.shape[0])
                padded[:cmin] = ref[:cmin]
                ref = padded

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        max_delay_ms = float(self.p(props, "max_delay_ms", self.params["max_delay_ms"]))
        window_ms = float(self.p(props, "window_ms", self.params["window_ms"]))
        thr = float(self.p(props, "event_threshold", self.params["event_threshold"]))
        types = self.p(props, "target_contact_types", self.params["target_contact_types"])
        try:
            types_tuple = tuple(int(t) for t in types)
        except Exception:
            types_tuple = (1, 2, 3, 5)

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)
        ctypes = sample_contact_types(seq, num_samples=S, sfps=sfps, sample_rate=sr)

        # Detect events: contiguous sections where env > thr and type in types
        max_delay = int(max_delay_ms * sr / 1000.0)
        window = max(2 * max_delay + 8, int(window_ms * sr / 1000.0))
        delay = np.zeros(S, dtype=np.float32)

        on = False
        start = 0
        for i in range(S):
            active = env[i] >= thr and int(ctypes[i]) in types_tuple
            if active and not on:
                on = True
                start = i
            elif not active and on:
                on = False
                # estimate delay in this event
                end = min(S, i)
                lo = max(0, start - max_delay)
                hi = min(S, end + max_delay)
                seg_len = hi - lo
                if seg_len > max(8, 2 * max_delay + 4):
                    x_seg = x[0, lo:hi].astype(np.float64)
                    r_seg = ref[0, lo:hi].astype(np.float64)
                    # remove means
                    x_seg = x_seg - x_seg.mean()
                    r_seg = r_seg - r_seg.mean()
                    nx = np.linalg.norm(x_seg)
                    nr = np.linalg.norm(r_seg)
                    if nx > 1e-9 and nr > 1e-9:
                        corr = np.correlate(x_seg, r_seg, mode="full")
                        # peak index maps to lag = idx - (seg_len - 1)
                        lag = int(np.argmax(corr)) - (seg_len - 1)
                        # clip to ±max_delay
                        lag = max(-max_delay, min(max_delay, lag))
                        for k in range(start, end):
                            delay[k] = float(lag)

        # Zero-out delay wherever env is below threshold
        for i in range(S):
            if env[i] < thr * 0.5:
                delay[i] = 0.0

        # Smooth delay to avoid zipper noise
        smoothed = np.zeros_like(delay)
        alpha = np.float32(0.05)
        prev = np.float32(0.0)
        for i in range(S):
            prev = (np.float32(1.0) - alpha) * prev + alpha * delay[i]
            smoothed[i] = prev

        # crossfade at events
        cf = max(8, int(0.002 * sr))     # 2 ms crossfade
        attack = cf
        release = cf

        return _crossfade_2d(x, ref, env, smoothed, np.int64(cf), np.int64(attack), np.int64(release))
