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

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT, AUDIO_FORCE

@register_processor("force_phase_align", track_group=MODAL_DISPLACEMENT, also_valid_for={AUDIO_FORCE})
class ForcePhaseAlign(BaseProcessor):
    """
    Align audio transients to force-signal onsets using short crossfades.

    Mirrors the old SynthPostProcess._phase_align():
      1. Find transient onsets in |force| via thresholded first difference.
      2. Find transient onsets in |x| the same way.
      3. For each force onset, pick nearest audio onset within a window.
      4. Crossfade the signal by the time-shift between them.

    Params
    ------
    crossfade_samples : int,   default 256
    sample_rate       : int,   default 48000
    onset_window_ms   : float, default 10.0     # +-search window for matching
    threshold_sigma   : float, default 2.0      # diff > k*std(diff) -> onset
    """

    required_side = {"force"}

    def __init__(self,
                 crossfade_samples: int = 256,
                 sample_rate: int = 48000,
                 onset_window_ms: float = 10.0,
                 threshold_sigma: float = 2.0, **kw):
        super().__init__(crossfade_samples=crossfade_samples,
                         sample_rate=sample_rate,
                         onset_window_ms=onset_window_ms,
                         threshold_sigma=threshold_sigma, **kw)

    def _detect_onsets(self, sig: np.ndarray, sigma_k: float) -> np.ndarray:
        diff = np.diff(np.abs(sig), prepend=0)
        s = float(np.std(diff))
        if s <= 0.0:
            return np.array([], dtype=np.int64)
        mask = diff > s * sigma_k
        return np.flatnonzero(mask).astype(np.int64)

    def _process(self, x, props):
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x

        cf = int(self.p(props, "crossfade_samples", self.params["crossfade_samples"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        win_ms = float(self.p(props, "onset_window_ms", self.params["onset_window_ms"]))
        sigma_k = float(self.p(props, "threshold_sigma", self.params["threshold_sigma"]))

        # onsets are detected on the force signal once
        force_onsets = self._detect_onsets(np.abs(force), sigma_k)
        if force_onsets.size == 0:
            return x

        window = max(1, int(win_ms * sr / 1000.0))
        half = window // 2

        out = x.astype(np.float32, copy=True)
        C = x.shape[0]

        for ch in range(C):
            sig = x[ch]
            sig_onsets = self._detect_onsets(np.abs(sig), sigma_k)
            if sig_onsets.size == 0:
                continue

            for f_idx in force_onsets:
                lo = f_idx - window
                hi = f_idx + window
                # onsets near this force hit
                near = sig_onsets[(sig_onsets > lo) & (sig_onsets < hi)]
                if near.size == 0:
                    continue

                nearest = int(near[np.argmin(np.abs(near - f_idx))])
                shift = int(nearest - f_idx)
                if abs(shift) >= half:
                    continue

                # crossfade region of length `cf`, clamped to signal
                end = min(f_idx + cf, S - abs(shift))
                if end <= f_idx:
                    continue

                fade_in = np.linspace(0.0, 1.0, end - f_idx, dtype=np.float32)
                fade_out = 1.0 - fade_in

                if shift > 0:
                    # audio arrives late -> pull shifted later into this window
                    fr = f_idx
                    to = end
                    if to > S:
                        continue
                    shifted = x[ch, fr - shift: to - shift] if fr - shift >= 0 else None
                    if shifted is None:
                        continue
                    out[ch, fr:to] = x[ch, fr:to] * fade_out + shifted * fade_in
                elif shift < 0:
                    fr = f_idx
                    to = min(end + (-shift), S)
                    shifted = x[ch, fr + (-shift): to + (-shift)]
                    if shifted.shape[0] < (to - fr):
                        continue
                    out[ch, fr:to] = x[ch, fr:to][: to - fr] * fade_out[: to - fr] + shifted[: to - fr] * fade_in[: to - fr]

        return out
