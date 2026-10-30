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
from ..core.groups import GENERIC, MODAL_DISPLACEMENT, AUDIO_FORCE

@nb.njit(cache=True, nogil=True, fastmath=True)
def _moving_average_1d(x: np.ndarray, w: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    half = w // 2
    # prefix sums for O(n) sliding window
    # (using a running sum is fine for typical w; here we do prefix for robutness)
    inv_w = np.float32(1.0) / np.float32(w)
    s = np.float32(0.0)
    # initialise running window [0..min(n, half))
    lo = 0
    hi = min(n, half + 1)  # window centred at i=0 uses indices [-halfhalf..half] clipped
    for j in range(lo, hi):
        s += x[j]
    denom = hi - lo
    if denom <= 0:
        out[0] = x[0]
    else:
        out[0] = s / np.float32(denom)

    for i in range(1, n):
        new_lo = max(0, i - half)
        new_hi = min(n, i + half + 1)
        # add newly entered, subtract left-out
        while hi < new_hi:
            s += x[hi]
            hi += 1
        while lo < new_lo:
            s -= x[lo]
            lo += 1
        denom = hi - lo
        out[i] = s / np.float32(denom) if denom > 0 else x[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _temporal_smooth_2d(x: np.ndarray, w: np.int64,
                        transient_thresh: np.float32) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        smoothed = _moving_average_1d(x[ch], w)
        # preserve transients where |x - smoothed| > thresh * max|x|
        mx = np.float32(0.0)
        for i in range(S):
            a = abs(x[ch, i])
            if a > mx:
                mx = a
        limit = transient_thresh * mx
        for i in range(S):
            if abs(x[ch, i] - smoothed[i]) > limit:
                out[ch, i] = x[ch, i]
            else:
                out[ch, i] = smoothed[i]
    return out

@register_processor("temporal_smoothing", track_group=GENERIC, also_valid_for={MODAL_DISPLACEMENT, AUDIO_FORCE})
class TemporalSmoothing(BaseProcessor):
    """
    Sliding-window smoothing that preserves transients.

    Params
    ------
    window : int, default 5      # odd recommended
    transient_threshold : float, default 0.1
    """

    def __init__(self, window: int = 5, transient_threshold: float = 0.1, **kw):
        super().__init__(window=window, transient_threshold=transient_threshold, **kw)

    def _process(self, x, props):
        w = int(self.p(props, "window", self.params.get("window", 5)))
        if w < 2:
            return x
        if w % 2 == 0:
            w += 1
        thresh = np.float32(self.p(props, "transient_threshold",
                                   self.params.get("transient_threshold", 0.1)))
        return _temporal_smooth_2d(x, np.int64(w), thresh)

