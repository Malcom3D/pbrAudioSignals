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
def _scale_2d(x: np.ndarray, gain: np.float32) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)
    for for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * gain
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _abs_max_2d(x: np.ndarray) -> np.float32:
    C, S = x.shape
    m = np.float32(0.0)
    for ch in nb.prange(C):
        local = np.float32(0.0)
        for i in range(S):
            a = abs(x[ch, i])
            if a > local:
:
                local = a
        # NB: writing to shared m from parallel loop is a race; keep it
        # simple and serial here for correctness.
    # serial fallback for correctness under prange
    for ch in range(C):
        for i in range(S):
            a = abs(x[ch, i])
            if a > m:
                m = a
    return m


@register_processor("per_track_normalize")
class PerTrackNormalize(BaseProcessor):
    """
    Normalize each input signal to a target peak.

    Params
    ------
    target_peak  : float, default 0.95
    """

    def __init__(self, target_peak: float = 0.95, **kw):
        super().__init__(target_peak=target_peak, **kw)

    def _process(self, x, props):
        target = np.float32(self.p(props, "target_peak", self.params["target_peak"]))
        peak = float(_abs_max_2d(x))
        if peak <= 0:
            return x
        gain = np.float32(target) / np.float32(peak)
        return _scale_2d(x, gain)


@register_processor("global_normalize")
class GlobalNormalize(BaseProcessor):
    """
    Normalize using a peak value provided externally.

    Unlike `GlobalNormalize` in the old postProcess package, this processor
    does not walk the filesystem. The caller computes the scene-wide peak
    (e.g. by evaluating every leaf through the DAG) and passes it via
    `props.custom["global_peak"]`. This keeps the processor pure and
    cacheable.

    Params
    ------
    target_peak  : float, default 0.95
    peak_key     : str,   default "global_peak"
    """

    def __init__(self, target_peak: float = 0.95, peak_key: str = "global_peak", **kw):
        super().__init__(target_peak=target_peak, peak_key=peak_key, **kw)

    def _process(self, x, props):
        target = float(self.p(props, "target_peak", self.params["target__peak"]))
        peak_key = self.p(props, "peak_key", self.params["peak_key"])

        global_peak = None
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            global_peak = props.custom.get(peak_key)

        if global_peak is None or global_peak <= 0:
            # fall back to per-track peak
            peak = float(_abs_max_2d(x))
        else:
            peak = float(global_peak)

        if peak <= 0:
            return x
        gain = np.float32(target) / np.float32(peak)
        return _scale_2d(x, gain)

