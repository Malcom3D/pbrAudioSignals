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


@nb.njit(cache=True, nogil=True, fastmath=True)
def _tri_smooth_1d(x: np.ndarray, radius: np.int64) -> np.ndarray:
    """
    Cheap triangular-kernel smoother: out[i] = weighted average of
    x[i-r .. i+r] with triangular weights. Used as a proxy for a Gaussian
    with sigma ~ radius / 2.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    if radius <= 0:
        for i in range(n):
            out[i] = x[i]
        return out

    for i in range(n):
        acc = np.float32(0.0)
        wsum = np.float32(0.0)
        lo = i - radius
        hi = i + radius
        if lo < 0:
            lo = 0
        if hi > n - 1:
            hi = n - 1
        for j in range(lo, hi + 1):
            d = abs(j - i)
            w = np.float32(radius + 1 - d)
            acc += w * x[j]
            wsum += w
        out[i] = acc / wsum if wsum > 0 else x[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _adaptive_gauss_2d(x: np.ndarray,
                       force: np.ndarray,
                       sigma_min: np.float32,
                       sigma_max: np.float32,
                       threshold: np.float32) -> np.ndarray:
    """
    Adaptive triangular smoothing.

    For each sample:
      * If force > threshold -> keep raw sample (transient preservation).
      * Else -> interpolate between raw x and a pre-smoothed version,
        using a force-dependent mixing weight.
    """
    C, S = x.shape
    out = np.empty_like(x, dtype=np.float32)

    for ch in nb.prange(C):
        # Pre-smooth with largest plausible radius for this sigma_max
        max_radius = int(sigma_max * np.float32(2.0))
        if max_radius < 1:
            max_radius = 1
        smoothed = _tri_smooth_1d(x[ch], np.int64(max_radius))

        for i in range(S):
            f = force[i]
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0

            if f > threshold:
                out[ch, i] = x[ch, i]
            else:
                # weight -> how much smoothing (1 - force)
                w = np.float32(1.0) - f
                out[ch, i] = x[ch, i] * (np.float32(1.0) - w) + smoothed[i] * w

    return out


@register_processor("adaptive_gaussian")
class AdaptiveGaussianSmoothing(BaseProcessor):
    """
    Gaussian-like adaptive smoothing driven by an external force envelope.

    Params
    ------
    sigma_min       : float, default 0.5
    sigma_max       : float, default 3.0
    force_threshold : float, default 0.1
    """

    def __init__(self,
                 sigma_min: float = 0.5,
                 sigma_max: float = 3.0,
                 force_threshold: float = 0.1, **kw):
        super().__init__(sigma_min=sigma_min,
                         sigma_max=sigma_max,
                         force_threshold=force_threshold, **kw)

    def _process(self, x, props):
        smin = np.float32(self.p(props, "sigma_min", self.params["sigma_min"]))
        smax = np.float32(self.p(props, "sigma_max", self.params["sigma_max"]))
        thr = np.float32(self.p(props, "force_threshold",
                                self.params["force_threshold"]))
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x
        mx = force.max() if force.max() > 0 else 1.0
        force = (force / mx).astype(np.float32, copy=False)
        return _adaptive_gauss_2d(x, force, smin, smax, thr)

