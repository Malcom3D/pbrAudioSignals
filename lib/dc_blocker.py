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
from ..core.groups import GENERIC, AUDIO_FORCE, AUDIO_SIGNAL


@nb.njit(cache=True, nogil=True, fastmath=True)
def _dc_block_1d(x: np.ndarray, alpha: np.float32) -> np.ndarray:
    """y[n] = x[n] - x[n-1] + alpha * y[n-1]."""
    n = x.shape[0]
    y = np.empty(n, dtype=np.float32)
    if n == 0:
        return y
    y[0] = x[0]
    for i in range(1, n):
        y[i] = x[i] - x[i - 1] + alpha * y[i - 1]
    return y


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _dc_block_2d(x: np.ndarray, alpha: np.float32) -> np.ndarray:
    C, S = x.shape
    y = np.empty_like(x)
    for ch in nb.prange(C):
        y[ch] = _dc_block_1d(x[ch], alpha)
    return y

@register_processor("adaptive_noise_gate", track_group=AUDIO_SIGNAL, also_valid_for={ACOUSTIC_PRESSURE, GENERIC})
class DCBlocker(BaseProcessor):
    """
    DC-offset removal via a one-pole high-pass filter.

    Params
    ------
    alpha : float, default 0.999
        Pole location. Higher values -> lower cutoff, slower response.
    """

    def __init__(self, alpha: float = 0.999, **kw):
        super().__init__(alpha=alpha, **kw)
        self._alpha = np.float32(alpha)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        alpha = np.float32(self.p(props, "alpha", self.params.get("alpha", 0.999)))
        return _dc_block_2d(x, alpha)

