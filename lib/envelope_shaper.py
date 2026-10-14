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
def _envelope_shape_2d(x: np.ndarray, env: np.ndarray,
                       depth: np.float32) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            # gain = (1 - depth) + depth * env[i]
            out[ch, i] = x[ch, i] * (np.float32(1.0) - depth
                                     + depth * env[i])
    return out


@register_processor("envelope_shaper")
class EnvelopeShaper(BaseProcessor):
    """
    Amplitude shaping driven by an external force envelope.

    Params
    ------
    depth : float, default 0.5   # 0 = bypass, 1 = fully driven by envelope
    """

    def __init__(self, depth: float = 0.5, **kw):
        super().__init__(depth=depth, **kw)

    def _process(self, x, props):
        depth = np.float32(self.p(props, "depth", self.params["depth"]))
        S = x.shape[1]
        env = self.side(props, "force", expected_len=S)
        if env is None:
            return x
        mx = env.max() if env.max() > 0 else 1.0
        env = (env / mx).astype(np.float32, copy=False)
        return _envelope_shape_2d(x, env, depth)

