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
def _smooth_env_1d(env: np.ndarray, attack: np.int64, release: np.int64) -> np.ndarray:
    n = env.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    out[0] = env[0]
    a_inv = np.float32(1.0) / np.float32(max(attack, 1))
    r_inv = np.float32(1.0) / np.float32(max(release, 1))
    for i in range(1, n):
        if env[i] > out[i - 1]:
            alpha = a_inv
        else:
            alpha = r_inv
        out[i] = (np.float32(1.0) - alpha) * out[i - 1] + alpha * env[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def _hold_1d(env: np.ndarray, hold: np.int64) -> np.ndarray:
    n = env.shape[0]
    out = np.empty_like(env)
    for i in range(n):
        lo = i - hold if i - hold > 0 else 0
        m = env[i]
        for j in range(lo, i):
            if env[j] > m:
                m = env[j]
        out[i] = m
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _adaptive_gate_2d(x: np.ndarray,
                      force: np.ndarray,
                      threshold_linear: np.float32,
                      attack: np.int64,
                      release: np.int64,
                      hold: np.int64,
                      force_weight: np.float32) -> np.ndarray:
    """
    For every channel, compute the smoothed |x| envelope, apply an adaptive
    threshold that decreases when force is high, then multiply x by the
    resulting gate.
    """
    C, S = x.shape
    out = np.empty_like(x)

    # Smooth the force envelope once (it is shared across channels)
    fe = np.empty(S, dtype=np.float32)
    for i in range(S):
        fe[i] = abs(force[i])
    fe = _smooth_env_1d(fe, attack, release)
    for i in range(S):
        if fe[i] > 1.0:
            fe[i] = 1.0

    for ch in nb.prange(C):
        env = np.empty(S, dtype=np.float32)
        for i in range(S):
            env[i] = abs(x[ch, i])
        env = _smooth_env_1d(env, attack, release)
        env = _hold_1d(env, hold)
        for i in range(S):
            thr = threshold_linear * (np.float32(1.0)
                                      + force_weight * fe[i])
            g = env[i] / (thr + np.float32(1e-12))
            # soft gate with smoothstep to avoid clicks
            if g <= 0.0:
                g = 0.0
            elif g >= 1.0:
                g = 1.0
            else:
                g = g * g * (np.float32(3.0) - np.float32(2.0) * g)
            out[ch, i] = x[ch, i] * g
    return out


@register_processor("adaptive_noise_gate")
class AdaptiveNoiseGate(BaseProcessor):
    """
    Adaptive noise gate driven by an optional force-envelope side channel
    (`props.custom['force']`).

    Params
    ------
    sample_rate           : int, default 48000
    threshold_db          : float, default -60
    attack_ms             : float, default 2
    release_ms            : float, default 50
    hold_ms               : float, default 10
    force_weight          : float, default 0.5  # 0 = fixed threshold
    """

    def __init__(self, sample_rate: int = 48000,
                 threshold_db: float = -60.0,
                 attack_ms: float = 2.0,
                 release_ms: float = 50.0,
                 hold_ms: float = 10.0,
                 force_weight: float = 0.5, **kw):
        super().__init__(sample_rate=sample_rate,
                         threshold_db=threshold_db,
                         attack_ms=attack_ms,
                         release_ms=release_ms,
                         hold_ms=hold_ms,
                         force_weight=force_weight, **kw)

    def _process(self, x, props):
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        thr_db = float(self.p(props, "threshold_db", self.params["threshold_db"]))
        att = int(self.p(props, "attack_ms", self.params["attack_ms"]) * sr / 1000)
        rel = int(self.p(props, "release_ms", self.params["release_ms"]) * sr / 1000)
        hold = int(self.p(props, "hold_ms", self.params["hold_ms"]) * sr / 1000)
        fw = np.float32(self.p(props, "force_weight", self.params["force_weight"]))

        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            force = np.zeros(S, dtype=np.float32)

        thr_lin = np.float32(10.0 ** (thr_db / 20.0))
        return _adaptive_gate_2d(x, force, thr_lin, np.int64(att), np.int64(rel), np.int64(hold), fw)

