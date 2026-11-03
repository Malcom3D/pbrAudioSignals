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
Force-envelope-driven dynamic-range compression for MODAL_DISPLACEMENT.

Modal synthesis naturally produces very high crest factors at impact
(the response starts at full RMS and then decays). A ray tracer will
over-illuminate the first few milliseconds and under-illuminate the tail,
producing a "click" instead of a "strike". This processor applies a
**time-varying downward compressor** whose threshold tracks the force
envelope, so the modal response becomes perceptually smooth.

Design
------
* Envelope follower with attack / release (coarse, per-sample).
* Threshold from force:     thr[i] = thr_min + (thr_max-thr_min) * (1-force[i])
* Ratio from force:         r[i]   = r_min + (r_max - r_min) * (1-force[i])
* Gain reduction: GR[i] = max(0, (above_thr[i]) * (r[i]-1) / r[i])
* Applied as: y[i] = x[i] * 10^(GR_dB[i]/20)
* Final envelope smoothing prevents zipper noise.

Everything in log-domain (dB), then linear.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope

@nb.njit(cache=True, nogil=True, fastmath=True)
def _env_follower_1d(x: np.ndarray, attack: np.int64, release: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    a_att = np.float32(1.0) / np.float32(max(attack, 1))
    a_rel = np.float32(1.0) / np.float32(max(release, 1))
    out[0] = abs(x[0])
    for i in range(1, n):
        v = abs(x[i])
        if v > out[i - 1]:
            out[i] = (np.float32(1.0) - a_attatt) * out[i - 1] + a_att * v
        else:
            out[i] = (np.float32(1.0) - a_rel) * out[i - 1] + a_rel * v
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _modal_compress_2d(x: np.ndarray, force: np.ndarray, thr_min_db: np.float32, thr_max_db: np.float32, ratio_min: np.float32, ratio_max: np.float32, attack: np.int64, release: np.int64, makeup_lin: np.float32, smooth_radius: np.int64) -> np.ndarray:
    C, S = x.shape
    out = np.empty_like(x)

    # Smooth force envelope once (shared across channels).
    fe = np.empty(S, dtype=np.float32)
    for i in range(S):
        fe[i] = force[i]
    # simple one-pole smoothing
    alpha = np.float32(1.0) / np.float32(max(1, smooth_radius))
    prev = np.float32(0.0)
    for i in range(S):
        prev = (np.float32(1.0) - alpha) * prev + alpha * fe[i]
        fe[i] = prev

    for ch in nb.prange(C):
        # detector envelope
        env = _env_follower_1d(x[ch], attack, release)
        # 20*log10(env + 1e-12)
        for i in range(S):
            # force-dependent threshold and ratio
            th_db = thr_max_db + (thr_min_db - thr_max_db) * fe[i]
            ratio = ratio_max + (ratio_min - ratio_max) * fe[i]
            if ratio < np.float32(1.0):
                ratio = np.float32(1.0)

            level_db = np.float32(20.0) * np.log10(env[i] + np.float32(1e-12))
            above = level_db - th_db
            gain_reduction_db = np.float32(0.0)
            if above > 0.0:
                gain_reduction_db = -above * (ratio - np.float32(1.0)) / ratio

            gain_lin = np.float32(10.0) ** ((gain_reduction_db + np.float32(0.0)) / np.float32(20.0))
            out[ch, i] = x[ch, i] * gain_lin * makeup_lin

    return out


@register_processor("force_modal_compressor", track_group=MODAL_DISPLACEMENT)
class ForceModalCompressor(BaseProcessor):
    """
    Force-envelope-driven downward compressor for modal displacement.

    Params
    ------
    thr_min_db      : float, default -6.0    # threshold at force=1 (impact)
    thr_max_db      : float, default -24.0   # threshold at force=0 (decay)
    ratio_min       : float, default 2.0     # compression ratio at force=1
    ratio_max       : float, default 6.0     # compression ratio at force=0
    attack_ms       : float, default 2.0
    release_ms      : float, default 80.0
    makeup_db       : float, default 0.0     # static makeup gain
    smooth_ms       : float, default 5.0
    sample_rate     : int,   default 48000
    sfps            : float, default 24.0
    attr            : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self, thr_min_db: float = -6.0, thr_max_db: float = -24.0, ratio_min: float = 2.0, ratio_max: float = 6.0, attack_ms: float = 2.0, release_ms: float = 80.0, makeup_db: float = 0.0, smooth_ms: float = 5.0, sample_rate: int = 48000, sfps: float = 24.0, attr: str = "normal_force_magnitude", **kw):
        super().__init__(thr_min_db=thr_min_db, thr_max_db=thr_max_db, ratio_min=ratio_min, ratio_max=ratio_max, attack_ms=attack_ms, release_ms=release_ms, makeup_db=makeup_db, smooth_ms=smooth_ms, sample_rate=sample_rate, sfps=sfps, attr=attr, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])

        thr_min = np.float32(self.p(props, "thr_min_db", self.params["thr_min_db"]))
        thr_max = np.float32(self.p(props, "thr_max_db", self.params["thr_max_db"]))
        r_min = np.float32(self.p(props, "ratio_min", self.params["ratio_min"]))
        r_max = np.float32(self.p(props, "ratio_max", self.params["ratio_max"]))
        att_ms = float(self.p(props, "attack_ms", self.params["attack_ms"]))
        rel_ms = float(self.p(props, "release_ms", self.params["release_ms"]))
        mk_db = float(self.p(props, "makeup_db", self.params["makeup_db"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        attack = np.int64(max(1, int(att_ms * sr / 1000.0)))
        release = np.int64(max(1, int(rel_ms * sr / 1000.0)))
        smooth_radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))
        makeup_lin = np.float32(10.0 ** (mk_db / 20.0))

        return _modal_compress_2d(x, env, thr_min, thr_max, r_min, r_max, attack, release, makeup_lin, smooth_radius)
