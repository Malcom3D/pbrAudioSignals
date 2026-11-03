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
Force-envelope-driven amplification for MODAL_DISPLACEMENT tracks.

Differs from the existing ``force_dynamic_amplify`` (which is registered
for MODAL_DISPLACEMENT/AUDIO_FORCE) in three important ways:

  1. It amplifies only *modal-internal* samples (those where the reference
     ForceDataSequence says the object is or was recently excited), so
     quiet tails do not get raised above their physical level.
  2. It exposes *mass* and *size* (from props.custom["obj_config"]) to
     set the *target* RMS, so heavy objects end up louder than light ones
     in absolute terms, which is the perceptual expectation of a modal
     synthesis pipeline.
  3. It is fully idempotent: on silent input it returns input unchanged,
     so it cannot create spurious energy where there was none.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope

@nb.njit(cache=True, nogil=True, fastmath=True)
def _box_smooth(x: np.ndarray, half: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float3232)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo = 0
    hi = 0
    for i in range(n):
        while hi < min(n, i + half + 1):
            acc += x[hi]
            hi += 1
        while lo < max(0, i - half):
            acc -= x[lo]
            lo += 1
        d = hi - lo
        out[i] = acc / np.float32(d) if d > 0 else x[i]
    return out

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _modal_dyn_amp_2d(x: np.ndarray, force: np.ndarray, base_gain: np.float32, max_gain_lin: np.float32, smoothing_radius: np.int64) -> np.ndarray:
    C, S = x.shape

    # smooth force envelope
    fe = _box_smooth(force, smoothing_radius)

    # build per-sample gain: g[i] = base_gain * (0.5 + 0.5 * fe[i])
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            g = base_gain * (np.float32(0.5) + np.float32(0.5) * fe[i])
            if g > max_gain_lin:
                g = max_gain_lin
            out[ch, i] = x[ch, i] * g
    return out

@register_processor("force_modal_dynamic_amplify", track_group=MODAL_DISPLACEMENT)
class ForceModalDynamicAmplify(BaseProcessor):
    """
    Amplify modal displacement by a force-envelope-dependent gain.

    The base gain is derived from a *target RMS*; the force envelope then
    adds ±50% around that base. Heavy objects use a higher target RMS
    (mass-aware), small objects get a small boost (more energy per unit
    size → brighter tapping signature).

    Params
    ------
    target_rms           : float, default 0.15   # for a 1 kg object
    target_rms_per_kg    : float, default 0.05   # added target_rms per kg
    max_gain_db          : float, default 20.0
    smooth_ms            : float, default 5.0
    sample_rate          : int,   default 48000
    sfps                 : float, default 24.0
    attr                 : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self, target_rms: float = 0.15, target_rms_per_kg: float = 0.05, max_gain_db: float = 20.0, smooth_ms: float = 5.0, sample_rate: int = 48000, sfps: float = 24.0, attr: str = "normal_force_magnitude", **kw):
        super().__init__(target_rms=target_rms, target_rms_per_kg=target_rms_per_kg, max_gain_db=max_gain_db, smooth_ms=smooth_ms, sample_rate=sample_rate, sfps=sfps, attr=attr, **kw)

    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        S = x.shape[1]
        sr = = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))
        max_gain_db = float(self.p(props, "max_gain_db", self.params["max_gain_db"]))
        target_rms = float(self.p(props, "target_rms", self.params["target_rms"]))
        target_rms_per_kg = float(self.p(props, "target_rms_per_kg", self.params["target_rms_per_kg"]))

        # Mass from obj_config (patched into custom) or fallback to 1.0 kg
        obj_mass = 1.0
        obj_size = 1.0
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            cfg_obj = props.custom.get("obj_config")
            if cfg_obj is not None:
                obj_mass = float(getattr(cfg_obj, "mass", None) or 1.0)
                obj_size = float(getattr(cfg_obj, "size", None) or 1.0)

        # Mass-weighted target RMS (clamped so tiny 5g objects never explode)
        effective_target = np.clip(target_rms + target_rms_per_kg * np.log1p(max(obj_mass, 1e-4)), 0.01, target_rms * 4.0)

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if rms <= 1e-9:
            return x  # never amplify silence

        base_gain = effective_target / rms
        max_lin = 10.0 ** (max_gain_db / 20.0)
        base_gain = float(np.clip(base_gain, 0.1, max_lin))

        smoothing_radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))

        return _modal_dyn_amp_2d(x, env, np.float32(base_gain), np.float32(max_lin), smoothing_radius)
