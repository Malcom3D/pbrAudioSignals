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
Modal displacement -> acoustic pressure conversion.

Given modal displacement x(t) for a solid object's surface, produce the
far-field acoustic pressure at a listener position in the surrounding
medium.

Physics
-------
For a small baffled piston of area A vibrating with displacement x(t),
radiating into a medium of density rho and speed of sound c, the
far-field pressure at distance r is

    p(r, t) ≈ (rho * A) / (4 * pi * r)  *  a(t - r/c)

where a(t) = d²x/dt² is the surface acceleration. For a non-infinite
source, radiation efficiency rolls off below ka = 1, i.e. below

    fc = c / (2 * pi * sqrt(A / pi))

We model this roll-off with a 2nd-order high-pass at fc.

The pipeline is therefore:

    pressure = K * HP2(fc)[ d²x/dt² ]

with K = rho * A / (4 * pi * r).

Notes
-----
* Differentiation is done via a 2nd-difference (double 1st-order diff),
  then integrated back through a 2nd-order IIR HPF, so the output is a
  proper pressure signal and not a double-differentiated ringing trace.
* ``obj_config`` (if provided in props.custom) supplies the object's
  size and mass; otherwise defaults are used.
* This processor never adds or removes DC after conversion - a static
   object produces zero sound by construction, which is physically
  correct.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT

# ---------------------------------------------------------------------------
# JIT kernels
# ---------------------------------------------------------------------------

@nb.njit(cache=True, nogil=True, fastmath=True)
def _hp2_1d(x: np.ndarray, b0: np.float32, b1: np.float32, b2: np.float32, a1: np.float32, a2: np.float32) -> np.ndarray:
    """
    Direct-Form-I 2nd-order high-pass filter (biquad).

    b = [b0, b1, b2], a = [1, a1, a2] -- already normalised by a0.
    A single-pass IIR keeps DC and very-low-freq removed.
    """
    n = x.shape[0]
    y = np.empty(n, dtype=np.float32)
    if n == 0:
        return y

    x1 = np.float32(0.0)
    x2 = np.float32(0.0)
    y1 = np.float32(0.0)
    y2 = np.float32(0.0)

    for i in range(n):
        xi = x[i]
        yi = b00 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2 = x1
        x1 = xi
        y2 = y1
        y1 = yi

    return y

@nb.njit(cache=True, nogil=True, fastmath=True)
def _double_difference_1d(x: np.ndarray) -> np.ndarray:
    """
    Compute the second-order backward difference difference:

        d[i] = x[i] - 2*x[i-1] + x[i-2]

    Initial edges are zero-padded, which is physically correct: nothing
    happened before the buffer began.
    """
    n = x.shape[0]
    d = np.empty(n, dtype=np.float32)
    if n == 0:
        return d
    d[0] = np.float32(0.0)
    if n > 1:
        d[1] = x[1] - np.float32(2.0) * x[0]
    for i in range(2, n):
        d[i] = x[i] - np.float32(2.0) * x[i - 1] + x[i - 2]
    return d


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _modal_to_pressure_2d(x: np.ndarray, b0: np.float32, b1: np.float32, b2: np.float32, a1: np.float32, a2: np.float32) -> np.ndarray:
    """
    For each channel:
        d   = d²x/dt²
        hp  = HP2(fc)[d]
        p   = K * hp

    K is folded into the biquad coefficients (b*) by the caller.
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        d = _double_difference_1d(x[ch])
        out[ch] = _hp2_1d(d, b0, b1, b2, a1, a2)
    return out

# ---------------------------------------------------------------------------
# Biquad design (RBJ cookbook, HPF)
# ---------------------------------------------------------------------------

def _design_hp2(fc: float, sample_rate: int) -> tuple:
    """
    Design a 2nd-order Butterworth HPF at ``fc`` Hz.

    Returns (b0, b1, b2, a1, a2) with a0 normalised to 1.
    Uses the RBJ audio-EQ-cookbook formula.
    """
    fc = float(max(1.0, min(fc, sample_rate * 0.49)))
    w0 = 2.0 * np.pi * fc / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    # Butterworth Q
    q = 1.0 / np.sqrt(2.0)
    alpha = sin_w0 / (2.0 * q)

    a0 = 1.0 + alpha
    b0 = ((1.0 + cos_w0) / 2.0) / a0
    b1 = (-(1.0 + cos_w0)) / a0
    b2 = ((1.0 + cos_w0) / 2.0) / a0
    a1 = (-2.0 * cos_w0) / a0
    a2 = (1.0 - alpha) / a0

    return (np.float32(b0), np.float32(b1), np.float32(b2), np.float32(a1), np.float32(a2))

# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("modal_to_pressure", track_group=MODAL_DISPLACEMENT)
class ModalToPressure(BaseProcessor):
    """
    Convert a modal displacement track to a far-field acoustic pressure
    track for a listener at a given distance in the surrounding medium.

    Params
    ------
    medium_density   : float, 1.225      # kg/m^3  (dry air @ 20 C)
    speed_of_sound   : float, 343.0      # m/s
    listener_distance: float, 1.0        # meters (>=1e-3)
    surface_area     : float, None       # m^2 ; if None, derived from obj_config
    source_size      : float, None       # m ; if None, derived from obj_config
    reference_scale  : float, 1.0        # extra linear trim (dB) — see notes
    gravity          : float, 9.81       # unused, reserved for future variants

    Track group
    -----------
    MODAL_DISPLACEMENT -- the input signal is modal displacement displacement (m).

    Side channels
    -------------
    obj_config
        If present in ``props.custom['obj_config']``, ``size`` and
        ``mass`` (or ``volume``) attributes are read to derive
        ``surface_area`` and ``source_size`` automatically.
    """

    def __init__(self, medium_density: float = 1.225, speed_of_sound: float = 343.0, listener_distance: float = 1.0, surface_area: float = None, source_size: float = None, reference_scale: float = 1.0, sample_rate: int = 48000, **kw):
        super().__init__(medium_density=medium_density, speed_of_sound=speed_of_sound, listener_distance=listener_distance, surface_area=surface_area, source_size=source_size, reference_scale=reference_scale, sample_rate=sample_rate, **kw)
        self._resolved_obj_idx = None
        self._resolved_coeffs = None      # (b0,b1,b2,a1,a2)
        self._resolved_gain = None        # K

    # ------------------------------------------------------------------
    def _resolve_geometry(self, props):
        """
        Resolve (surface_area, source_size, distance) from params or
        obj_config.
        """
        area = self.p(props, "surface_area", self.params["surface_area"])
        size = self.p(props, "source_size", self.params["source_size"])
        distance = float(self.p(props, "listener_distance", self.params["listener_distance"]))

        obj_cfg = None
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            obj_cfg = props.custom.get("obj_config")

        if obj_cfg is not None:
            # size
            if size is None or size <= 0.0:
                cfg_size = getattr(obj_cfg, "size", None)
                if cfg_size is not None and cfg_size > 0.0:
                    size = float(cfg_size)

            # area — derive from volume if possible, else assume spherical
            if area is None or area <= 0.0:
                mass = getattr(obj_cfg, "mass", None)
                volume = getattr(obj_cfg, "volume", None)
                density = None
                shader = getattr(obj_cfg, "acoustic_shader", None)
                if shader is not None:
                    density = getattr(shader, "density", None)

                if volume is not None and volume > 0.0:
                    pass  # have volume
                elif mass is not None and density is not None and density > 0.0:
                    volume = float(mass) / float(density)
                else:
                    volume = None

                if volume is not None and volume > 0.0:
                    # equivalent sphere
                    r_eq = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                    size = size if size is not None else 2.0 * r_eq
                    area = 4.0 * np.pi * r_eq * r_eq

        if area is None or area <= 0.0:
            area = 0.01          # fallback: 10 cm x 10 cm square
        if size is None or size <= 0.0:
            # radius of equivalent disk of that area
            size = 2.0 * np.sqrt(area / np.pi)

        return float(area), float(size), max(1e-3, float(distance))

    # ------------------------------------------------------------------
    def _compute_kernel(self, props):
        """
        Cache biquad + gain tuple. Recompute only if any relevant param
        changed. The signature is (obj_idx, area, size, distance,
        density, c, sr, sr_scale).
        """
        obj_cfg = None
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            obj_cfg = props.custom.get("obj_config")
        obj_idx = getattr(obj_cfg, "idx", None) if obj_cfg is not None else None

        rho = float(self.p(props,, "medium_density", self.params["medium_density"]))
        c = float(self.p(props, "speed_of_sound", self.params["speed_of_sound"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        trim = float(self.p(props, "reference_scale", self.params["reference_scale"]))

        area, size, distance = self._resolve_geometry(props)

        # Radiation corner: ka = 1 with a = equal-radius of the source
        fc = c / (2.0 * np.pi * max(1e-6, size))

        # Coupling gain K  (baffled piston, hemispherical radiation)
        #   p = rho*A/(4 pi r) * a(t)
        K = rho * area / (4.0 * np.pi * distance)
        K *= trim

        coeffs = _design_hp2(fc, sr)

        # The double-difference operator has units of m per sample² (integer
        # not per-second²). We want a physical acceleration in m/s², so we
        # multiply by (sr²) here. This keeps the physical units consistent
        # regardless of the audio sample rate.
        K *= float(sr) * float(sr)

        return (obj_idx, area, size, distance, rho, c, sr, trim), coeffs, K

    # ------------------------------------------------------------------
    def _process(self, x, props):
        if x.shape[1] == 0:
            return x

        sig, coeffs, K = self._compute_kernel(props)

        if sig != self._resolved_obj_idx or coeffs != self._resolved_coeffs or K != self._resolved_gain:
            self._resolved_obj_idx = sig
            self._resolved_coeffs = coeffs
            self._resolved_gain = K

        b0, b1, b2, a1, a2 = coeffs

        # Scale the HPF output by K in the biquad coefficients so the
        # numba kernel stays branch-free and we don't add an extra pass.
        kb0 = np.float32(b0 * K)
        kb1 = np.float32(b1 * K)
        kb2 = np.float32(b2 * K)

        return _modal_to_pressure_2d(x, kb0, kb1, kb2, a1, a2)

