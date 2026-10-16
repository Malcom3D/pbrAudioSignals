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
def _decode_2d(ambi: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    ambi   : (C, S)  float32
    coeffs : (C,)    float32 per-channel gains for one output direction
    returns: (1, S)  float32
    """
    C, S = ambi.shape
    K = coeffs.shape[0]
    if K > C:
        K = C
    out = np.zeros((1, S), dtype=np.float32)
    for i in nb.prange(S):
        s = np.float32(0.0)
        for c in range(K):
            s += coeffs[c] * ambi[c, i]
        out[0, i] = s
    return out


def _spherical_harmonics(az: float, el: float, order: int) -> np.ndarray:
    """Real-valued (ACN/SN3D) spherical harmonic decode vector."""
    theta = az
    phi = np.pi / 2.0 - el
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    from math import factorial
    out = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if m == 0:
                norm = np.sqrt((2 * n + 1) / (4 * np.pi))
            else:
                norm = np.sqrt(
                    (2 * n + 1) / (2 * np.pi)
                    * factorial(n - abs(m)) / factorial(n + abs(m))
                )
            # P (associated Legendre) -- explicit for n<=3, fallback 0
            if n == 0:
                P = 1.0
            elif n == 1:
                P = sin_phi if m == -1 else (cos_phi if m == 0 else sin_phi)
            elif n == 2:
                if m == -2:   P = 3 * sin_phi ** 2
                elif m == -1: P = 3 * sin_phi * cos_phi
                elif m == 0:  P = (3 * cos_phi ** 2 - 1) / 2
                elif m == 1:  P = 3 * sin_phi * cos_phi
                else:         P = 3 * sin_phi ** 2
            elif n == 3:
                if m == -3:   P = 15 * sin_phi ** 3
                elif m == -2: P = 15 * sin_phi ** 2 * cos_phi
                elif m == -1: P = sin_phi * (15 * cos_phi ** 2 - 3) / 2
                elif m == 0:  P = cos_phi * (5 * cos_phi ** 2 - 3) / 2
                elif m == 1:  P = sin_phi * (15 * cos_phi ** 2 - 3) / 2
                elif m == 2:  P = 15 * sin_phi ** 2 * cos_phi
                else:         P = 15 * sin_phi ** 3
            else:
                P = 0.0

            if m < 0:
                h = norm * P * np.sin(abs(m) * theta)
            elif m == 0:
                h = norm * P
            else:
                h = norm * P * np.cos(m * theta)
            out.append(h)
    return np.asarray(out, dtype=np.float32)


# standard layouts (azimuth, elevation) in degrees
_DEFAULT_LAYOUTS = {
    "stereo":   [(-30.0, 0.0), (30.0, 0.0)],
    "5.1":      [(-30.0, 0.0), (30.0, 0.0), (0.0, 0.0),
                 (-110.0, 0.0), (110.0, 0.0), (0.0, 0.0)],  # last = LFE
    "7.1":      [(-30.0, 0.0), (30.0, 0.0), (0.0, 0.0),
                 (-110.0, 0.0), (110.0, 0.0),
                 (-135.0, 0.0), (135.0, 0.0), (0.0, 0.0)],  # last = LFE
    "quad":     [(-45.0, 0.0), (45.0, 0.0), (-135.0, 0.0), (135.0, 0.0)],
    "22.2":     [(-45.0, -30.0), (45.0, -30.0), (-135.0, -30.0), (135.0, -30.0),
                 (-30.0, 0.0), (30.0, 0.0), (0.0, 0.0), (-110.0, 0.0), (110.0, 0.0),
                 (-135.0, 0.0), (135.0, 0.0), (-180.0, 0.0),
                 (-45.0, 45.0), (45.0, 45.0), (-90.0, 45.0), (90.0, 45.0),
                 (-135.0, 45.0), (135.0, 45.0), (0.0, 45.0),
                 (0.0, 0.0), (0.0, 0.0),  # LFE 1, LFE 2
                 (0.0, 90.0)],            # VOG
}


@register_processor("ambi_decode")
class AmbisonicDecoder(BaseProcessor):
    """
    Decode an ambisonic buffer (C, S) into a multi-channel layout.

    Unlike the previous postProcess AmbisonicDecoder, this processor:
      * never touches the filesystem,
      * works on the incoming buffer,
      * returns (n_speakers, S).

    Params
    ------
    order  : int
        Ambisonic order. Inferred from C if None.
    layout : str
        One of _DEFAULT_LAYOUTS keys, or "custom".
    speakers : list of (azimuth_deg, elevation_deg)
        Required when layout == "custom".
    normalize : bool, default True
    """

    def __init__(self, order: int = None, layout: str = "stereo",
                 speakers: list = None, normalize: bool = True, **kw):
        super().__init__(order=order, layout=layout,
                         speakers=speakers, normalize=normalize, **kw)

    def _resolve_layout(self, props):
        layout = self.p(props, "layout", self.params["layout"])
        speakers = self.p(props, "speakers", self.params["speakers"])
        if layout == "custom":
            if not speakers:
                raise ValueError("layout='custom' requires non-empty speakers list")
            names = [f"spk_{i}" for i in range(len(speakers))]
            return names, speakers
        if layout not in _DEFAULT_LAYOUTS:
            raise KeyError(f"Unknown layout '{layout}'. "
                           f"Known: {sorted(_DEFAULT_LAYOUTS)}")
        sp = _DEFAULT_LAYOUTS[layout]
        names = [f"spk_{i}" for i in range(len(sp))]
        return names, sp



    def _process(self, x, props):
        C, S = x.shape

        order = self.p(props, "order", self.params["order"])
        if order is None:
            order = int(round(np.sqrt(C))) - 1
        order = int(order)
        expected = (order + 1) ** 2
        if C < expected:
            # zero-pad to expected channels
            padded = np.zeros((expected, S), dtype=np.float32)
            padded[:C] = x
            x = padded
            C = expected

        names, speakers = self._resolve_layout(props)

        # Build the (n_speakers, C) decode matrix.
        decode = np.zeros((len(speakers), C), dtype=np.float32)
        for i, (az_deg, el_deg) in enumerate(speakers):
            coeffs = _spherical_harmonics(np.deg2rad(az_deg),
                                          np.deg2rad(el_deg), order)
            if coeffs.shape[0] > C:
                coeffs = coeffs[:C]
            decode[i, :coeffs.shape[0]] = coeffs

        # Apply decode row-by-row.
        out = np.empty((len(speakers), S), dtype=np.float32)
        for i in range(len(speakers)):
            row = np.ascontiguousarray(decode[i])
            out[i * 1:(i + 1) * 1] = _decode_2d(x, row)

        if self.p(props, "normalize", self.params["normalize"]):
            peak = np.max(np.abs(out))
            if peak > 0:
                out *= np.float32(0.95) / np.float32(peak)

        # Stash the layout names on the props for downstream consumers.
        if props is not None and isinstance(getattr(props, "custom", None), dict):
            props.custom["speaker_names"] = names
            props.custom["speaker_positions"] = list(speakers)

        return out

