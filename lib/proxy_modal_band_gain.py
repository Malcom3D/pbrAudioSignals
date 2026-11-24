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

# pbrAudioSignals/lib/proxy_modal_band_gain.py
"""
Per-modal-band gain for ProxySynth content.

In `ProxySynth._convolve_with_irs`, every input signal is convolved
with every IR row (one row per modal frequency band), then summed.
That is mathematically equivalent to applying a *peaking filter bank*
whose centre frequencies are the modal frequencies and whose gains
are the individual IR row energies, then adding them.

This processor reconstructs that operation as a cascade of stable
2nd-order peaking filters. Because the modal frequencies are
*dynamic* (they vary with `size_scale`), we derive them from
`obj_config` using the same formula as `ProxyIRTable.get_modes`:

    frequencies ≈ shape_factor * c_shear / (2*pi * r_eq)

where r_eq is a spherical-equivalent radius of the bounding box.

Unlike `force_modal_equalizer` (which uses shelves), this processor
uses *band* gain — it will not touch frequencies outside the modal
bands, which is exactly what we want: we are shaping the *resonances*
that ProxySynth would have produced.

Params
-----
num_modes          : int,   default 5     # proxy_type + 4 for ProxySynth
ref_size           : float, default 0.2
band_q             : float, default 6.0
max_band_gain_db   : float, default 3.0
mode_gain_flatness : float, default 0.7  # 0=all bands equal, 1=1/n roll-off
sample_rate        : int,   default 48000
sfps               : float, default 24.0
attr               : str,   default "normal_force_magnitude"
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _apply_biquad_bank_2d(
    x: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    a1: np.ndarray, a2: np.ndarray,
) -> np.ndarray:
    """
    Apply a cascade of 2nd-order peaking filters to every channel.

    Each array has length `n_bands`; the coefficients have already been
    normalized by a0. The cascade is evaluated in a single pass over
    the signal to minimise cache misses.
    """
    C, S = x x.shape
    n_bands = b0.shape[0]
    out = np.empty_like(x)

    for ch in nb.prange(C):
        # Intermediate buffer reused for each band.
        y = x[ch].copy()

        for band_idx in range(n_bands):
            B0 = b0[band_idx]
            B1 = b1[band_idx]
            B2 = b2[band_idx]
            A1 = a1[band_idx]
            A2 = a2[band_idx]

            x1 = np.float32(0.0)
            x2 = np.float32(0.0)
            y1 = np.float32(0.0)
            y2 = np.float32(0.0)

            for i in range(S):
                xi = y[i]
                yi = B0 * xi + B1 * x1 + B2 * x2 - A1 * y1 - A2 * y2
                y[i] = yi
                x2 = x1
                x1 = xi
                y2 = y1
                y1 = yi

        out[ch] = y
    return out

def _design_peaking_biquad(fc: float, sample_rate: int, gain_db: float, q: float):
    """
    RBJ audio-EQ-cookbook peaking filter.

    Returns (b0, b1, b2, a1, a2) with a0 normalised to 1.
    When gain_db == 0 the filter is the identity.
    """
    if abs(gain_db) < 1e-6:
        return (np.float32(1.0), np.float32(0.0), np.float32(0.0),
                np.float32(0.0), np.float32(0.0))

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    a0 = 1.0 + alpha / A
    b0 = (1.0 + alpha * A) / a0
    b1 = (-2.0 * cos_w0) / a0
    b2 = (1.0 - alpha * A) / a0
    a1 = (-2.0 * cos_w0) / a0
    a2 = (1.0 - alpha / A) / a0

    return (np.float32(b0), np.float32(b1), np.float32(b2),
            np.float32(a1), np.float32(a2))


@register_processor("proxy_modal_band_gain", track_group=MODAL_DISPLACEMENT)
class ProxyModalBandGain(BaseProcessor):
    """
    ProxySynth-specific modal-band peaking bank.

    Reconstructs the per-mode summation that `ProxySynth` performs
    (convolve with every IR row, then sum) as a cascade of 2nd-order
    peaking filters centred on the modal frequencies ProxySynth
    would generate for this object's size.

    This is idempotent: when both the size and force channels are
    missing, the processor returns its input unchanged (i.e. it is
    a no-op). It is safe to place *after* `proxy_modal_size_tilt`
    and `proxy_modal_force_dynamic_eq` — it operates on different
    frequency bands and does not fight those shelves.

    Params
    ------
    num_modes          : int,   default 5        # for proxy_types 0/1/2 = proxy_type + 4
    ref_size           : float, default 0.2
    band_q             : float, default 6.0
    max_band_gain_db   : float, default 3.0
    mode_gain_flatness : float, default 0.7       # 0 = equal, 1 = 1/n roll-off
    shape_factor       : float, default 1.0       # multiplier on modal frequencies
    sample_rate        : int,   default 48000
    sfps               : float, default 24.0
    attr               : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self,
                 num_modes: int = 5,
                 ref_size: float = 0.2,
                 band_q: float = 6.0,
                 max_band_gain_db: float = 3.0,
                 mode_gain_flatness: float = 0.7,
                 shape_factor: float = 1.0,
                 sample_rate: int = 48000,
                 sfps: float = 24.0,
                 attr: str = "normal_force_magnitude",
                 **kw):
        super().__init__(
            num_modes=num_modes,
            ref_size=ref_size,
            band_q=band_q,
            max_band_gain_db=max_band_gain_db,
            mode_gain_flatness=mode_gain_flatness,
            shape_factor=shape_factor,
            sample_rate=sample_rate,
            sfps=sfps,
            attr=attr,
            **kw,
        )

    # ------------------------------------------------------------------
    def _resolve_geometry(self, props):
        """
        Derive (size, proxy_type, material_properties) from obj_config.

        The exporter for ProxySynth typically writes a `size` attribute
        on ObjectConfig. If it is absent, we try to infer it from
        `mass` and `density`.
        """
        if props is None or not isinstance(getattr(props, "custom", None), dict):
            return None, None, None
        cfg = props.custom.get("obj_config")
        if cfg is None:
            return None, None, None

        size = getattr(cfg, "size", None)
        mass = getattr(cfg, "mass", None)
        proxy_type = getattr(cfg, "proxy_type", False)

        density = None
        shader = getattr(cfg, "acoustic_shader", None)
        young_modulus = None
        poisson_ratio = None
        if shader is not None:
            density = getattr(shader, "density", None)
            young_modulus = getattr(shader, "young_modulus", None)
            poisson_ratio = getattr(shader, "poisson_ratio", None)

        if size is None and mass is not None and density and density > 0:
            size = (float(mass) / float(density)) ** (1.0 / 3.0)

        if size is None:
            return None, None, None

        return float(size), proxy_type, {
            "density": density,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
        }

    # ------------------------------------------------------------------
    def _wave_speeds(self, young_modulus, poisson_ratio, density) -> float:
        """
        Shear wave speed used by ProxyIRTable, with safe defaults.
        """
        E = float(young_modulus) if young_modulus else 1e9
        nu = float(poisson_ratio) if poisson_ratio is not None else 0.3
        rho = float(density) if density else 1000.0

        # Avoid the singular nu = 0.5 case for shear.
        if nu >= 0.5:
            nu = 0.45
        if nu <= -0.5:
            nu = -0.45

        c_shear = np.sqrt(E / (2.0 * rho * (1.0 + nu)))
        return float(c_shear) if np.isfinite(c_shear) and c_shear > 0 else 1500.0

    # ------------------------------------------------------------------
    def _modal_frequencies(self, size, proxy_type, material, sf, n_modes):
        """
        Modal frequencies for the proxy object.

        Mirrors `ProxyIRTable._icosahedron_mode_frequencies` /
        `_cube_mode_frequencies` family structure (a simple harmonic
        series scaled by the shear wave speed over the equivalent
        radius). This gives us the correct *shape* of the spectrum
        without needing the full IR table at processor time.
        """
        c_shear = self._wave_speeds(
            material.get("young_modulus"),
            material.get("poisson_ratio"),
            material.get("density"),
        )

        # Equivalent spherical radius from max dimension.
        radius = max(size(size * 0.5, 1e-3)

        # Mode-family factors are distinct from ProxyIRTable because
        # we want a *smoothed* version of a modal-bank response. The
        # typical proxy has 4-9 discrete modes; the frequency ratios
        # 1.0 : 1.6 : 2.3 : 3.1 : 3.9 come from the spherical harmonic
        # n-th root approximation and match the IR table reasonably
        # well.
        ratio_table = np.array([
            1.0, 1.6, 2.3, 3.1, 3.9, 4.8, 5.8, 6.9, 8.1, 9.4,
        ], dtype=np.float64)

        f_fundamental = c_shear * sf / (2.0 * np.pi * radius)

        # Clip to Nyquist range.
        nyq = 0.5 * int(self.params["sample_rate"])
        freqs = f_fundamental * ratio_table[:n_modes]
        freqs = np.clip(freqs, 20.0, nyq * 0.95)

        # Avoid duplicate frequencies that would double-tune the
        # cascade (broad is fine, over-narrow is the pitfall).
        freqs = np.unique(np.round(freqs, 6))
        if freqs.size == 0:
            freqs = np.array([100.0])

        return freqs

    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        C, S = x.shape
        if S <= 1:
            return x

        size, proxy_type, material = self._resolve_geometry(props)
        if size is None:
            return x

        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        n_modes = int(self.p(props, "num_modes", self.params["num_modes"]))
        band_q = float(self.p(props, "band_q", self.params["band_q"]))
        max_gain = float(self.p(props, "max_band_gain_db", self.params["max_band_gain_db"]))
        flatness = float(self.p(props, "mode_gain_flatness", self.params["mode_gain_flatness"]))
        shape_factor = float(self.p(props, "shape_factor", self.params["shape_factor"]))
        attr = self.p(props, "attr", self.params["attr"])

        # Proxy-type-specific mode count: 4 modes for pyramid, etc.
        if proxy_type is not False and isinstance(proxy_type, int):
            n_modes = max(n_modes, proxy_type + 4)

        # Proxy type de-tuning factor: pyramid shape responds with
        # higher modes than cube (shorter path length through material).
        proxy_shape_factor = {0: 1.3, 1: 1.15, 2: 1.0, 3: 0.95, 4: 0.92, 5: 0.90}
        sf = shape_factor * proxy_shape_factor.get(int(proxy_type) if isinstance(proxy_type, int) else 2, 1.0)

        freqs = self._modal_frequencies(size, proxy_type, material, sf, n_modes)
)
        if freqs.size == 0:
            return x

        # Force-gated gains: the bands are only boosted *while the
        # object is actually excited*. This prevents tail hiss from
        # being peaking-boosted into noise.
        env = sample_envelope(
            seq, num_samples=S, sfps=sfps, sample_rate=sr,
            attr=attr, normalize=True, clip_positive=True,
        )

        # Per-band gain: 1/n roll-off controlled by flatness.
        n = freqs.shape[0]
        if n > 1:
            rolloff = np.linspace(1.0, 1.0 / n, n)  # linear from 1 to 1/n
        else:
            rolloff = np.array([1.0])
        rolloff = 1.0 - flatness * (1.0 - rolloff)   # blend toward flat=1
        band_gains = np.clip(rolloff * max_gain, 0.0, max_gain).astype(np.float32)

        # Because force gating would require time-varying biquads (which
        # are expensive in the general case), we *precompute* the
        # coefficient cascade and apply a static bank. Gating is then
        # applied to the *difference* between processed and dry signals.
        b0 = np.zeros(n, dtype=np.float32)
        b1 = np.zeros(n, dtype=np.float32)
        b2 = np.zeros(n, dtype=np.float32)
        a1 = np.zeros(n, dtype=np.float32)
        a2 = np.zeros(n, dtype=np.float32)

        for i in range(n):
            fc = float(freqs[i])
            if fc <= 0.0:
                b0[i] = 1.0
                continue
            bb0, bb1, bb2, aa1, aa2 = _design_peaking_biquad(
                fc, sr, float(band_gains[i]), band_q,
            )
            b0[i], b1[i], b2[i], a1[i], a2[i] = bb0, bb1, bb2, aa1, aa2

        # Apply the cascade.
        y = _apply_biquad_bank_2d(x, b0, b1, b2, a1, a2)

        # Force-gated blend: dry + env * (wet - dry)
        # Silent regions get the dry track byte-for-byte.
        diff = y - x
        # Broadcast env to (C, S) on the fly.
        out = np.empty_like(x)
        for ch in range(C):
            out[ch] = x[ch] + env * diff[ch]

        return out
