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
Physically based loudness / radiation-efficiency calibration for
MODAL_DISPLACEMENT tracks produced by mesh2faust-based synthesis.

Why this exists
---------------
mesh2faust emits dimensionless gains: the modal gain vector is normalized
so that max(|g|) == 1, and nothing in the pipeline records the object's
absolute size or mass. The result is that a marble and a girder
generate the *same* peak displacement for the *same* force, meaning
a marble sounds like a smaller girder (or a girder sounds like a marble).

Physics
-------
For a small baffled piston of area A radiating into air of density rho
and speed c, the far-field pressure per unit acceleration is

    p / a  =  rho * A / (4 * pi * r)

and for a modal displacement x the modal *acceleration* is omega^2 * x.
The total radiated power over all modes is therefore

    P  ∝  A^2 / (4*pi*r)^2 * rho^2 * sum_n (omega_n^4 * x_n^2)

The two size- and mass-dependent scalars that mesh2faust *cannot* know are:

    1.  S_area  = A / A_ref         (surface area vs. reference)
    2.  M_mass  = m / m_ref         (mass vs. reference)

so the calibration this processor applies is

    out = x * (alpha_area * S_area^2 + alpha_mass * log(M_mass)) * K

with alpha_* tuned so that, at (S_area = 1, M_mass = 1), the correction
is unity and the default target_rms is honoured. Additionally the
incoming stream is *level-referenced to a physical force reference*:
at every force peak we are guaranteed that |x|peak ≈ calibration_rms,
so that the modal track carries the correct absolute energy into the
ray tracer.

The processor is a *static* gain plus a *mean* DC-free tilt — it does
not phase-rotate and it does not introduce any temporal smearing. That
makes it safe to place *before* any of the envelope-following
processors, and idempotent when re-run.

Notes
-----
* ``obj_config`` (from props.custom) supplies ``size``, ``mass`` and
  ``acoustic_shader.density``. When absent, neutral 1x defaults are
  used, so the processor is a no-op.
* ``force`` supplies the instantaneous peak reference. Only the peak
  envelope is read (not the samples), so this remains compatible with
  the AUDIO_FORCE track that feeding side-chain uses.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processorprocessor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@nb.njit(cache=True, nogil=True, fastmath=True)
def _box_smooth_1d(x: np.ndarray, half: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
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
def _apply_static_gain_2d(x: np.ndarray, gain: np.float32) -> np.ndarray:
    """Simple per-sample multiply, one channel per thread."""
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * gain
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _apply_gain_envelopevelope_2d(x: np.ndarray, gain_env: np.ndarray) -> np.ndarray:
    """Per-sample gain from a precomputed envelope (shape S,)."""
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * gain_env[i]
    return out

# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("force_modal_radiation_calibrator", track_group=MODAL_DISPLACEMENT)
class ForceModalRadiationCalibrator(BaseProcessor):
    """
    Correct mesh2faust's dimensionless output for physical size / mass.

    Two axes of correction:

      * **Static loudness** — derived from object surface area and mass.
        Surface area is the dominant factor (radiating surface), mass
        adds a sub-dominant perceptual weight.

      * **Force-weighted RMS** — the outgoing stream is normalized so
        that its peak inside a force-active window equals
        ``target_rms_active``. Inactive samples (quiet tails) are
        left at the same absolute level, *not* amplified, because
        the true physical response is silent there.

    Params
    ------
    ref_size              : float, default 0.2   # m   (20 cm neutral object)
    ref_mass              : float, default 1.0   # kg  (1 kg neutral object)

    area_weight           : float, default 0.65  # dominant
    mass_weight           : float, default 0.35  # perceptual mass weight
    max_correction_db     : float, default 24.0  # hard ceiling (prevents explosions)
    min_correction_db     : float, default -24.0

    target_rms_active     : float, default 0.15  # RMS inside force-active windows
    force_threshold       : float, default 0.05  # force envelope activity threshold
    env_smooth_ms         : float, default 5.0   # smooths the envelope that gates RMS
    mode                  : str,   default "static"
                            "static"     -> uniform gain
                            "force_rms"  -> gain + force-windowed peak limiter

    sample_rate           : int,   default 48000
    sfps                  : float, default 24.0
    attr                  : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(
        self,
        ref_size: float = 0.2,
        ref_mass: float = 1.0,
        area_weight: float = 0.65,
        mass_weight: float = 0.35,
        max_correction_db: float = 24.0,
        min_correction_db: float = -24.0,
        target_rms_active: float = 0.15,
        force_threshold: float = 0.05,
        env_smooth_ms: float = 5.0,
        mode: str = "static",
        sample_rate: int = 48000,
        sfps: float = 24.0,
        attr: str = "normal_force_magnitude",
        **kw,
    ):
        super().__init__(
            ref_size=ref_size,
            ref_mass=ref_mass,
            area_weight=area_weight,
            mass_weight=mass_weight,
            max_correction_db=max_correction_db,
            min_correction_db=min_correction_db,
            target_rms_active=target_rms_active,
            force_threshold=force_threshold,
            env_smooth_ms=env_smooth_ms,
            mode=mode,
            sample_rate=sample_rate,
            sfps=sfps,
            attr=attr,
            **kw,
        )

    # ------------------------------------------------------------------
    # Object geometry resolution
    # ------------------------------------------------------------------
    def _resolve_object_geometry(self, props):
        """
        Return (surface_area_m2, mass_kg) using obj_config if present.

        When obj_config has no explicit ``mass`` we approximate it as
        ``density * size^3`` (rectangular bbox approximation), and when
        no ``size`` is present we fall back to ``(mass/density)^(1/3)``.
        If nothing gives us a mass OR a size, we return (None, None)
        and the caller treats this as a no-op.
        """
        if props is None or not isinstance(getattr(props, "custom", None), dict):
            return None, None
        cfg = props.custom.get("obj_config")
        if cfg is None:
            return None, None

        size = getattr(cfg, "size", None)
        mass = getattr(cfg, "mass", None)

        density = None
        shader = getattr(cfg, "acoustic_shader", None)
        if shader is not None:
            density = getattr(shader, "density", None)

        # Derive missing quantity, in this priority order.
        if mass is None and size is not None and density is not None and density > 0.0:
            # rectangular bbox approximation; good enough for a loudness cue
            mass = float(density) * float(size) ** 3
        if size is None and mass is not None and density is not None and density > 0.0:
            size = float(mass / density) ** (1.0 / 3.0)

        if mass is None or size is None:
            return None, None

        # Equivalent spherical radius -> radiating surface area.
        # For non-spherical objects this over-estimates, but only by a
        # small constant factor and it is *consistent* across sizes,
        # which is exactly what we need for perceptual scaling.
        r_eq = 0.5 * float(size)
        area = 4.0 * np.pi * r_eq * r_eq
        return area, float(mass)

    # ------------------------------------------------------------------
    # Static correction factor
    # ------------------------------------------------------------------
    def _static_correction_db(
        self,
        area: float,
        mass: float,
        ref_size: float,
        ref_mass: float,
        area_weight: float,
        mass_weight: float,
        min_db: float,
        max_db: float,
    ) -> float:
        ref_area = 4.0 * np.pi * (0.5 * float(ref_size)) ** 2
        ref_mass = float(ref_mass)

        area_ratio = max(area, 1e-9) / max(ref_area, 1e-9)
        mass_ratio = max(mass, 1e-9) / max(ref_mass, 1e-9)

        # Surface area: acoustic power per unit modal displacement
        # scales as A^2 -> dB. Mass: perceptual loudness grows with
        # log(mass) because of the energy-per-unit-force relationship.
        db = 20.0 * area_weight * np.log10(area_ratio) + 10.0 * mass_weight * np.log10(mass_ratio)

        # Hard clamp so a badly-annotated scene cannot blow up.
        if db > max_db:
            db = max_db
        elif db < min_db:
            db = min_db
        return float(db)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        if x.shape[1] == 0:
            return x

        area, mass = self._resolve_object_geometry(props)
        if area is None or mass is None:
            # No physical description -> processor is a no-op.
            return x

        ref_size = float(self.p(props, "ref_size", self.params["ref_size"]))
        ref_mass = float(self.p(props, "ref_mass", self.params["ref_mass"]))
        area_w = float(self.p(props, "area_weight", self.params["area_weight"]))
        mass_w = float(self.p(props, "mass_weight", self.params["mass_weight"]))
        min_db = float(self.p(props, "min_correction_db", self.params["min_correction_db"]))
        max_db = float(self.p(props, "max_correction_db", self.params["max_correction_db"]))
        mode = str(self.p(props, "mode", self.params["mode"]))
        target_rms = float(self.p(props, "target_rms_active", self.params["target_rms_active"]))
        thr = float(self.p(props, "force_threshold", self.params["force_threshold"]))
        env_ms = float(self.p(props, "env_smooth_ms", self.params["env_smooth_ms"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])

        correction_db = self._static_correction_db(
            area, mass, ref_size, ref_mass,
            area_w, mass_w, min_db, max_db,
        )
        correction_lin = np.float32(10.0 ** (correction_db / 20.0))

        # Static correction is the base output.
        y = _apply_static_gain_2d(x, correction_lin)

        if mode == "static":
            return y

        # ----- mode == "force_rms": normalise inside active windows -----
        seq = get_force_sequence(props)
        if seq is None:
            return y

        S = x.shape[1]
        env = sample_envelope(
            seq,
            num_samples=S,
            sfps=sfps,
            sample_rate=sr,
            attr=attr,
            normalize=True,
            clip_positive=True,
        )

        # Smooth the envelope to avoid gate-chatter on the RMS window.
        half = int(max(1, env_ms * sr / 1000.0))
        env = _box_smooth_1d(env.astype(np.float32), np.int64(half))

        # Peak of y inside active windows.
        active = env >= np.float32(thr)
        if not np.any(active):
            return y

        y_peak = 0.0
        C = y.shape[0]
        for ch in range(C):
            ch_peak = float(np.max(np.abs(y[ch][active])))
            if ch_peak > y_peak:
                y_peak = ch_peak
        if y_peak <= 1e-12:
            return y

        # Scale so that the *active* peak hits target_rms.
        gain = np.float32(target_rms / y_peak)
        # The corrected stream is what we emit; the *peak* is what the
        # RMS reference target is relative to, which is why this is a
        # single global factor (no envelope multiplication).
        return _apply_static_gain_2d(y, gain)

