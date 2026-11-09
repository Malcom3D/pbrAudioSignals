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
Contact-type-aware spectral tilt for MODAL_DISPLACEMENT tracks.

The physical reality mesh2faust cannot represent
-----------------------------------------------
mesh2faust returns *one* modal model per material/geometry pair. It
has no concept of *how* the object is being touched. But real
perception distinguishes immediately:

  * IMPACT on a hard object   -> broadband impulse, low-mid punch,
                                 very short radiation time
  * ROLLING over a rough surface -> narrow-band low-frequency hiss,
                                 soft high-frequency tail
  * SLIDING                   -> continuous mid-band tone, distinctive
                                 "shhhh" texture
  * SCRAPING                  -> irregular bursts, high-frequency
                                 dominated short clusters
  * STATIC                    -> near-silence after the initial settle
  * MIXED (rolling+sliding)   -> between ROLLING and SLIDING

Solution
--------
Track the contact type frame-by-frame from ForceDataSequence and
apply a *slow* (few-ms) bipolar tilt around a neutral (0 dB) reference.
The tilt is implemented as two one-pole shelving filters, the same
machinery as ForceModalEqualizer but the *source* of the tilt is
`contact_type`, not object geometry. This is deliberately a different
axis from the equalizer:

    ForceModalEqualizer        -> object's *size / mass* (static-ish)
    ForceModalContactTilt      -> contact *behaviour* (dynamic)

They are designed to compose, in either order, without fighting each
other: contact tilt contributes ±tilt_db around the equalizer's
output; the equalizer contributes a size-pivot around the contact
tilt's output. Neither has a net DC component.

Additionally, IMPACT contact gets a short **transient sharpen** — a
very gentle first-order difference blended back in only during the
first ~3 ms of an impact — to restore the "knock" that gets lost
when the modal-synth ring-up smears across many samples. This is
applied *after* the shelving so its time constant is expressed in
real milliseconds, not in mode-frequency cycles.

Params
------
tilt_db_impact        : float, default (lo=+2.0, hi=+3.0)
tilt_db_rolling       : float, default (lo=+1.0, hi=-4.0)
tilt_db_sliding       : float, default (lo=-1.0, hi=+2.0)
tilt_db_scraping      : float, default (lo=+0.5, hi=+3.5)
                      # The tuple is (low_shelf_dB, high_shelf_dB)

low_shelf_hz          : float, default 250.0
high_shelf_hz         : float, default 3000.0

impulse_sharpen_ms    : float, default 3.0    # IMPACT only
impulse_sharpen_depth : float, default 0.20   # blend amount, 0 = off

smooth_ms             : float, default 8.0    # shelf-gain smoothing

sample_rate           : int,   default 48000
sfps                  : float, default 24.0
attr                  : str,   default "normal_force_magnitude"
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types

# ContactType enum values, kept literal so the kernel is self-contained.
_CT_NO      = 0
_CT_IMPACT  = 1
_CT_SCRAPE  = 2
_CT_SLIDE   = 3
_CT_ROLLING = 4
_CT_MIXED   = 5
_CT_STATIC  = 6

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@nb.njit(cache=True, nogil=True, fastmath=True)
def _one_pole_lp_1d(x: np.ndarray, fc_norm: np.float32) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    a = np.float32(1.0) - np.exp(
        np.float32(-2.0) * np.float32(np.pi) * fc_norm
    )
    if a < np.float32(1e-6):
        a = np.float32(1e-6)
    if a > np.float32(1.0):
        a = np.float32(1.0)
    y = np.float32(0.0)
    for i in range(n):
        y = y + a * (x[i] - y)
        out[i] = y
    return out


@nb.njit(cache=True, nogil=True, fastmath=True)
def _smooth_1d(x: np.ndarray, radius: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo = 0
    hi = 0
    for i in range(n):
        while hi < min(n, i + radius + 1):
            acc += x[hi]
            hi += 1
        while lo < max(0, i - radius):
            acc -= x[lo]
            lo += 1
        d = hi - lo
        out[i] = acc / np.float32(d) if d > 0 else x[i]
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _shelf_tilt_2d(x: np.ndarray,
                   low_db: np.ndarray,
                   high_db: np.ndarray,
                   fc_low_norm: np.float32,
                   fc_high_norm: np.float32) -> np.ndarray:
    """
    Two-band tilt around a neutral mid-band reference.

    low-band  = one-pole LP at fc_low
    high-band = input - one-pole LP at fc_high
    mid-band  = input - low - high   (neutral, untouched)

    output    = mid + low * 10^(low_db/20) + high * 10^(high_db/20)
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        low = _one_pole_lp_1d(x[ch], fc_low_norm)
        high_lp = _one_pole_lp_1d(x[ch], fc_high_norm)
        for i in range(S):
            mid = x[ch, i] - high_lp[i]
            high = high_lp[i] - low[i]
            gl = np.float32(10.0) ** (low_db[i] / np.float32(20.0))
            gh = np.float32(10.0) ** (high_db[i] / np.float32(20.0))
            out[ch, i] = low[i] * gl + mid + high * gh
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _impulse_sharpen_2d(x: np.ndarray,
                        impact_mask: np.ndarray,
                        depth: np.float32) -> np.ndarray:
    """
    Blend a first-order difference into x during IMPACT windows.

    Only the *onset* of the impulse is sharpened (mask == 1), so the
    long modal tail is untouched. y = x + depth * (x - x[n-1]) for
    masked samples; zero delay compensation because it is < 1 sample.
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        prev = np.float32(0.0)
        for i in range(S):
            cur = x[ch, i]
            if impact_mask[i] != 0:
                out[ch, i] = cur + depth * (cur - prev)
            else:
                out[ch, i] = cur
            prev = cur
    return out

# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

@register_processor("force_modal_contact_tilt", track_group=MODAL_DISPLACEMENT)
class ForceModalContactTilt(BaseProcessor):
    """
    Contact-type-aware low/high shelf tilt + IMPACT transient sharpen.

    Per-sample control flow
    -----------------------
    1. Read contact_type from ForceDataSequence at sample rate.
    2. Read force envelope.
    3. For each sample, blend the contact-type's (low_db, high_db)
       target weighted by the force envelope. Where force is 0 we
       sit at 0 dB (no-op) so quiet tails are never touched.
    4. Smooth low_db / high_db by `smooth_ms` before applying.
    5. Apply the shelf tilt.
    6. On the first `impulse_sharpen_ms` of each IMPACT event,
       blend in a tiny first-difference to restore attack.

    Notes
    -----
    * Object config is *not* read here. Contact tilt is a pure
      behaviour modifier; size/mass compensation lives in
      ``ForceModalRadiationCalibrator``.
    """

    required_side = {"force"}

    def __init__(
        self,
        tilt_db_impact: tuple = (2.0, 3.0),
        tilt_db_rolling: tuple = (1.0, -4.0),
        tilt_db_sliding: tuple = (-1.0, 2.0),
        tilt_db_scraping: tuple = (0.5, 3.5),
        tilt_db_mixed: tuple = (0.5, -1.0),
        tilt_db_static: tuple = (0.0, 0.0),
        low_shelf_hz: float = 250.0,
        high_shelf_hz: float = 3000.0,
        impulse_sharpen_ms: float = 3.0,
        impulse_sharpen_depth: float = 0.20,
        smooth_ms: float = 8.0,
        sample_rate: int = 48000,
        sfps: float = 24.0,
        attr: str = "normal_force_magnitude",
        **kw,
    ):
        super().__init__(
            tilt_db_impact=tilt_db_impact,
            tilt_db_rolling=tilt_db_rolling,
            tilt_db_sliding=tilt_db_sliding,
            tilt_db_scraping=tilt_db_scraping,
            tilt_db_mixed=tilt_db_mixed,
            tilt_db_static=tilt_db_static,
            low_shelf_hz=low_shelf_hz,
            high_shelf_hz=high_shelf_hz,
            impulse_sharpen_ms=impulse_sharpen_ms,
            impulse_sharpen_depth=impulse_sharpen_depth,
            smooth_ms=smooth_ms,
            sample_rate=sample_rate,
            sfps=sfps,
            attr=attr,
            **kw,
        )

    # ------------------------------------------------------------------
    # Resolve per-sample tilt targets
    # ------------------------------------------------------------------
    def _resolve_tilt_tables(self, props):
        """
        Return two length-7 arrays indexed by ContactType.value:
            low_db_table[type], high_db_table[type].

        `props` allows per-evaluation override of the whole tuple of
               tilts, which is useful for testing and for per-object
        calibration.
        """
        impact = tuple(self.p(props, "tilt_db_impact", self.params["tilt_db_impact"]))
        rolling = tuple(self.p(props, "tilt_db_rolling", self.params["tilt_db_rolling"]))
        sliding = tuple(self.p(props, "tilt_db_sliding",self.params["tilt_db_sliding"]))
        scraping = tuple(self.p(props, "tilt_db_scraping", self.params["tilt_db_scraping"]))
        mixed = tuple(self.p(props, "tilt_db_mixed", self.params["tilt_db_mixed"]))
        static = tuple(self.p(props, "tilt_db_static", self.params["tilt_db_static"]))

        low_db_table = np.zeros(7, dtype=np.float32)
        high_db_table = np.zeros(7, dtype=np.float32)

        low_db_table[_CT_IMPACT] = np.float32(impact[0])
        high_db_table[_CT_IMPACT] = np.float32(impact[1])
        low_db_table[_CT_SCRAPE] = np.float32(scraping[0])
        high_db_table[_CT_SCRAPE] = np.float32(scraping[1])
        low_db_table[_CT_SLIDE] = np.float32(sliding[0])
        high_db_table[_CT_SLIDE] = np.float.float32(sliding[1])
        low_db_table[_CT_ROLLING] = np.float32(rolling[0])
        high_db_table[_CT_ROLLING] = np.float32(rolling[1])
        low_db_table[_CT_MIXED] = np.float32(mixed[0])
        high_db_db_table[_CT_MIXED] = np.float32(mixed[1])
        low_db_table[_CT_STATIC] = np.float32(static[0])
        high_db_table[_CT_STATIC] = np.float32(static[1])
        # NO_CONTACT stays at (0, 0) by construction.

        return low_db_table, high_db_table

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        C, S = x.shape
        if S <= 1:
            return x

        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        attr = self.p(props, "attr", self.params["attr"])
        low_hz = float(self.p(props, "low_shelf_hz", self.params["low_shelf_hz"]))
        high_hz = float(self.p(props, "high_shelf_hz", self.params["high_shelf_hz"]))
        smooth_ms = float(self.p(props, "smooth_ms", self.params["smooth_ms"]))
        imp_ms = float(self.p(props, "impulse_sharpen_ms", self.params["impulse_sharpen_ms"]))
        imp_depth = np.float32(self.p(props, "impulse_sharpen_depth", self.params["impulse_sharpen_depth"]))

        # -------- 1. Per-sample force envelope + contact types --------
        env = sample_envelope(
            seq,
            num_samples=S,
            sfps=sfps,
            sample_rate=sr,
            attr=attr,
            normalize=True,
            clip_positive=True,
        )
        ctypes = sample_contact_types(seq, num_samples=S, sfps=sfps, sample_rate=sr)

        # -------- 2. Lookup per-sample (low_db, high_db) targets ------
        low_table, high_table = self._resolve_tilt_tables(props)

        low_db = np.empty(S, dtype=np.float32)
        high_db = np.empty(S, dtype=np.float32)
        for i in range(S):
            t = int(ctypes[i])
            f = env[i]
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            if t < 0 or t >= 7:
                t = 0
            # only the *energy* of a contact should drive the tilt; the
            # contact-type tables are already ±tilt_db around 0 dB.
            low_db[i] = low_table[t] * f
            high_db[i] = high_table[t] * f

        # -------- 3. Smooth the tilts (avoid zipper noise) -----------
        smooth_radius = int(max(1, smooth_ms * sr / 1000.0))
        low_db = _smooth_1d(low_db, np.int64(smooth_radius))
        high_db = _smooth_1d(high_db, np.int64(smooth_radius))

        # -------- 4. Apply the shelf tilt ----------------------------
        nyq = 0.5 * sr
        fc_low = float(np.clip(low_hz, 20.0, nyq * 0.9))
        fc_high = float(np.clip(high_hz, 200.0, nyq * 0.9))

        y = _shelf_tilt_2d(
            x,
            low_db,
            high_db,
            np.float32(fc_low / sr),
            np.float32(fc_high / sr),
        )

        # -------- 5. IMPACT transient sharpen ------------------------
        if imp_depth <= 0.0 or imp_ms <= 0.0:
            return y

        impact_len = int(max(1, imp_ms * sr / 1000.0))
        impact_mask = np.zeros(S, dtype=np.uint8)
        on = False
        start = 0
        for i in range(S):
            active = int(ctypes[i]) == _CT_IMPACT and env[i] >= 0.05
            if active and not on:
                on = True
                start = i
            elif not active and on:
                on = False
                end = min(S, start + impact_len)
                impact_mask[start:end] = 1
        if on:
            end = min(S, start + impact_len)
            impact_mask[start:end] = 1

        if not impact_mask.any():
            return y

        return _impulse_sharpen_2d(y, impact_mask, imp_depth)
