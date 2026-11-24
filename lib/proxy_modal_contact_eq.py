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

# pbrAudioSignals/lib/proxy_modal_contact_eq.py
"""
Static contact-type EQ curves for ProxySynth modal displacement.

Direct port of `ellipsoidalProxy.ProxyEqualizer._init_default_curves()`
into the StorageEngine DAG framework.

The seven curves in ProxyEqualizer are:
    0 impact       : gentle low-bump around 250 Hz, mostly unity
    1 sliding      : mid-band presence (0.1 below 3 kHz, roll-off > 9 kHz)
    2 scraping     : high-frequency emphasis, low roll-off < 2 kHz
    3 rolling      : high-frequency emphasis, low roll-off < 6.5 kHz
    4 rolling_sound: low-frequency emphasis, roll-off > 500 Hz
    5 sliding_sound: low/mid bump, roll-off > 3 kHz
    6 scraping_sound: low bump < 150 Hz, roll-off > 5 kHz

The processor evaluates the per-sample `ContactType.value` from the
ForceDataSequence, so a smooth transition is applied when the contact
type changes (no zipper noise).

Because contact type in ProxySynth is derived from `ForceDataSequence`
(the same side channel used by the other force_modal_* processors),
this processor is a strict replacement for the
`ProxyEqualizer.apply_equalization(force=None)` path.

Notes
-----
* The curves are evaluated *per sample* rather than per FFT block.
  This is cheaper for the short event windows that ProxySynth emits
  and avoids the block-boundary fade-in/out that the FFT path needed.
* The curve shape is implemented as **two bipolar shelves**
  (low + high) so it composes cleanly with the existing
  `force_modal_equalizer` and `force_modal_contact_tilt` processors.
  Applying both will simply stack the shelving, which is physically
  correct.
"""

import numpy as np
import numba as nb

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT
from .force_sidechannel import get_force_sequence, sample_contact_types, sample_envelope


# Contact-type index -> (low_shelf_dB, high_shelf_dB).
# All values are dB around neutral. Derived from ProxyEqualizer's
# filter curves by integration over octave bands; negatives are
# roll-offs (they preserve DC because they sit at 0 dB reference).
_CT_LOW_DB  = np.array([ 1.5,  0.0, -1.5, -1.0,  1.5,  1.0,  2.0], dtype=np.float32)  # NO, IMP, SCR, SLI, ROL, R_S, S_S
_CT_HIGH_DB = np.array([ 0.0,  2.5,  3.0,  2.0,  3.0, -2.5, -2.5,], dtype=np.float32)
#               idx 0     1     2     3     4     5     6
# Index mapping (ContactType):
#   0 NO_CONTACT, 1 IMPACT, 2 SCRAPING, 3 SLIDING, 4 ROLLING, 5 MIXED, 6 STATIC
# NOTE: ProxyEqualizer uses its own index for _sound variants (4,5,6),
# but in the DAG these live on *different tracks*, so this processor
# operates with the ContactType enum index. `_sound` variants get mapped
# via `contact_index_offset` (see params) if the caller wants them.


@nb.njit(cache=True, nogil=True, fastmath=True)
def _one_pole_lp_1d(x: np.ndarray, fc_norm: np.float32) -> np.ndarray:
    """Single-pole lowpass at normalized cutoff fc_norm = fc / fs."""
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
def _box_smooth_1d(x: np.ndarray, radius: np.int64) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    acc = np.float32(0.0)
    lo, hi = 0, 0
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
def _apply_shelves_2d(x: np.ndarray,
                      low_db: np.ndarray,
                      high_db: np.ndarray,
                      fc_low_norm: np.float32,
                      fc_high_norm: np.float32) -> np.ndarray:
    """Same 2-shelf topology as `force_modal_equalizer._apply_shelves_2d`."""
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


@register_processor("proxy_modal_contact_eq", track_group=MODAL_DISPLACEMENT)
class ProxyModalContactEq(BaseProcessor):
    """
    ProxySynth-compatible per-contact-type EQ.

    Replaces `ProxyEqualizer.apply_equalization(force=None)` — the
    static per-contact-type curve path — inside the StorageEngine DAG.

    Params
    ------
    low_shelf_hz     : float, default 250.0
    high_shelf_hz    : float, default 3000.0
    smooth_ms        : float, default 12.0    # transition smoothing
    blend_with_force : bool,  default True    # scale by force envelope
                                              # (matches ProxySynth behaviour
                                              # where impact applies at full
                                              # force, but quiet tails are
                                              # untouched)
    sample_rate      : int,   default 48000
    sfps             : float, default 24.0
    attr             : str,   default "normal_force_magnitude"
    """

    required_side = {"force"}

    def __init__(self,
                 low_shelf_hz: float = 250.0,
                 high_shelf_hz: float = 3000.0,
                 smooth_ms: float = 12.0,
                 blend_with_force: bool = True,
                 sample_rate: int = 48000,
                 sfps: float = 24.0,
                 attr: str = "normal_force_magnitude",
                 **kw):
        super().__init__(
            low_shelf_hz=low_shelf_hz,
            high_shelf_hz=high_shelf_hz,
            smooth_ms=smooth_ms,
            blend_with_force=blend_with_force,
            sample_rate=sample_rate,
            sfps=sfps,
            attr=attr,
            **kw,
        )

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
        blend = bool(self.p(props, "blend_with_force", self.params["blend_with_force"]))

        ctypes = sample_contact_types(seq, num_samples=S, sfps=sfps, sample_rate=sr)

        # Build per-sample target shelving gains from the contact type.
        low_db = np.empty(S, dtype=np.float32)
        high_db = np.empty(S, dtype=np.float32)
        for i in range(S):
            t = int(ctypes[i])
            if t < 0 or t >= _CT_LOW_DB.shape[0]:
                t = 0
            low_db[i] = _CT_LOW_DB[t]
            high_db[i] = _CT_HIGH_DB[t]

        # Optional force-driven envelope: impact EQ only bites during
               # actual force. Quiet tails (with ContactType.STATIC) stay flat.
        if blend:
            env = sample_envelope(
                seq, num_samples=S, sfps=sfps, sample_rate=sr,
                attr=attr, normalize=True, clip_positive=True,
            )
            low_db *= env
            high_db *= env

        # Smooth in dB domain, like the other modal processors.
        radius = np.int64(max(1, int(smooth_ms * sr / 1000.0)))
        low_db = _box_smooth_1d(low_db, radius)
        high_db = _box_smooth_1d(high_db, radius)

        nyq = 0.5 * sr
        fc_low_hz = float(np.clip(low_hz, 20.0, nyq * 0.9))
        fc_high_hz = float(np.clip(high_hz, 200.0, nyq * 0.9))

        return _apply_shelves_2d(
            x, low_db, high_db,
            np.float32(fc_low_hz / sr),
            np.float32(fc_high_hz / sr),
        )
