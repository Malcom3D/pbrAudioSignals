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
from scipy import signal

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import AUDIO_FORCE
from .force_sidechannel import get_force_sequence, sample_envelope, eval_at_frame

@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _apply_leak_gate(mag: np.ndarray, floor: np.ndarray, time_gain: np.ndarray, bin_gain: np.ndarray) -> np.ndarray:
    """
    mag       : (F, T) linear magnitude
    floor     : (F,)   per-bin leakage floor
    time_gain : (T,)   attenuation per frame (from force envelope)
    bin_gain  : (F,)   attenuation per bin  (from spectral shape)
    """
    F, T = mag.shape
    out = np.empty_like(mag)
    for t in nb.prange(T):
        g_t = time_gain[t]
        for f in range(F):
            m = mag[f, t]
            # remove the per-bin leakage floor
            m = m - floor[f]
            if m < 0.0:
                m = 0.0
            # soft-gate with an envelope-derived gain plus a spectral shape
            m *= (np.float32(1.0) * g_t) * bin_gain[f]
            out[f, t] = m
    return out

@register_processor("force_spectral_leak_reduce", track_group=AUDIO_FORCE)
class ForceSpectralLeakReduce(BaseProcessor):
    """
    Reduce leakage/shape artifacts left by ForceSynth by using the
    ForceDataSequence as an *oracle* for the temporal-spectral envelope.

    Difference vs. ``ForceSpectralDenoise``
    ---------------------------------------
    * ``ForceSpectralDenoise`` subtracts a *noise floor* per bin driven
      by the quiet portion of the force.
    * This processor treats the *whole buffe* as a target: it estimates
      the "leak floor" as the portion of the spectrum that ForceSynth
      can't plausibly have generated (its impulse envelope length is
      known from ``impact_duration`` and ``coupling_strength``). It
      then time-gates frames whose force envelope is too small relative
      to a *neighbour-weighted* peak.

    Params
    ------
    n_fft               : int   = 1024
    hop_length          : int   = 256
    min_gain_db         : float = -30.0  # strongest attenuation
    leak_margin         : float = 0.25   # fraction of peak that still leaks
    attr                : str   = "normal_force_magnitude"
    sample_rate         : int   = 48000
    sfps                : float = 24.0
    """

    required_side = {"force"}

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, min_gain_db: float = -30.0, leak_margin: float = 0.25, attr: str = "normal_force_magnitude", sample_rate: int = 48000, sfps: float = 24.0, **kw):
        super().__init__(n_fft=n_fftt, hop_length=hop_length, min_gain_db=min_gain_db, leak_margin=leak_margin, attr=attr, sample_rate=sample_rate, sfps=sfps, **kw)

    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        C, S = x.shape
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        n_fft = int(self.p(props, "n_fft", self.params["n_fft"]))
        hop = int(self.p(props, "hop_length", self.params["hop_length"]))
        min_db = float(self.p(props, "min_gain_db", self.params["min_gain_db"]))
        margin = float(self.p(props, "leak_margin", self.params["leak_margin"]))
        attr = self.p(props, "attr", self.params["attr"])

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr=attr, normalize=True, clip_positive=True)

        # Precompute the spatial (bin) leakage profile once for the whole stream.
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        # ForceSynth envelopes are typically short, low-frequency dominated;
        # a mid-side frequency bump leaks outward. We dampen the tails
        # of the spectrum by a smooth raised-cosine that starts just past
        # 0.5 * Nyquist. This is a *shape prior*, not a noise floor.
        bin_gain = np.ones(freqs.shape[0], dtype=np.float32)
        active_bins = freqs < 0.5 * (sr / 2.0)
        bin_gain[~active_bins] = np.float32(0.4)

        # Time gain per STFT frame
        t_floor = np.float32(10.0 ** (min_db / 20.0))
        gain_range = np.float32(1.0) - t_floor

        out = np.empty_like(x)
        for ch in range(C):
            _, _, Z = signal.stft(x[ch], nperseg=n_fft, noverlap=n_fft - hop, padded=False, boundary=None)
            mag = np.abs(Z)
            phase = np.angle(Z)

            t_frames = mag.shape[1]
            # map each STFT frame to a sample index
            frame_samples = np.arange(t_frames) * hop
            frame_samples = np.clip(frame_samples, 0, S - 1)
            env_at_frames = env[frame_samples].astype(np.float32)

            # neighbour-weighted envelope peak -> only genuine local maxima
            # get full gain; leakage gets attenuated.
            peak = float(np.max(env_at_frames)) if env_at_frames.size else 0.0
            # force-shrink to encourage leakage suppression away from peaks
            if peak > 0.0:
                env_at_frames = np.power(env_at_frames / peak, 1.0 / max(1.0, 1.0 - margin))
            # Gate
            time_gain = t_floor + gain_range * np.clip(env_at_frames, 0.0, 1.0)
            time_gain = time_gain.astype(np.float32)

            # Per-bin leak floor: use the 10th percentile of the magnitude
            # *within quiet frames* to estimate the additive leakage.
            quiet = time_gain < (t_floor + 0.1 * gain_range)
            if quiet.any():
                floor = np.percentile(mag[:, quiet], 10, axis=1)
            else:
                floor = np.percentile(mag, 10, axis=1)
            floor = floor.astype(np.float32)

            mag_clean = _apply_leak_gate(mag.astype(np.float32), floor, time_gain, bin_gain)

            Zc = mag_clean * np.exp(1j * phase)
            _, y = signal.istft(Zc, nperseg=n_fft, noverlap=n_fft - hop, boundary=False)
            if y.shape[0] > S:
                y = y[:S]
            elif y.shape[0] < S:
                y = np.pad(y, (0, S - y.shape[0]))
            out[ch] = y.astype(np.float32)

        return out
