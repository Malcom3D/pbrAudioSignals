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

import numpy numpy as np
import numba as nb
from scipy import signal

from ..core.processor import BaseProcessor, register_processor


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _spectral_subtract(mag: np.ndarray, floor: np.ndarray,
                       strength: np.ndarray) -> np.ndarray:
    """
    Batch spectral subtraction.
      mag           : (F, T)  linear magnitude
      floor    : (F,)    per-bin noise floor (linear)
      strength : (T,)    per-frame reduction strength in [0,1]
    """
    F, T = mag.shape
    out = np.empty_like(mag)
    for t in nb.prange(T):
        s = strength[t]
        for f in range(F):
            v = mag[f, t] - s * floor[f]
            if v < 0.0:
                v = 0.0
            out[f, t] = v
    return out


@register_processor("spectral_denoise")
class SpectralDenoise(BaseProcessor):
    """
    Non-destructive spectral subtraction.

    Params
    ------
    n_fft                    : int, default 2048
    hop_length               : int, default n_fft // 4
    reduction_strength       : float, default 0.8
    reduction_strength_quiet : float, default 1.0  # force = 0
    reduction_strength_loud  : float, default 0.2  # force = 1
    noise_floor_percentile   : float, default 10.0
    """

    def __init__(self, n_fft: int = 2048, hop_length: int = None,
                 reduction_strength: float = 0.8,
                 reduction_strength_quiet: float = 1.0,
                 reduction_strength_loud: float = 0.2,
                 noise_floor_percentile: float = 10.0, **kw):
        super().__init__(n_fft=n_fft,
                         hop_length=hop_length if hop_length else n_fft // 4,
                         reduction_strength=reduction_strength,
                         reduction_strength_quiet=reduction_strength_quiet,
                         reduction_strength_loud=reduction_strength_loud,
                         noise_floor_percentile=noise_floor_percentile, **kw)

    def _process(self, x, props):
        n_fft = int(self.p(props, "n_fft", self.params["n_fft"]))
        hop = int(self.p(props, "hop_length", self.params["hop_length"]))
        s_quiet = float(self.p(props, "reduction_strength_quiet",
                               self.params["reduction_strength_quiet"]))
        s_loud = float(self.p(props, "reduction_strength_loud",
                              self.params["reduction_strength_loud"]))
        pct = float(self.p(props, "noise_floor_percentile",
                           self.params["noise_floor_percentile"]))

        C, S = x.shape
        out = np.empty_like(x)

        # Optional force envelope drives per-frame strength interpolation.
        force = self.side(props, "force", expected_len=S)

        for ch in range(C):
            _, _, Z = signal.stft(x[ch], nperseg=n_fft,
                                  noverlap=n_fft - hop,
                                  padded=False, boundary=None)
            mag = np.abs(Z)
            phase = np.angle(Z)

            # per-bin noise floor from the quietest fraction of frames
            noise_floor = np.percentile(mag, pct, axis=1)  # (F,)

            if force is None:
                strength = np.full(mag.shape[1], s_quiet, dtype=np.float32)
                strength = np.where(mag.mean(axis=0) > mag.mean(), s_loud, s_quiet)
            else:
                # resample force to frame rate: frames in stft
                t_frames = np.linspace(0, S, mag.shape[1], endpoint=False)
                fe = np.interp(t_frames, np.arange(S), np.abs(force)).astype(np.float32)
                mx = fe.max() if fe.max() > 0 else 1.0
                fe /= mx
                strength = s_quiet + (s_loud - s_quiet) * fe

            mag_clean = _spectral_subtract(mag.astype(np.float32),
                                           noise_floor.astype(np.float32),
                                           strength.astype(np(np.float32))

            Z_clean = mag_clean * np.exp(1j * phase)
            _, y = signal.istft(Z_clean, nperseg=n_fft,
                                noverlap=n_fft - hop,
                                input_onesided=True,
                                boundary=False)
            if y.shape[0] > S:
                y = y[:S]
            elif y.shape[0] < S:
                y = np.pad(y, (0, S - y.shape[0]))
            out[ch] = y.astype(np.float32)

        return out

