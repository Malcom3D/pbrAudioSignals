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
from scipy import signal

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import MODAL_DISPLACEMENT, AUDIO_FORCE

@register_processor("force_spectral_denoise", track_group=MODAL_DISPLACEMENT, also_valid_for={AUDIO_FORCE})
class ForceSpectralDenoise(BaseProcessor):
    """
    Force-driven spectral subtraction.

    Mirrors SynthPostProcess._spectral_denoise():
      * STFT the signal,
      * estimate noise floor from the quiet portion of force (< 0.1),
      * interpolate per-frame reduction strength as
            strength = base * (1 - 0.8 * force_env),
      * spectral subtract, then ISTFT back to original length.

    Params
    ------
    fft_size             : int,   default 2048
    hop_size             : int,   default 512
    reduction_strength   : float, default 0.8       # base strength
    force_coupling       : float, default 0.8       # how strongly force cuts strength
    quiet_threshold      : float, default 0.1       # force < this -> "quiet"
    sample_rate          : int,   default 48000
    """

    required_side = {"force"}

    def __init__(self,
                 fft_size: int = 2048,
                 hop_size: int = 512,
                 reduction_strength: float = 0.8,
                 force_coupling: float = 0.8,
                 quiet_threshold: float = 0.1,
                 sample_rate: int = 48000, **kw):
        super().__init__(fft_size=fft_size,
                         hop_size=hop_size,
                         reduction_strength=reduction_strength,
                         force_coupling=force_coupling,
                         quiet_threshold=quiet_threshold,
                         sample_rate=sample_rate, **kw)

    def _process(self, x, props):
        S = x.shape[1]
        force = self.side(props, "force", expected_len=S)
        if force is None:
            return x

        n_fft = int(self.p(props, "fft_size", self.params["fft_size"]))
        hop = int(self.p(props, "hop_size", self.params["hop_size"]))
        strength = float(self.p(props, "reduction_strength",
                                self.params["reduction_strength"]))
        coupling = float(self.p(props, "force_coupling",
                                self.params["force_coupling"]))
        quiet_thr = float(self.p(props, "quiet_threshold",
                                 self.params["quiet_threshold"]))
        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))

        # normalize force to 0..1 (abs)
        f_abs = np.abs(force).astype(np.float32)
        fmax = float(f_abs.max()) if f_abs.max() > 0 else 1.0
        f_norm = f_abs / fmax

        out = np.empty_like(x)
        for ch in range(x.shape[0]):
            _, _, Zxx = signal.stft(x[ch], fs=sr, nperseg=n_fft,
                                    noverlap=n_fft - hop,
                                    boundary=None, padded=False)

            t_frames = Zxx.shape[1]
            force_at_frames = np.interp(
                np.linspace(0, S, t_frames, endpoint=False),
                np.arange(S), f_norm
            ).astype(np.float32)

            # noise floor from quiet frames
            quiet_mask = force_at_frames < quiet_thr
            if quiet_mask.any():
                noise_floor = np.sqrt(np.mean(np.abs(Zxx[:, quiet_mask]) ** 2, axis=1))
            else:
                noise_floor = np.median(np.abs(Zxx), axis=1) * 0.1

            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)

            per_frame_strength = strength * (1.0 - coupling * force_at_frames)

            mag_clean = np.maximum(
                magnitude - per_frame_strength[np.newaxis, :] * noise_floor[:, np.newaxis],
                0.0,
            )

            Zclean = mag_clean * np.exp(1j * phase)
            _, y = signal.istft(Zclean, fs=sr, nperseg=n_fft,
                                noverlap=n_fft - hop, boundary=False)

            if y.shape[0] > S:
                y = y[:S]
            elif y.shape[0] < S:
                y = np.pad(y, (0, S - y.shape[0]))

            out[ch] = y.astype(np.float32)

        return out
