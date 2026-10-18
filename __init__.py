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

__version__ = "0.0.3"
__author__ = "Malcom3D"
__description__ = (
    "Non-destructive audio signal processors for the pbrAudioStorage DAG. "
    "All processors implement the SignalProcessor interface and are blosc2 JIT friendly."
)

import os
import numpy as np

decimals = 18
np.set_printoptions(precision=decimals, floatmode='fixed', threshold=np.inf)

from .core.processor import BaseProcessor, register_processor, get_processor
from .lib.dc_blocker import DCBlocker
from .lib.noise_gate import AdaptiveNoiseGate
from .lib.temporal_smoothing import TemporalSmoothing
from .lib.spectral_denoise import SpectralDenoise
from .lib.envelope_shaper import EnvelopeShaper
from .lib.adaptive_gaussian import AdaptiveGaussianSmoothing
from .lib.force_weighted import ForceWeightedGain, ForceWeightedBlend
from .lib.normalize import GlobalNormalize, PerTrackNormalize
from .lib.chain import ChainProcessor
from .lib.ambi_decode import AmbisonicDecoder
from .lib.plugin_processor import PluginProcessor
from .lib.plugin_host import PluginHost, HostNotAvailable

__all__ = [
    "BaseProcessor",
    "register_processor",
    "get_processor",
    "DCBlocker",
    "AdaptiveNoiseGate",
    "TemporalSmoothing",
    "SpectralDenoise",
    "EnvelopeShaper",
    "AdaptiveGaussianSmoothing",
    "ForceWeightedGain",
    "ForceWeightedBlend",
    "GlobalNormalize",
    "PerTrackNormalize",
    "ChainProcessor",
    "AmbisonicDecoder",
    "PluginProcessor",
    "PluginHost",
    "HostNotAvailable",
]
