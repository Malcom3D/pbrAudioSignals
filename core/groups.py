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
Canonical track-group identifiers.

A track_group describes the *physicalphysical meaning* of the signal carried by
a track in an AudioSchema:

- ``generic``            : any-track common processors (dc-block, normalise, ...)
- ``audio-force``        : modal excitation signals (force / impulse side channels)
- ``modal-displacement`` : modal synthesis output (per-mode / per-vertex displacement)
- ``acoustic-pressure``  : pressure variations in a medium due to sound waves
- ``audio-signal``       : electrical signal after transducer (e.g. microphone)
- ``ambisonic``          : ambisonic (B-format) track
- ``surround``           : fixed-layout surround tracks
- ``control``            : non-audio automation / envelope tracks

Callers may also pass a custom string; the registry only enforces
matching when a track_group is *explicitly requested*.
"""

from typing import Dict, FrozenSet

# ---- canonical names ---------------------------------------------------

GENERIC              = "generic"
AUDIO_FORCE          = "audio-force"
MODAL_DISPLACEMENT   = "modal-displacement"
ACOUSTIC_PRESSURE    = "acoustic-pressure"
AUDIO_SIGNAL         = "audio-signal"
AMBISONIC            = "ambisonic"
SURROUND             = "surround"
CONTROL              = "control"

KNOWN_GROUPS: FrozenSet[str] = frozenset({
    GENERIC,
    AUDIO_FORCE,
    MODAL_DISPLACEMENT,
    ACOUSTIC_PRESSURE,
    AUDIO_SIGNAL,
    AMBISONIC,
    SURROUND,
    CONTROL,
})

# ---- human-readable descriptions --------------------------------------

GROUP_DESCRIPTIONS: Dict[str, str] = {
    GENERIC: "Common processors applicable to any track.",
    AUDIO_FORCE: "Processors for modal excitation / contact forces.",
    MODAL_DISPLACEMENT: "Processors for modal synthesis displacement output.",
    ACOUSTIC_PRESSURE: "Processors for pressure variations due to sound waves in a medium.",
    AUDIO_SIGNAL: "Processors for transducer-converted audio (microphones, hydrophones).",
    AMBISONIC: "Processors for ambisonic (B-format) tracks.",
    SURROUND: "Processors for fixed-layout surround tracks.",
    CONTROL: "Processors for non-audio control / automation tracks.",
}
