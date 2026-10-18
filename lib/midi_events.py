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

# pbrAudioSignals/lib/midi_events.py
"""
MIDI event representation for plugin side-channels.

Plugins receive MIDI via `props.custom['midi']`, expected as one of:

    - None                    -> no MIDI events
    - list[dict]              -> events with 'sample_offset', 'type', ...
    - np.ndarray (N, 4)       -> (sample_offset, status, data1, data2) uint/int

The dtype layout for the ndarray form:
    col 0: sample_offset   (int32)   offset within current block
    col 1: status          (uint8)   MIDI status byte (0x90 note_on, etc.)
    col 2: data1           (uint8)   note number or CC number
    col 3: data2           (uint8)   velocity or CC value
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional


# MIDI status bytes (channel 0 by default)
NOTE_OFF = 0x80
NOTE_ON = 0x90
POLY_PRESSURE = 0xA0
CONTROL_CHANGE = 0xB0
PROGRAM_CHANGE = 0xC0
CHANNEL_PRESSURE = 0xD0
PITCH_BEND = 0xE0


def _events_to_records(midi: Any) -> List[dict]:
    """Normalise any accepted MIDI representation to a list of dicts."""
    if midi is None:
        return []

    if isinstance(midi, np.ndarray):
        # Structured ndarray of shape (N, 4) -> (offset, status, data1, data2)
        arr = np.asarray(midi)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"MIDI ndarray must be (N, 4), got {arr.shape}")
        out = []
        for row in arr:
            status = int(row[1]) & 0xF0
            channel = int(row[1]) & 0x0F
            if status == NOTE_ON:
                kind = "note_on"
            elif status == NOTE_OFF:
                kind = "note_off"
            elif status == CONTROL_CHANGE:
                kind = "cc"
            elif status == PITCH_BEND:
                kind = "pitchbend"
            elif status == PROGRAM_CHANGE:
                kind = "program"
            else:
                kind = "raw"
            out.append({
                "sample_offset": int(row[0]),
                "type": kind,
                "status": int(row[1]),
                "channel": channel,
                "data1": int(row[2]),
                "data2": int(row[3]),
            })
        return out

    if isinstance(midi, list):
        return [dict(evt) for evt in midi]

    raise TypeError(f"Unsupported MIDI side-channel type: {type(midi)}")


def _to_minihost_input_events(midi: Any, num_frames: int, block_start: int):
    """
    Convert MIDI events into the list-of-tuples format expected by
    minihost's `PluginInstance.process_block`:

        [(offset, make_event(...)), ...]

    `make_event` is bound late so we can tolerate different minihost builds.
    """
    from minihost import make_event  # local import; may be `minihost.events.make_event`

    evts = _events_to_records(midi)
    out = []
    for e in evts:
        off = int(e.get("sample_offset", 0)) - block_start
        if off < 0:
            off = 0
        elif off >= num_frames:
            off = num_frames - 1

        status = int(e.get("status", e.get("type_code", 0)))
        d1 = int(e.get("data1", e.get("note", 0)))
        d2 = int(e.get("data2", e.get("velocity", 0)))

        # If only type/note info was given, synthesise a status byte
        if status == 0:
            t = e.get("type", "")
            channel = int(e.get("channel", 0)) & 0x0F
            if t == "note_on":
                status = NOTE_ON | channel
                d1, d2 = d1, d2 if d2 > 0 else 100
            elif t == "note_off":
                status = NOTE_OFF | channel
            elif t == "cc":
                status = CONTROL_CHANGE | channel
            elif t == "pitchbend":
                status = PITCH_BEND | channel
            elif t == "program":
                status = PROGRAM_CHANGE | channel
            else:
                status = NOTE_ON | channel  # fallback

        try:
            out.append((off, make_event(status, d1, data2=d2)))
        except TypeError:
            # Older minihost signatures
            out.append((off, make_event(status, d1, d2)))
    return out
