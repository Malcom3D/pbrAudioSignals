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
Helpers shared by the audio-force processors.

The StorageEngine._get_props() injects the resolved `ForceDataSequence`
into `props.custom["force"]`. These helpers wrap it so the processors
never have to guess at `frames` alignment or deal with NaN gaps.
"""
from typing import Optional, Tuple
import numpy as np

from pbrAudioCommon.lib.force_data import ForceDataSequence, ContactType

def get_force_sequence(props) -> Optional[ForceDataSequence]:
    """Return the ForceDataSequence injected by StorageEngine, or None."""
    if props is None:
        return None
    custom custom = getattr(props, "custom", None)
    if not isinstance(custom, dict):
        return None
    seq = custom.get("force")
    return seq if isinstance(seq, ForceDataSequence) else None


def _safe_eval(value) -> float:
    """Evaluate a scalar / spline / ndarray entry to a float safely."""
    if value is None:
        return 0.0
    if isinstance(value, np.ndarray):
        return float(np.nan_to_num(value, nan=0.0).ravel()[0]) if value.size else 0.0
    try:
        v = value(0.0) if callable(value) and not hasattr(value, "x") else value
    except Exception:
        try:
            v = value
        except Exception:
            return 0.0
    try:
        return float(np.nan_to_num(v, nan=0.0))
    except Exception:
        return 0.0


def eval_at_frame(seq: ForceDataSequence, frame_idx: float, attr: str) -> float:
    """
    Evaluate a scalar attribute of the ForceDataSequence at the given frame.

    `attr` must be the *getter* name, one of:
        normal_force_magnitude, tangential_force_magnitude,
        relative_velocity (magnitude), coupling_strength, contact_radius,
        contact_pressure, penetration_depth, impact_duration, rolling__radius
    """
    if seq is None or getattr(seq, "frames", None) is None or seq.frames.size == 0:
        return 0.0
    getter = getattr(seq, f"get_{attr}", None)
    if getter is None:
        return 0.0
    try:
        value = getter(frame_idx)
    except Exception:
        return 0.0

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        if value.size == 1:
            return float(np.nan_to_num(value.ravel()[0], nan=0.0))
        # Vector getter (normal_force / tangential_force / velocities)
        return float(np.linalg.norm(np.nan_to_num(value, nan=0.0)))
    return _safe_eval(value)


def eval_contact_type(seq: ForceDataSequence, frame_idx: float) -> int:
    """Return the ContactType.value at the given frame (0 for none)."""
    if seq is None or getattr(seq, "frames", None) is None or seq.frames.size == 0:
        return 0
    try:
        raw = seq.get_contact_type(frame_idx)
    except Exception:
        return 0
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return 0
        raw = raw.ravel()[0]
    try:
        return int(raw)
    except Exception:
        return 0


def sample_envelope(seq: ForceDataSequence, num_samples: int, sfps: float, sample_rate: int, attr: str = "normal_force_magnitude", normalize: bool = True, clip_positive: bool = True) -> np.ndarray:
    """
    Sample a scalar force attribute densely along the audio stream.

    The audio stream (arrives at the processor) is at `sample_rate`.
    The ForceDataSequence is at `sfps` frames per second.
    """
    out = np.zeros(num_samples, dtype=np.float32)
    if seq is None or num_samples == 0:
        return out

    frames = np.asarray(seq.frames, dtype=np.float64)
    if frames.size == 0:
        return out

    # Render a per-sample timestep in *frame units* the ForceDataSequence uses
    # (StorageEngine passes the full array, sample 0 == frame 0).
    sample_to_frame = sfps / float(sample_rate)
    sample_frames = np.arange(num_samples, dtype=np.float64) * sample_to_frame

    lo = float(frames.min())
    hi = float(frames.max())

    # Vectorized path when the getter supports arrays; else loop.
    try:
        raw = seq.__dict__[f"{attr}"]
        getter_vec = np.vectorize(
            lambda f: _safe_eval(seq.__class__.__dict__[f"get_{attr}"](seq, f))
        )
        # keep it simple; only use the vector path for scalar-only attrs
        raise RuntimeError("use scalar path")
    except Exception:
        pass

    # Chunked scalar path (fast enough at 48 kHz because getters are splines).
    in_range = (sample_frames >= lo) & (sample_frames <= hi)
    idxs = np.flatnonzero(in_range)
    if idxs.size == 0:
        return out

    # Coarse render: 1 frame worth of resolution
    frames_step = max(1.0, (hi - lo) / max(1, frames.size - 1))
    coarse_len = max(2, int(np.ceil((hi - lo) / frames_step)) + 1)
    coarse_frames = np.linspace(lo, hi, coarse_len, dtype=np.float64)
    coarse_vals = np.fromiter(
        (eval_at_frame(seq, f, attr) for f in coarse_frames),
        dtype=np.float32,
        count=coarse_len,
    )
    if clip_positive:
        np.maximum(coarse_vals, 0.0, out=coarse_vals)
    if normalize:
        peak = float(np.max(coarse_vals)) if coarse_vals.size else 0.0
        if peak > 0.0:
            coarse_vals /= peak

    # Linear interp at sample resolution
    out[idxs] = np.interp(sample_frames[idxs], coarse_frames, coarse_vals)
    return out


def sample_contact_types(seq: ForceDataSequence, num_samples: int, sfps: float, sample_rate: int) -> np.ndarray:
    """Per-sample ContactType.value array (dense, sample rate)."""
    out = np.zeros(num_samples, dtype=np.int32)
    if seq is None or num_samples == 0:
        return out
    frames = np.asarray(seq.frames, dtype=np.float64)
    if frames.size == 0:
        return out

    sample_to_frame = sfps / float(sample_rate)
    sample_frames = np.arange(num_samples, dtype=np.float64) * sample_to_frame

    lo, hi = float(frames.min()), float(frames.max())
    in_range = (sample_frames >= lo) & (sample_frames <= hi)
    idxs = np.flatnonzero(in_range)
    for i in idxs:
        out[i] = eval_contact_type(seq, sample_frames[i])
    return out
