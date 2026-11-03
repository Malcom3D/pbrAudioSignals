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

from ..core.processor import BaseProcessor, register_processor
from ..core.groups import AUDIO_FORCE
from .force_sidechannel import get_force_sequence, sample_envelope, sample_contact_types, eval_at_frame
from pbrAudioCommon import ContactType


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _stochastic_modulation_2d(x: np.ndarray, lfo_a: np.ndarray, lfo_b: np.ndarray, gain_peak: np.float32) -> np.ndarray:
    """
    Two independent low-rate LFOs per sample. lfo_a and lfo_b are
    already per-sample and per-channel-independent (same across channels
    by design — we *do* want a stable stereo image, only a subtle wobble).
    """
    C, S = x.shape
    out = np.empty_like(x)
    for ch in nb.prange(C):
        for i in range(S):
            # amplitude modulation
            a = np.float32(1.0) + gain_peak * lfo_a[i]
            # tiny phase-preserving fractional delay via linear interp
            d = np.float32(0.5) * lfo_b[i]  # in samples, ±0.5 max
            idx0 = i
            if d > 0:
                if i + 1 < S:
                    s = x[ch, i] * (np.float32(1.0) - d) + x[ch, i + 1] * d
                else:
                    s = x[ch, i]
            else:
                if i - 1 >= 0:
                    s = x[ch, i] * (np.float32(1.0) + d) + x[ch, i - 1] * (-d)
                else:
                    s = x[ch, i]
            out[ch, i] = s * a
    return out

def _build_lfos(num_samples: int, sfps: float, sample_rate: int, rate_hz: float, depth: float, seed: int) -> np.ndarray:
    """
    Build a per-sample, mean-zero, band-limited stochastic LFO.

    White noise is pushed through a one-pole low-pass whose pole derives
    from `rate_hz` (the LFO's cutoff). The result is mean-centered and
    hard-limited to ±depth so it can safely modulate audio.
    """
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(num_samples).astype(np.float32)

    # one-pole low-pass: y[n] = a*y[n-1] + (1-a)*x[n]
    # a = exp(-2*pi*fc/fs)
    a = float(np.exp(-2.0 * np.pi * float(rate_hz) / float(sample_rate)))
    a = np.float32(np.clip(a, 0.0, 0.99))
    one_minus_a = np.float32(1.0) - a

    out = np.empty(num_samples, dtype=np.float32)
    y = np.float32(0.0)
    for i in range(num_samples):
        y = a * y + one_minus_a * noise[i]
        out[i] = y

    # Center and normalize to ±1
    m = float(out.mean())
    out -= np.float32(m)
    peak = float(np.max(np.abs(out)))
    if peak > 0.0:
        out /= np.float32(peak)

    # Scale by depth and hard-limit (extra safety after normalization)
    out *= np.float32(depth)
    np.clip(out, -depth, depth, out=out)
    return out


@nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
def _event_amplitude_jitter_2d(x: np.ndarray, envelope: np.ndarray, event_onsets: np.ndarray, event_gains: np.ndarray, event_t60_samples: np.ndarray) -> np.ndarray:
    """
    Apply a per-event amplitude gain that decays exponentially.

    For each event e:
      - `event_onsets[e]` is the sample offset where the event begins
      - `event_gains[e]`  is the multiplicative gain (e.g. 1.0 ± 0.2)
      - `event_t60_samples[e]` is the decay envelope length in samples

    Gain at sample i is the product over all active events. This is
    what breaks the "same sample played N times" feel: the *same*
    ForceSynth envelope now has *different* amplitudes per repetition.
    """
    C, S = x.shape
    out = np.empty_like(x)
    active_gain = np.ones(S, dtype=np.float32)

    n_events = event_onsets.shape[0]
    for e in range(n_events):
        onset = int(event_onsets[e])
        if onset < 0 or onset >= S:
            continue
        gain = float(event_gains[e])
        if abs(gain - 1.0) < 1e-6:
            continue
        length = int(event_t60_samples[e])
        if length <= 0:
            active_gain[onset] *= gain
            continue
        end = onset + length
        if end > S:
            end = S
        # exponential decay: gain * exp(-t/tau), tau chosen so that the
        # envelope reaches 1.0 (unity) exactly at `end`.
        decay_len = float(length)
        for i in range(onset, end):
            t = float(i - onset)
            # blend toward unity as the event decays: only fade the
            # *modification* around unity, never the audio itself
            w = float(np.exp(-3.5 * t / decay_len))
            active_gain[i] *= (1.0 + (gain - 1.0) * w)

    for ch in nb.prange(C):
        for i in range(S):
            out[ch, i] = x[ch, i] * active_gain[i]
    return out


def _detect_events(contact_types: np.ndarray, envelope: np.ndarray, sfps: float, sample_rate: int, min_event_seconds: float = 0.001, target_types: tuple = (ContactType.IMPACT.value,)) -> np.ndarray:
    """
    Return (event_onsets, event_lengths) where an event is a contiguous
    run of samples whose contact_type is in `target_types` AND whose
    force envelope rises above 0.1.

    The run length is then used both to size the exponential decay
    and to feed the max event length back into the caller.
    """
    S = envelope.shape[0]
    onsets = []
    lengths = []

    in_event = False
    start = 0
    for i in range(S):
        active = ((int(contact_types[i]) in target_types) and (envelope[i] > 0.05))
        if active and not in_event:
            in_event = True
            start = i
        elif not active and in_event:
            in_event = False
            length = i - start
            if length >= max(1, int(min_event_seconds * sample_rate)):
                onsets.append(start)
                lengths.append(length)

    if in_event:
        length = S - start
        if length >= max(1, int(min_event_seconds * sample_rate)):
            onsets.append(start)
            lengths.append(length)

    return (np.asarray(onsets, dtype=np.int64), np.asarray(lengths, dtype=np.int64))

@register_processor("force_periodicity_breaker", track_group=AUDIO_FORCE)
class ForcePeriodicityBreaker(BaseProcessor):
    """
    Break the "pre-recorded" repetition feel produced by ForceSynth by
    adding stochastic variation *driven by the ForceDataSequence*.

    Two complementary mechanisms, applied additively:

    1. **Per-event amplitude jitter** — detect contiguous events marked
       IMPACT (and typically with the same coupling_strength as the
       previous events), then apply small ± ``event_gain_jitter`` gains
       to each event with an exponential tail so the modification never
       clicks. This is the biggest win for "same sample repeated N
       times".

    2. **Low-rate stochastic LFOs** — applied to ROLLING / SLIDING /
       SCRAPING sections (via ``contact_types``) so the continuous
       contact never sounds like a looped pad. Amplitude (``lfo_depth``)
       and fractional sample delay (``lfo_delay_depth``) are independent.

    The repetition detector
    -----------------------
    When the same ``coupling_strength`` value recurs, we boost both
    ``event_gain_jitter`` and ``lfo_depth`` for that region — this
    catches the "same collision retriggered" case where ForceSynth's
    envelope values are bit-identical and the ear *will* notice.

    Params
    ------
    event_gain_jitter   : float = 0.12   # ±fractional gain per event
    lfo_depth           : float = 0.04   # ±fractional amplitude
    lfo_delay_depth     : float = 0.25   # ±samples (max 0.5)
    lfo_rate_hz         : float = 0.8    # LFO corner
    repetition_boost    : float = 1.8    # multiplier when coupling recurs
    coupling_tol        : float = 0.02   # tolerance for "same coupling"
    sample_rate         : int   = 48000
    sfps                : float = 24.0
    seed                : int   = 1337
    """

    required_side = {"force"}

    def __init__(self, event_gain_jitter: float = 0.12, lfo_depth: float = 0.04, lfo_delay_depth: float = 0.25, lfo_rate_hz: float = 0.8, repetition_boost: float = 1.8, coupling_tol: float = 0.02, sample_rate: int = 48000, sfps: float = 24.0, seed: int = 1337, **kw):
        super().__init__(event_gain_jitter=event_gain_jitter, lfo_depth=lfo_depth, lfo_delay_depth=lfo_delay_depth, lfo_rate_hz=lfo_rate_hz, repetition_boost=repetition_boost, coupling_tol=coupling_tol, sample_rate=sample_rate, sfps=sfps, seed=seed, **kw)
        # cache: a stable feature is the stream length, so we key on it
        self._lfo_cache: dict = {}

    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        seq = get_force_sequence(props)
        if seq is None:
            return x

        C, S = x.shape
        if S <= 2:
            return x

        sr = int(self.p(props, "sample_rate", self.params["sample_rate"]))
        sfps = float(self.p(props, "sfps", self.params["sfps"]))
        seed = int(self.p(props, "seed", self.params["seed"]))

        gain_jit = float(self.p(props, "event_gain_jitter", self.params["event_gain_jitter"]))
        lfo_depth = float(self.p(props, "lfo_depth", self.params["lfo_depth"]))
        lfo_delay = float(self.p(props, "lfo_delay_depth", self.params["lfo_delay_depth"]))
        lfo_rate = float(self.p(props, "lfo_rate_hz", self.params["lfo_rate_hz"]))
        boost = float(self.p(props, "repetition_boost", self.params["repetition_boost"]))
        tol = float(self.p(props, "coupling_tol", self.params["coupling_tol"]))

        env = sample_envelope(seq, num_samples=S, sfps=sfps, sample_rate=sr, attr="normal_force_magnitude", normalize=True, clip_positive=True)
        ctypes = sample_contact_types(seq, num_samples=S, sfps=sfps, sample_rate=sr)

        # ---------- 1. Detect events and detect repetition ----------
        onsets, lengths = _detect_events(ctypes, env, sfps=sfps, sample_rate=sr)

        # Coupling strengths at each onset (for repetition detection)
        coupling_per_event = np.zeros(onsets.shape[0], dtype=np.float32)
        for i, onset in enumerate(onsets):
            frame = onset * (sfps / float(sr))
            eaf = eval_at_frame(seq, float(frame), "coupling_strength")
            coupling_per_event[i] = np.float32(eaf)

        # repetition score for each event: how many *earlier* events share
        # a coupling_strength within `tol`? (bounded to keep it O(n^2) safe)
        MAX_EVENTS = 4096
        n_events = min(onsets.shape[0], MAX_EVENTS)
        repetition = np.zeros(onsets.shape[0], dtype=np.float32)
        for i in range(n_events):
            hits = 0
            for j in range(i):
                if abs(coupling_per_event[i] - coupling_per_event[j]) <= tol:
                    hits += 1
            # map hits -> [0..1] via a soft saturation
            repetition[i] = np.float32(1.0 - np.exp(-hits / 3.0))

        # ---------- 2. Per-event random amplitude jitter ----------
        rng = np.random.default_rng(seed)
        n_total = onsets.shape[0]
        event_gains = np.zeros(n_total, dtype=np.float32)
        event_t60 = np.zeros(n_total, dtype=np.int64)
        for i in range(n_total):
            # base jitter + boost when the event repeats the same coupling
            jitter = gain_jit * (1.0 + boost * float(repetition[i]))
            # triangular distribution keeps the perturbation in ±jitter
            g = 1.0 + rng.triangular(-jitter, 0.0, jitter)
            event_gains[i] = np.float32(max(0.3, min(2.0, g)))
            # tail = one event length, capped to 8 * sfps/sample_rate so
            # the modulation envelope doesn't smear across the whole file
            event_t60[i] = np.int64(min(lengths[i], int(8.0 * sr / max(sfps, 1.0))))

        y = _event_amplitude_jitter_2d(x, env, onsets.astype(np.int64), event_gains.astype(np.float32), event_t60.astype(np.int64))

        # ---------- 3. Low-rate stochastic LFOs ----------
        # Only apply LFO modulation to continuous contacts (rolling,
        # sliding, scraping) — impacts are already handled above.
        continuous_mask = np.zeros(S, dtype=np.uint8)
        for i in range(S):
            ct = int(ctypes[i])
            if ct in (ContactType.ROLLING.value, ContactType.SLIDING.value, ContactType.SCRAPING.value, ContactType.MIXED.value):
                continuous_mask[i] = 1
        if not continuous_mask.any():
            return y

        cache_key = (S, seed, int(sr), int(round(sfps * 1000)))
        if cache_key in self._lfo_cache:
            lfo_a, lfo_b = self._lfo_cache[cache_key]
        else:
            lfo_a = _build_lfos(S, sfps, sr, rate_hz=lfo_rate, depth=1.0, seed=seed)
            lfo_b = _build_lfos(S, sfps, sr, rate_hz=lfo_rate * 0.7, depth=1.0, seed=seed + 17)
            self._lfo_cache[cache_key] = (lfo_a, lfo_b)
            if len(self._lfo_cache) > 8:
                self._lfo_cache.pop(next(iter(self._lfo_cache)))

        # Scale LFOs per-sample: continuous regions get the full depth,
        # impulse regions get 0.
        # `repetition_local` boosts the LFO where the force recovers
        # the same coupling_strength (so the loop feel is broken there).
        repetition_local = np.zeros(S, dtype=np.float32)
        for i, onset in enumerate(onsets):
            if repetition[i] <= 0.0:
                continue
            length = int(lengths[i])
            end = min(S, onset + length)
            if end > onset:
                repetition_local[onset:end] = np.maximum(repetition_local[onset:end], repetition[i])

        depth_a = lfo_depth * (1.0 + boost * repetition_local)
        depth_b = lfo_delay * (1.0 + boost * repetition_local)
        # Only continuous-contact samples contribute
        for i in range(S):
            if continuous_mask[i] == 0:
                depth_a[i] = 0.0
                depth_b[i] = 0.0

        return _stochastic_modulation_2d(y, (lfo_a * depth_a).astype(np.float32), (lfo_b * depth_b).astype(np.float32), np.float32(1.0))
