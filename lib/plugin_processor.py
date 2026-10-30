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
BaseProcessor implementation that wraps an arbitrary plugin hosted
through `minihost`.

Parameters
----------
path        : str   filesystem path to plugin binary/bundle
format      : str   "ladspa" | "lv2" | "vst2" | "vst3" | "au"
plugin_uri  : str, optional  LV2 URI or VST3 class ID
sample_rate : int   default 48000, but must match the pipeline's sr
block_size  : int   plugin block size hint (default 512)
parameter_map : dict, optional
    Maps plugin parameter indices or names to input-channel indices.
    Example: {0: 0, 1: 1}  (param idx -> channel in x)

    If None, no parameters are driven from the audio; the plugin runs
    with its saved/default parameter state.

channel_map : list[list[int]], optional
    Channel routing from input x to plugin audio inputs.
    If None, channels 0..N-1 of x map to plugin inputs 0..N-1.
    Output channels are always returned as-is.

midi_key : str, default "midi"
    Side-channel key under which MIDI events are read from
    `props.custom`. Set to `None` to disable MIDI.

auto_bypass_when_silent : bool, default True
    Skip processing when the input block is silent (all zeros).
    Falls back to identity output, which matches pbrAudio's
    chainable processor contract.

latency_compensation : bool, default True
    If the plugin reports latency, pad the output tail so the
    downstream DAG sees a length-preserving signal.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.groups import GENERIC
from ..core.processor import BaseProcessor, register_processor
from .midi_events import _to_minihost_input_events

try:
    from ..lib.plugin_host import PluginHost, HostNotAvailable
    _HOST_IMPORTED = True
except Exception:  # pragma: no cover
    _HOST_IMPORTED = False
    PluginHost = None
    HostNotAvailable = RuntimeError

log = logging.getLogger("pbrAudioSignals.plugin")


def _is_silent_block(x: np.ndarray) -> bool:
    if x is None or x.size == 0:
        return True
    return bool(np.max(np.abs(x)) < 1e-9)


class _PluginInstanceWrapper:
    """
    A thin adapter so we can adapt to different minihost builds.

    Methods we probe (in this order):
        - `process_block(midi_in, audio_out, audio_in)`      (most common)
        - `process(midi_in, audio_out, audio_in)`
        - `process(audio_in)`                              (AUPreset style)
    """

    def __init__(self, instance, num_in: int, num_out: int, block_size: int):
        self.inst = instance
        self.num_in = num_in
        self.num_out = num_out
        self.block_size = max(1, int(block_size))
        self._call = self._detect_call()

    def _detect_call(self):
        for name in ("process_block", "process", "run"):
            fn = getattr(self.inst, name, None)
            if fn is not None:
                return (name, fn)
        raise RuntimeError("Plugin instance exposes no process/process_block/run method")

    # ------------------------------------------------------------------
    def process(self, audio_in: np.ndarray, midi: Any) -> np.ndarray:
        """
        audio_in : (C_in, S) float32
        midi     : any accepted by midi_events
        returns  : (C_out, S) float32
        """
        C_in, S = audio_in.shape
        if C_in < self.num_in:
            padded = np.zeros((self.num_in, S), dtype=np.float32)
            padded[:C_in] = audio_in
            audio_in = padded
        elif C_in > self.num_in:
            audio_in = audio_in[:self.num_in]

        out = np.zeros((self.num_out, S), dtype=np.float32)

        midi_events = []
        if midi is not None:
            try:
                midi_events = _to_min_minihost_input_events(midi, num_frames=S, block_start=0)
            except Exception as e:
                log.warning("MIDI parsing failed: %s", e)

        name, fn = self._call

        if name in ("process_block", "process"):
            try:
                fn(midi_events, out, audio_in)
            except TypeError:
                # Variant: fn(out, audio_in, midi_in)
                try:
                    fn(out, audio_in, midi_events)
                except TypeError:
                    # Variant: fn(audio_in, audio_out)
                    fn(audio_in, out)
        elif name == "run":
            # Some hosts use run(n_frames, audio_in, audio_out, midi_in)
            if midi_events:
                fn(S, audio_in, out, midi_events)
            else:
                fn(S, audio_in, out)

        return out

    # ------------------------------------------------------------------
    def set_parameter(self, idx: int, value: float):
        fn = getattr(self.inst, "set_parameter", None) or getattr(self.inst, "set_param", None)
        if fn is not None:
            try:
                fn(idx, float(value))
            except Exception:
                log.debug("set_parameter failed for idx=%d", idx)

    def get_latency(self) -> int:
        fn = getattr(self.instinst, "get_latency", None)
        if fn is not None:
            try:
                return int(fn())
            except Exception:
                pass
        return 0


# ----------------------------------------------------------------------

@register_processor("plugin", track_group=GENERIC)
class PluginProcessor(BaseProcessor):
    """
    Host a LADSPA / LV2 / VST2 / VST3 / AU plugin as a DAG processor.

    Chains are stored as plain dicts in `AudioDataType.chains`; the
    plugin wrapper reads its config from there. Load the plugin lazily,
    once, on first `process()` call.
    """

#    required_side = {"plugin_params", "param_values"}
    accepts_any_group = True

    def __init__(self,
                 path: str,
                 format: str = "lv2",                 # noqa: A002 - API name
                 plugin_uri: Optional[str] = None,
                 sample_rate: int = 48000,
                 block_size: int = 512,
                 parameter_map: Optional[Dict[Any, Any]] = None,
                 channel_map: Optional[List[List[int]]] = None,
                 midi_key: Optional[str] = "midi",
                 auto_bypass_when_silent: bool = True,
                 latency_compensation: bool = True,
                 **kw):
        super().__init__(
            path=path,
            format=format,
            plugin_uri=plugin_uri,
            sample_rate=sample_rate,
            block_size=block_size,
            parameter_map=parameter_map or {},
            channel_map=channel_map,
            midi_key=midi_key,
            auto_bypass_when_silent=auto_bypass_when_silent,
            latency_compensation=latency_compensation,
            **kw,
        )
        # Lazy state
        self._wrapper: Optional[_PluginInstanceWrapper] = None
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._applied_params: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._wrapper is not None:
            return True
        with self._load_lock:
            if self._wrapper is not None:
                return True
            if self._load_error is not None:
                return False

            if not _HOST_IMPORTED:
                self._load_error = "minihost is not available"
                return False

            try:
                host = PluginHost.instance()
                path = self.params["path"]
                fmt = self.params["format"]
                uri = self.params.get("plugin_uri")
                sr = int(self.params.get("sample_rate", 48000))
                bs = int(self.params.get("block_size", 512))

                inst = host.create_instance(path, fmt, uri,
                                            sample_rate=sr, block_size=bs)

                # Determine I/O channel counts. Plugins differ; try
                # to query them, fall back to 2 / 2.
                num_in = getattr(inst, "num_inputs", None) or getattr(inst, "inputs", None) or 2
                num_out = getattr(inst, "num_outputs", None) or getattr(inst, "outputs", None) or 2
                try:
                    num_in = int(num_in)
                    num_out = int(num_out)
                except Exception:
                    num_in, num_out = 2, 2

                # A mono plugin may report 1 in / 1 out: keep that.
                if num_in <= 0:
                    num_in = 2
                if num_out <= 0:
                    num_out = num_in

                self._wrapper = _PluginInstanceWrapper(inst, num_in, num_out, bs)

                # Apply any parameters that were cached before load.
                for k, v in self.params.get("parameter_map", {}).items():
                    self._wrapper.set_parameter(int(k), float(v))

                return True
            except HostNotAvailable as e:
                self._load_error = str(e)
                log.warning("Plugin not loaded: %s", e)
                return False
            except Exception as e:  # pragma: no cover
                self._load_error = str(e)
                log.exception("Unexpected error loading plugin")
                return False

    # ------------------------------------------------------------------
    def _process(self, x: np.ndarray, props) -> np.ndarray:
        if not self._ensure_loaded():
            # Passthrough keeps the chain alive if a plugin is missing.
            return x

        C, S = x.shape

        # Optional silence bypass.
        if self.params.get("auto_bypass_when_silent", True):
            if _is_silent_block(x):
                # Keep any MIDI tickled by force events? Not needed here.
                return np.zeros((self._wrapper.num_out, S), dtype=np.float32)

        # Resolve optional parameter side-channel.
        # `parameter_map` maps param_idx -> input channel index.
        # The caller can also override per-evaluation via props.custom['params'].
        self._apply_parameter_updates(props)

        # MIDI from side-channel.
        midi_key = self.params.get("midi_key")
        midi = None
        if midi_key and props is not None:
            custom = getattr(props, "custom", None) or {}
            midi = custom.get(midi_key)

        out = self._wrapper.process(x, midi)

        # Latency compensation: pad the tail if needed.
        if self.params.get("latency_compensation", True) and self._wrapper is not None:
            lat = self._wrapper.get_latency()
            if lat > 0 and S > 0:
                if out.shape[1] >= S:
                    out = out[:, :S]
                else:
                    pad = S - out.shape[1]
                    out = np.pad(out, ((0, 0), (0, pad)))
                # Delay the signal by `lat` samples (prepend zeros).
                if lat < S:
                    out = np.pad(out[:, :-lat], ((0, 0), (lat, 0)))
                else:
                    out = np.zeros_like(out)

        return out

    # ------------------------------------------------------------------
    def _apply_parameter_updates(self, props):
        """
        Parameter automation driven either by:
          - props.custom['plugin_params'] = {idx_or_name: value}, or
          - parameter_map: {param_idx: input_channel_idx} paired with
            props.custom['param_values'] = np.ndarray indexed by channel.
        """
        if self._wrapper is None or props is None:
            return

        custom = getattr(props, "custom", None) or {}

        # 1) Direct param dict
        direct = custom.get("plugin_params")
        if isinstance(direct, dict):
            for k, v in direct.items():
                self._wrapper.set_parameter(int(k), float(v))

        # 2) Channel-driven parameters
        pmap = self.params.get("parameter_map") or {}
        if not pmap:
            return
        pvals = custom.get("param_values")
        if pvals is None:
            return
        pvals = np.asarray(pvals, dtype=np.float32).ravel()

        for param_idx, ch_idx in pmap.items():
            if 0 <= ch_idx < pvals.shape[0]:
                value = float(pvals[ch_idx])
                if abs(value - self._applied_params.get(int(param_idx), -1e9)) > 1e-6:
                    self._wrapper.set_parameter(int(param_idx), value)
                    self._applied_params[int(param_idx)] = value
