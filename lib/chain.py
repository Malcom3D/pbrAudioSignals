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
from typing import Any, Dict, Iterable, List, Sequence, Optional

from ..core.processor import BaseProcessor, register_processor, get_processor
from ..core.groups import GENERIC

# chain factory
def build_chain(spec: List[Dict[str, Any]], track_group: Optional[str] = None) -> BaseProcessor:
    """
    Build a ChainProcessor from a declarative spec.

    Spec format (list of step dicts):
        [
          {"type": "dc_blocker", "params": {"alpha": 0.995}},
          {"type": "adaptive_noise_gate",
           "params": {"threshold_db": -55, "force_weight": 0.5}},
          {"type": "force_weighted_gain",
           "params": {"target_rms": 0.15}}
        ]

    Unknown types raise KeyError. Empty spec returns a pass-through ChainProcessor.
    """
    chain = ChainProcessor()
    for step in spec or []:
        t = step["type"]
        cls = get_processor(t, track_group=track_group)
        proc = cls(**(step.get("params") or {}))
        chain.add(proc)
    return chain

@register_processor("chain", track_group=GENERIC)
class ChainProcessor(BaseProcessor):
    """
    Compose multiple processors into one.

    Params
    ------
    processors : Sequence[BaseProcessor]
        Processors applied in order (first -> last).
    """

    def __init__(self, processors: Sequence[BaseProcessor] = (), **kw):
        super().__init__(processors=list(processors), **kw)

    def add(self, proc: BaseProcessor) -> "ChainProcessor":
        self.params["processors"].append(proc)
        return self

    def _process(self, x, props):
        y = x
        for proc in self.params["processors"]:
            y = proc.process(y, props)
        return y

