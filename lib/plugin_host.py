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
Singleton wrapper around `minihost` that:

- Loads plugin descriptors once per (path, type) pair
- Keeps a plugin descriptor cache to avoid re-scanning
- Creates plugin *instances* per processor wrapper
- Serialises access under a soft lock (minihost PluginInstance is
  not thread-safe in general; the DAG scheduler is single-threaded
  in practice, but we still guard against accidental parallel use)
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Local imports at top-level; fails clearly if minihost is missing.
try:
    import minihost
    from minihost import (
        PluginType,
        descriptor,
        load_plugin,
        PluginInstance,
    )
    _HAS_HOST = True
except ImportError as _e:  # pragma: no cover
    _HAS_HOST = False
    _IMPORT_ERROR = str(_e)
    # Provide dummies so the module imports even if minihost is absent.
    minihost = None
    PluginType = None
    descriptor = None
    load_plugin = None
    PluginInstance = None


class HostNotAvailable(RuntimeError):
    """Raised when minihost is not importable."""


# Possible names for plugin type enum in various minihost builds.
def _resolve_plugin_type(fmt: str):
    if not _HAS_HOST:
        return None
    fmt = fmt.lower()
    # Newer minihost uses PluginType enum; older uses strings.
    if hasattr(PluginType, "LADSPA"):
        mapping = {
            "ladspa": PluginType.LADSPA,
            "lv2":    PluginType.LV2,
            "vst2":   PluginType.VST2,
            "vst3":   PluginType.VST3,
            "au":     PluginType.AU,
        }
        if fmt in mapping:
            return mapping[fmt]
    # Fallback: string-typed
    return fmt


@dataclass
class _DescriptorKey:
    path: str
    fmt: str
    plugin_uri: Optional[str] = None  # for LV2 or VST3 class IDs


class PluginHost:
    """
    Process-wide singleton that owns the minihost descriptor cache.

    Use `PluginHost.instance()` to get it.
    """

    _instance: Optional["PluginHost"] = None
    _install_lock = threading.Lock()

    def __init__(self):
        if not _HAS_HOST:
            raise HostNotAvailable(
                f"`minihost` is not available: {_IMPORT_ERROR}. "
                f"Install with: pip install minihost[numpy]"
            )
        self._descriptor_cache: Dict[_DescriptorKey, Any] = {}
        self._lock = threading.RLock()
        self._initialized = False

    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "PluginHost":
        with cls._install_lock:
            if cls._instance is None:
                cls._instance = PluginHost()
            return cls._instance

    # ------------------------------------------------------------------
    def _ensure_init(self):
        """Call minihost's global init once (some builds require it)."""
        if self._initialized:
            return
        init_fn = getattr(minihost, "initialize", None) or getattr(minihost, "init", None)
        if init_fn is not None:
            try:
                init_fn()
            except Exception:
                pass  # most builds are lazy; init may be a no-op
        self._initialized = True

    # ------------------------------------------------------------------
    def get_descriptor(self, path: str, fmt: str,
                       plugin_uri: Optional[str] = None):
        """
        Get (or create) a plugin descriptor.

        Parameters
        ----------
        path      : str   Path to the plugin binary (or bundle dir for AU/LV2)
        fmt       : str   "ladspa" | "lv2" | "vst2" | "vst3" | "au"
        plugin_uri: str   For LV2 (URI) or multi-plugin bundles (VST3 class ID)
        """
        self._ensure_init()

        path = os.path.abspath(path)
        ptype = _resolve_plugin_type(fmt)
        key = _DescriptorKey(path=path, fmt=fmt.lower(), plugin_uri=plugin_uri)

        with self._lock:
            if key in self._descriptor_cache:
                return self._descriptor_cache[key]

            # Try a few call signatures depending on the minihost version.
            desc = self._open_descriptor(path, ptype, plugin_uri)
            self._descriptor_cache[key] = desc
            return desc

    # ------------------------------------------------------------------
    def _open_descriptor(self, path: str, ptype, plugin_uri: Optional[str]):
        """Try different minihost APIs to open a descriptor."""
        # API variant A: descriptor.load(path, PLUGIN_TYPE)
        try:
            if hasattr(descriptor, "load"):
                if plugin_uri is not None:
                    try:
                        return descriptor.load(path, ptype, plugin_uri)
                    except TypeError:
                        pass
                return descriptor.load(path, ptype)
        except Exception:
            pass

        # API variant B: load_plugin(path, ptype)
        try:
            return load_plugin(path, ptype)
        except TypeError:
            pass

        except Exception as e:
            raise RuntimeError(f"Failed to load plugin descriptor for {path}: {e}")

        # API variant C: descriptor.from_path(path)
        if hasattr(descriptor, "from_path"):
            return descriptor.from_path(path)

        raise RuntimeError(
            f"Could not find a working descriptor loader in this minihost build. "
            f"Tried `descriptor.load`, `load_plugin`, `descriptor.from_path`."
        )

    # ------------------------------------------------------------------
    def create_instance(self, path: str, fmt: str,
                        plugin_uri: Optional[str] = None,
                        sample_rate: int = 48000,
                        block_size: int = 512) -> "PluginInstance":
        """
        Instantiate the plugin. Each processor owns one instance.
        """
        desc = self.get_descriptor(path, fmt, plugin_uri)
        # Prefer the modern API `desc.create_instance(sr, block)`,
        # fall back to `PluginInstance(desc, sr, block)`.
        try:
            if hasattr(desc, "create_instance"):
                inst = desc.create_instance(sample_rate, block_size)
            else:
                inst = PluginInstance(desc, sample_rate, block_size)
        except TypeError:
            inst = PluginInstance(desc, sample_rate, block_size)

        self._prepare_instance(inst)
        return inst

    # ------------------------------------------------------------------
    def _prepare_instance(self, inst):
        """Send required state to a freshly-created instance."""
        for meth in ("activate", "setup", "init"):
            fn = getattr(inst, meth, None)
            if fn is not None:
                try:
                    fn()
                except TypeError:
                    try:
                        fn(0)  # some APIs need a channel count
                    except Exception:
                        pass
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def clear(self):
        """Drop all cached descriptors (e.g., dev env plugin reload)."""
        with self._lock:
            self._descriptor_cache.clear()
