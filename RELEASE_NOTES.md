# pbrAudioDAW
Welcome to version v0.3.0 of **pbrAudioDAW**! This release introduces major architectural upgrades, powerful new audio processing tools, and a brand-new interface.

* **New Terminal User Interface (TUI):** Explore and control your audio projects directly from the terminal with the newly introduced `pbrAudioDAW` TUI, complete with interactive playback and browser controls.
* **Advanced Audio & Physics Processing:** Added a wide array of new signal processors—including modal displacement-to-pressure converters, ambisonic decoding, noise gating, spectral denoising, and specialized force processors for precise acoustic simulation.
* **Improved Performance & Storage:** Completely overhauled the storage engine with parallel appending and mmapped mode using `blosc2` for faster, more reliable data handling, alongside bz2 compression support.
* **Blender & DAW Integration:** Enhanced 3D viewport widgets and panel integration within Blender, alongside better alignment between physics solvers, rigid bodies, and audio tracks.
* **Bug Fixes:** Numerous fixes addressing memory usage in modal composition, module import errors, and signal-to-score event alignments.
