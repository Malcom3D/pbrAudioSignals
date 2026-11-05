
# pbrAudioDAW
## [v0.3.0] - 2026-11-04

### Added
- New `pbrAudioDAW` application featuring a Terminal User Interface (TUI), including a REPL, browser state management, command system, and playback controls.
- New audio signal processors for mono modal displacement to acoustic pressure conversion, modal displacement, and audio forces.
- New signal processing and filtering features including adaptive Gaussian processing, ambisonic decoding, DC blockers, envelope shapers, noise gates, normalization, spectral denoisers, temporal smoothing, MIDI events handling, and various force-based processors (smoothing, amplification, compression, equalization, phase alignment, sidechaining, and more).
- New storage engine capabilities for editing metadata on backends and loading custom configuration properties via `StorageEngine._get_props()`.
- New `SynthPostProcess` and `postProcess` classes to describe audio data processing chains and rigid body synth post-processing in storage metadata.
- Track group handling in signal processors to subdivide processors according to DAG node/edge goals.
- bz2 compression and decompression support for classes saved by `KleptoEntityStore`.
- Viewport Panel integration and 3D viewport widget GUI in `pbrAudioBlender`.

### Changed
- Complete code rewrite and architectural refactoring of `pbrAudioStorage`.
- Switched `blosc2Backend` to parallel append and mmapped mode.
- Aligned `rigidBody` and `physicsSolver` postProcess classes with new storage and signal features.

### Fixed
- Fixed signal and ScoreEvent misalignment between physicsSolver and rigidBody.
- Fixed out-of-memory (OOM) issues in `ModalComposer`.
- Fixed post-processing variable naming conventions across modules.
- Fixed typos and errors in storage engines, Blosc2Backend, KleptoEntityStore, rigidBody integration, and signal processors.
- Fixed storage root paths in CollisionExporter and RenderExporter, and indentation issues in Blender property groups.
- Fixed missing dependencies and module import errors (such as missing `zarr` and incorrect module paths).
- Fixed various syntax, indentation, and name errors across the codebase.
