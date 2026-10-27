# Selected MLX-Audio enhancement inference components

Source: https://github.com/Blaizzy/mlx-audio
Version: PyPI mlx-audio 0.6.2, installed and tested in this project.
License: MIT; the upstream license and copyright notice are retained in LICENSE.

This is a deliberately limited copy of the DeepFilterNet and MossFormer2 enhancement
implementations and their inference dependencies. It is not the full MLX-Audio SDK.
The original directory layout and model math are retained. Absolute imports were
namespaced under textuplink._vendor to avoid colliding with installed packages.
Package initializers omit unrelated TTS/STT loaders. MossFormer's shared audio-file
loader is replaced by a small local librosa wrapper; array inference is unchanged.

This subset requires MLX, NumPy, Hugging Face Hub and local audio libraries, all already
included in requirements/app.txt. It does not depend on the SDK's Transformers 5
features. Do not replace it with the full SDK without checking Qwen compatibility.
Changes to upstream model code should be checked against reference waveforms.
