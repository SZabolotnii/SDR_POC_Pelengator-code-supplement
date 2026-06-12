# Tampere drone RF recordings (not redistributed here)

The raw I/Q recordings are **not** included in this repository because of
their size. Download them from the open Zenodo record:

> Vuorenmaa, M., Marin, J., Heino, M., Turunen, M. & Riihonen, T.
> "Radio-Frequency Control and Video Signal Recordings of Drones".
> Zenodo, 2020. DOI: [10.5281/zenodo.4264467](https://doi.org/10.5281/zenodo.4264467)

The recordings are interleaved little-endian int16 I/Q samples at
120 MS/s (2.4 GHz band). Place (or symlink) the five controller
recordings used by the experiments in this directory under the
following names:

| Expected filename                  | Zenodo recording (2.4 GHz controller) |
|------------------------------------|----------------------------------------|
| `DJI_mavic_pro_2G.bin`             | DJI Mavic Pro                           |
| `DJI_inspire_2_2G.bin`             | DJI Inspire 2                           |
| `DJI_phantom_4_2G.bin`             | DJI Phantom 4                           |
| `Parrot_disco_2G.bin`              | Parrot Disco                            |
| `Yuneec_typhoon_h_2G_1of2.bin`     | Yuneec Typhoon H (part 1 of 2)          |

Only the first ~0.5 s of each recording is consumed by the end-to-end
proxy (`duration_s: 0.5` in `experiments/poc_pelengator/config.yaml`);
the fingerprint training subsamples 1000 windows × 4096 samples per
recording.
