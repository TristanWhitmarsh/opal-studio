# Models

Opal Studio uses two kinds of model weights. Neither is included in the Python
package; both are downloaded the first time they are needed.

## Opal Studio's own models

Trained for this project and published on the
[`models-v1` release](https://github.com/TristanWhitmarsh/opal-studio/releases/tag/models-v1)
of this repository, under the **MIT licence** — the same as the code. Each archive
contains a copy of the licence.

| Model | Used by | Asset |
|---|---|---|
| Cell positivity CNN | Cell positivity → AI | `cellpos.zip` |
| StarDist General | Segmentation → StarDist | `stardist-General.zip` |
| StarDist IMC | Segmentation → StarDist | `stardist-IMC.zip` |
| Cellpose General | Segmentation → Cellpose | `cellpose-General.zip` |
| Cellpose IMC | Segmentation → Cellpose | `cellpose-IMC.zip` |
| InstanSeg IMC | Segmentation → InstanSeg | `instanseg-IMC.zip` |
| Mesmer IMC | Segmentation → Mesmer | `mesmer-IMC.zip` |

To publish a new version: update the models under `opal_studio/models/`, run
`python tools/build_model_release.py`, create a release with a new tag, upload the
zips unchanged, and update `RELEASE_TAG` in `opal_studio/model_store.py`.

## Models that come with the segmentation libraries

These belong to their authors and are **not redistributed** by Opal Studio. Each
library downloads them from its own source into its own cache, under its own licence.

| Library | Models | Licence | Cached in |
|---|---|---|---|
| [StarDist](https://github.com/stardist/stardist) | `2D_versatile_fluo`, `2D_paper_dsb2018`, `2D_versatile_he` | BSD-3-Clause (code) | `~/.keras/models/StarDist2D/` |
| [Cellpose](https://github.com/MouseLand/cellpose) | `nuclei`, `cyto`, `cyto2`, `cyto3` | BSD-3-Clause | `~/.cellpose/models/` |
| [InstanSeg](https://github.com/instanseg/instanseg) | `fluorescence_nuclei_and_cells`, `brightfield_nuclei` | Apache-2.0 | InstanSeg's package directory |
| [DeepCell](https://github.com/vanvalenlab/deepcell-tf) | Mesmer (default) | Modified Apache 2.0 — **non-commercial, academic use** | needs a DeepCell access token |

InstanSeg's `single_channel_nuclei` is an exception in *how* it is fetched, not in
where from: it is published in InstanSeg's releases (`instanseg_models_v0.1.2`) but
missing from the model index of the installed InstanSeg version, so InstanSeg cannot
fetch it by name. Opal Studio downloads it directly from InstanSeg's release into its
own models directory. Apache-2.0.

**Citations.** InstanSeg asks that you cite it when you publish results from its
models — [arXiv:2408.15954](https://doi.org/10.48550/arXiv.2408.15954) for brightfield
nuclei and [bioRxiv 2024.09.04.611150](https://doi.org/10.1101/2024.09.04.611150) for
fluorescence. The StarDist, Cellpose and DeepCell papers apply likewise.

**StarDist `2D_versatile_he`.** The model archive carries no licence of its own. It
was trained on the MoNuSeg and TNBC datasets, whose terms are worth checking before
commercial use.

## Where models are kept

In `opal_studio/models/` inside the installed package — under `site-packages` for a
pip install — next to the code, so nothing is left elsewhere on the machine.
Deleting the `opal_studio` folder deletes its models too.

Set `OPAL_STUDIO_MODELS` to use a different directory — shared storage on a cluster,
for instance. **File → Download All Models** fetches everything at once, for use on a
machine without internet access afterwards.

pip only removes files it installed itself, so downloaded models survive
`pip install --force-reinstall`, and `pip uninstall` leaves the `models/` folder
behind in `site-packages/opal_studio/`. To fetch a model again, delete its folder; it
is downloaded afresh the next time it is used.

A download is written to a temporary file and moved into place only once it has
arrived in full and every file in the archive has passed its CRC check, so an
interrupted download is retried rather than left behind as a broken model.

## Using your own models

Place a model folder under `<models>/<engine>/<name>/` and it is offered in that
engine's model list:

- StarDist — a folder with `config.json` and `weights_best.h5`
- Cellpose — a folder containing the weights file
- InstanSeg — a folder with `model_weights.pth` and `experiment_log.csv`, or `instanseg.pt`
- Mesmer — a folder containing a `.keras` file
- Omnipose — the weights file itself, under `<models>/omnipose/`
