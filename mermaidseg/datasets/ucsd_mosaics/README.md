# UCSD Mosaics

PyTorch dataset and HuggingFace-mirror tour for the **UCSD Mosaics**
dense semantic segmentation dataset: 512x512 patches sliced from large-area
coral reef mosaics gathered across 16 dive sites in Palau and the Northern
Line Islands (Palmyra, Fanning, Kingman, Jarvis), with single-channel uint8
segmentation masks over 34 named benthic classes.

Pixel value `0` is the *unlabeled / unidentified* ignore label (it absorbs
the source legend's `Unidentified` row, raw mosaic ID `34`); it is **not**
itself counted as one of the 34 classes.

## Provenance and history

This dataset has accumulated three layers of processing on top of the
original mosaics. We mirror the final, cleaned variant. Each layer must be
cited (see [Citations](#citations)).

### 1. Edwards et al. 2017 — original large-area mosaics

The mosaics were first released as the supplementary
[MOESM5 PDF](https://static-content.springer.com/esm/art%3A10.1007%2Fs00338-017-1624-3/MediaObjects/338_2017_1624_MOESM5_ESM.pdf)
to Edwards et al. 2017. That PDF release is **not directly usable** as
training data:

- The mosaics and their colored segmentation masks are embedded as
  rasterized JPEG images inside the PDF, so each pixel of the mask carries
  some JPEG color noise.
- Multiple distinct classes in the legend share the same display color,
  which makes a strict color-to-class reverse mapping ambiguous even before
  the compression artifacts are considered.
- We attempted to recover the labels anyway by finding connected
  components in the mask (they are separated by near-black borders that
  survive compression), taking a majority-vote color per component, and
  doing a nearest-neighbor lookup against the legend. That still failed:
  some legend colors never appear in the masks, and vice versa, so
  legend and mask color spaces are not consistent with each other.

### 2. Alonso et al. 2019 (CoralSeg) — clean labels + 512x512 patches

Alonso et al. ran into the same problems and, by their own route, produced
a consistent label set and pixel-accurate masks for the same mosaics, then
sliced everything into 512x512 patches for their study. They published
*train* and *test* patch splits — but the *validation* split appears to
have been lost. This means the patches can no longer be perfectly stitched
back together into the original square mosaics; some patches are missing.

### 3. Raine et al. 2024 — GT-Clean variant (what we use)

A few pixels in the top rows of the Alonso patch masks were corrupted.
Raine et al. (Human-in-the-loop segmentation of multi-species coral
imagery) drop those corrupted pixels and call the resulting cleaned-up
release **GT-Clean**. This is the version we ship and consume.

## HuggingFace mirror

We re-upload the GT-Clean release to HuggingFace at
[`josauder/UCSD-mosaics-mirror`](https://huggingface.co/datasets/josauder/UCSD-mosaics-mirror)
so it can be consumed directly via the `datasets` library. The source data
and methodology are unchanged. See
[`nbs/ucsd_mosaics_EDA.ipynb`](nbs/ucsd_mosaics_EDA.ipynb) for a hands-on
tour that instantiates `UCSDMosaicsDataset`, summarises per-site patch
counts, renders one sample image + colored mask per site, plots class pixel
frequencies, and sanity-checks `dataset[idx]`.

### Layout on HuggingFace

- One **HuggingFace split per dive site** (16 splits total) under the
  `default` config. The split name is the lowercased site ID (for example,
  `PALWave13` becomes `palwave13`).
- Each row carries:
  - `image`: PIL RGB image, 512x512.
  - `label`: PIL `L` image, 512x512, values in `0..34` (`0` = ignore).
  - `filename`: original GT-Clean patch filename (e.g.
    `FR3_1024_1536_1024_1536.png`).
  - `site`: site ID in original case (e.g. `PALWave13`).
  - `original_split`: `"train"` or `"test"` from the source GT-Clean
    release.
- A top-level `classes.json` lists the 34 named classes with `id`,
  `raw_legend_id`, `name`, `description` (morphology), and `color_rgb`.

### Sites

| Site      | HF split   | Patches | of which train | of which test |
| --------- | ---------- | ------- | -------------- | ------------- |
| FR3       | fr3        | 220     | 184            | 36            |
| FR5       | fr5        | 262     | 225            | 37            |
| FR7       | fr7        | 415     | 353            | 62            |
| FR9       | fr9        | 310     | 258            | 52            |
| PAL132    | pal132     | 323     | 274            | 49            |
| PAL239    | pal239     | 292     | 244            | 48            |
| PAL36     | pal36      | 259     | 221            | 38            |
| PAL69     | pal69      | 304     | 269            | 35            |
| PALStrawn | palstrawn  | 477     | 413            | 64            |
| PALWave13 | palwave13  | 314     | 272            | 42            |
| PALWave14 | palwave14  | 318     | 275            | 43            |
| PALWave37 | palwave37  | 297     | 251            | 46            |
| PALWave38 | palwave38  | 242     | 198            | 44            |
| PALWave39 | palwave39  | 254     | 217            | 37            |
| PALWave4  | palwave4   | 145     | 126            | 19            |
| PALWave40 | palwave40  | 238     | 194            | 44            |
| **Total** |            | **4670**| **3974**       | **696**       |

### Classes

Label masks are single-channel uint8 PNGs with values in `0..34`. Pixel
value `0` is the *unlabeled / unidentified* ignore label and is not listed
below. The 34 named classes use IDs `1..34` and are taken from the source
`legend.csv`, skipping the catch-all `Unidentified` row (raw `mosaic ID =
34`), which is folded into the ignore label. The RGB colors below are the
BGR palette from the source README converted to RGB.

| Id | Name                     | Morphology            | Color (RGB)             |
| -- | ------------------------ | --------------------- | ----------------------- |
| 1  | Acropora (branching)     | Branching             | #9F12A7 (159, 18, 167)  |
| 2  | Acropora (corymbose)     | Corymbose             | #5C1BB4 (92, 27, 180)   |
| 3  | Acropora (plating)       | Tabular               | #E98B68 (233, 139, 104) |
| 4  | Astreopora myriophthalma | Sub_massive           | #87C631 (135, 198, 49)  |
| 5  | Clavularia               | Fleshy_invert         | #1ACF62 (26, 207, 98)   |
| 6  | Corallimorph             | Corallimorph          | #85D076 (133, 208, 118) |
| 7  | Dictyosphaeria           | Encrusting_macroalgae | #5A769E (90, 118, 158)  |
| 8  | Favia matthai            | Massive               | #A6480C (166, 72, 12)   |
| 9  | Favia stelligera         | Sub_massive           | #EE4F45 (238, 79, 69)   |
| 10 | Favites (encrusting)     | Encrusting            | #31C351 (49, 195, 81)   |
| 11 | Favites (submassive)     | Sub_massive           | #34ECDD (52, 236, 221)  |
| 12 | Fungia                   | Free_living           | #DEC8A0 (222, 200, 160) |
| 13 | Halimeda                 | Erect_macroalgae      | #D83FFF (216, 63, 255)  |
| 14 | Halomitra pileus         | Free_living           | #075E10 (7, 94, 16)     |
| 15 | Hydnophora exesa         | Plating               | #402FE2 (64, 47, 226)   |
| 16 | Hydnophora microconos    | Massive               | #056CB7 (5, 108, 183)   |
| 17 | Leptastrea               | Encrusting            | #C1FC37 (193, 252, 55)  |
| 18 | Lobophyllia              | Massive               | #C49A93 (196, 154, 147) |
| 19 | Montastrea curta         | Massive               | #A54EE9 (165, 78, 233)  |
| 20 | Montipora (encrusting)   | Encrusting            | #5F196C (95, 25, 108)   |
| 21 | Montipora (plating)      | Plating               | #2EDDB8 (46, 221, 184)  |
| 22 | Pasmmocora               | Encrusting            | #91CD36 (145, 205, 54)  |
| 23 | Pavona (submassive)      | Sub_massive           | #D2650E (210, 101, 14)  |
| 24 | Pavona varians           | Encrusting            | #E6E8C7 (230, 232, 199) |
| 25 | Platygyra                | Massive               | #670A42 (103, 10, 66)   |
| 26 | Pocillopora              | Corymbose             | #3BE4A1 (59, 228, 161)  |
| 27 | Pocillopora eydouxi      | Corymbose             | #68026C (104, 2, 108)   |
| 28 | Porites (massive)        | Massive               | #7F310D (127, 49, 13)   |
| 29 | Porites rus              | Massive               | #2663BA (38, 99, 186)   |
| 30 | Porites superfusa        | Encrusting            | #F68C61 (246, 140, 97)  |
| 31 | Soft coral               | Soft                  | #CA722C (202, 114, 44)  |
| 32 | Stylophora pistillata    | Corymbose             | #761F24 (118, 31, 36)   |
| 33 | Turbinaria reniformis    | Plating               | #8F4D92 (143, 77, 146)  |
| 34 | Zooanthid                | Fleshy_invert         | #0E64BC (14, 100, 188)  |

## Using the dataset class

```python
from mermaidseg.datasets import UCSDMosaicsDataset

# Hold out two sites for validation; train on the rest, train half only.
train_ds = UCSDMosaicsDataset(
    blacklist_sites=["palwave4", "palwave40"],
    whitelist_original_splits=["train"],
)
val_ds = UCSDMosaicsDataset(
    whitelist_sites=["palwave4", "palwave40"],
)

image, source_mask = train_ds[0]   # (H, W, 3) uint8 image, (H, W) int64 mask
```

A few API notes:

- Site IDs passed to `whitelist_sites` / `blacklist_sites` are the
  **lowercased HF split names** (e.g. `"fr3"`, `"palwave13"`), not the
  camel-cased `site` column values.
- `whitelist_original_splits` / `blacklist_original_splits` filter on the
  source GT-Clean `original_split` field (`"train"` or `"test"`).
- `class_subset` accepts class names from `classes.json`; everything else
  (including the ignore label) is remapped to `0` background.
- `SOURCE_NAME = "ucsd_mosaics"` so the dataset plugs straight into
  [`SourceLabelRegistry`](../../dataset_reconciliation/registry.py); the
  provider-side source-to-MERMAID mapping is fetched from
  `https://api.datamermaid.org/v1/classification/labelmappings/?provider=UCSD%20Mosaics`
  via
  [`fetch_ucsd_mosaics_to_mermaid()`](../../dataset_reconciliation/label_mapping.py).
  Until that provider is populated on the MERMAID side, the mapping is
  empty and every UCSD Mosaics label collapses to background at training
  time (same fallback as `moorea_labeled_corals`, `pacific_labeled_corals`,
  and `benthos_yuval`).

## License and usage notice

The source dataset is distributed for **research purposes only** and
should not be redistributed beyond research use. By using this dataset
(directly or through this mirror) you agree to the same terms and to cite
all of the works listed below.

## Citations

Please cite **all three** of the following works when using this dataset:

Original large-area mosaics:

```bibtex
@article{edwards2017largearea,
  title   = {Large-area imaging reveals biologically driven non-random spatial patterns of corals at a remote reef},
  author  = {Edwards, Clinton B. and others},
  journal = {Coral Reefs},
  volume  = {36},
  year    = {2017}
}
```

Patch splitting and dense annotations:

```bibtex
@article{alonso2019coralseg,
  title   = {CoralSeg: Learning coral segmentation from sparse annotations},
  author  = {Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and Murillo, Ana C.},
  journal = {Journal of Field Robotics},
  volume  = {36},
  year    = {2019}
}
```

GT-Clean variant (corrupted-mask removal) shipped here:

```bibtex
@inproceedings{raine2024human,
  title     = {Human-in-the-loop segmentation of multi-species coral imagery},
  author    = {Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Sunderhauf, Niko and Fischer, Tobias},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2723--2732},
  year      = {2024}
}
```

## Layout

```
ucsd_mosaics/
├── README.md                       # this file
├── __init__.py                     # re-exports UCSDMosaicsDataset
├── ucsd_mosaics_dataset.py         # PyTorch dataset (HuggingFace-backed)
└── nbs/
    └── ucsd_mosaics_EDA.ipynb      # HF-mirror EDA: class table, per-site counts,
                                    # sample renders, class frequencies, sanity check
```
