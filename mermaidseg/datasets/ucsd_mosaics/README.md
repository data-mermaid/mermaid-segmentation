# UCSD Mosaics

Dense label masks over 34 classes, in 512x512 patches cut from reef mosaics of 16 dive sites in Palau and the Northern Line Islands. `UCSDMosaicsDataset` in `ucsd_mosaics_dataset.py` reads them from a Hugging Face mirror.

## Source and citation

The data has three layers of processing. Cite all three works:

1. Edwards et al. 2017 released the large-area mosaics as a supplementary PDF ([MOESM5](https://static-content.springer.com/esm/art%3A10.1007%2Fs00338-017-1624-3/MediaObjects/338_2017_1624_MOESM5_ESM.pdf)). The masks in that PDF are JPEG-compressed and some legend colors are shared, so the labels cannot be recovered from it.
2. Alonso et al. 2019 (CoralSeg) made pixel-accurate masks for the same mosaics and cut them into 512x512 patches. They published the train and test patches. The validation patches are not available, so the patches do not cover the full mosaics.
3. Raine et al. 2024 removed corrupted pixels from the top rows of the Alonso masks. This variant, GT-Clean, is the one in the mirror.

```bibtex
@article{edwards2017largearea,
  title   = {Large-area imaging reveals biologically driven non-random spatial patterns of corals at a remote reef},
  author  = {Edwards, Clinton B. and others},
  journal = {Coral Reefs},
  volume  = {36},
  year    = {2017}
}

@article{alonso2019coralseg,
  title   = {CoralSeg: Learning coral segmentation from sparse annotations},
  author  = {Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and Murillo, Ana C.},
  journal = {Journal of Field Robotics},
  volume  = {36},
  year    = {2019}
}

@inproceedings{raine2024human,
  title     = {Human-in-the-loop segmentation of multi-species coral imagery},
  author    = {Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Sunderhauf, Niko and Fischer, Tobias},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2723--2732},
  year      = {2024}
}
```

| Site | HF split | Patches | Train | Test |
|---|---|---|---|---|
| FR3 | `fr3` | 220 | 184 | 36 |
| FR5 | `fr5` | 262 | 225 | 37 |
| FR7 | `fr7` | 415 | 353 | 62 |
| FR9 | `fr9` | 310 | 258 | 52 |
| PAL132 | `pal132` | 323 | 274 | 49 |
| PAL239 | `pal239` | 292 | 244 | 48 |
| PAL36 | `pal36` | 259 | 221 | 38 |
| PAL69 | `pal69` | 304 | 269 | 35 |
| PALStrawn | `palstrawn` | 477 | 413 | 64 |
| PALWave13 | `palwave13` | 314 | 272 | 42 |
| PALWave14 | `palwave14` | 318 | 275 | 43 |
| PALWave37 | `palwave37` | 297 | 251 | 46 |
| PALWave38 | `palwave38` | 242 | 198 | 44 |
| PALWave39 | `palwave39` | 254 | 217 | 37 |
| PALWave4 | `palwave4` | 145 | 126 | 19 |
| PALWave40 | `palwave40` | 238 | 194 | 44 |
| **Total** | | **4670** | **3974** | **696** |

## Licence

Research use only. The source dataset is distributed for research purposes and must not be redistributed beyond research use. Use of the data, directly or through the mirror, carries the same terms and the obligation to cite the three works above. The Hugging Face card gives the licence as `other`, named `research-only`.

## Where the class reads from

The Hugging Face dataset [`josauder/UCSD-mosaics-mirror`](https://huggingface.co/datasets/josauder/UCSD-mosaics-mirror), default configuration:

- One split for each dive site, named by the lowercased site ID (for example `palwave13`). `whitelist_sites` and `blacklist_sites` take these names.
- The class reads the columns `image` (RGB, 512x512), `label` (single channel, 512x512, values 0 to 34) and `original_split` (`train` or `test` in GT-Clean). The rows also have `filename` and `site`.
- `classes.json` at the top of the repo lists the 34 classes with `id`, `raw_legend_id`, `name`, `description` and `color_rgb`.

Label value 0 is the ignore label. It includes the `Unidentified` row of the source legend (raw mosaic ID 34). The class maps it to background.

## How to rebuild

There is no ingestion step. The class downloads the mirror into `HF_HOME` the first time it is built. See [Get the Hugging Face datasets](../../../docs/runbooks/training-datasets.md#get-the-hugging-face-datasets).

The mirror was uploaded by hand. The data and the processing are the GT-Clean release, unchanged. No script in this repository builds or uploads it.
