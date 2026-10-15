# CoralNet label-coverage audit

How CoralNet annotations resolve into the training target space (class subset = 77 classes, `label_roll_up=True`, 836 API mappings, 3418 distinct coralnet_ids).

Annotations that are **not** `in-subset` collapse to background/ignore at training time.

| Split | Annotations | Mapped to a class | In 77-class subset (trained) |
|---|---|---|---|
| val |  781,194 | 697,989 (89.3%) | 528,947 (67.7%) |
| train |  23,662,863 | 20,255,979 (85.6%) | 18,078,177 (76.4%) |

## Top unmapped coralnet_ids by annotation count

These `coralnet_id`s have no MERMAID mapping and are candidates for mapping on the MERMAID side.

| coralnet_id | annotations | val | train |
|---|---|---|---|
| 6911 | 1,797,645 | 48,837 | 1,748,808 |
| 2784 | 117,445 | 0 | 117,445 |
| 1639 | 86,682 | 0 | 86,682 |
| 2787 | 75,015 | 0 | 75,015 |
| 6775 | 61,507 | 0 | 61,507 |
| 9722 | 37,543 | 0 | 37,543 |
| 9015 | 35,879 | 0 | 35,879 |
| 7530 | 28,891 | 0 | 28,891 |
| 5347 | 26,914 | 0 | 26,914 |
| 9278 | 26,031 | 0 | 26,031 |
| 5291 | 24,591 | 11,199 | 13,392 |
| 4196 | 24,562 | 0 | 24,562 |
| 9145 | 16,971 | 0 | 16,971 |
| 9017 | 16,646 | 0 | 16,646 |
| 1741 | 16,133 | 0 | 16,133 |

## Top mapped-but-excluded target classes by annotation count

These map to a valid MERMAID benthic attribute but fall outside the current 77-class subset (even after roll-up) — candidates for a subset expansion (needs taxonomy review).

| target (mapped, lowercased) | annotations |
|---|---|
| other | 1,716,709 |
| unknown | 204,277 |
| other invertebrates | 109,241 |
| bryozoan | 86,583 |
| bivalvia | 67,389 |
| xeniidae | 40,995 |
| palythoa caribaeorum | 24,039 |
| tunicate | 19,119 |
| ascidian | 15,487 |
| hydroid | 9,329 |
| foraminifera | 9,246 |
| rhytisma | 7,099 |
| tape | 5,878 |
| palythoa | 5,484 |
| obscured | 4,687 |
