"""Source-dataset to MERMAID benthic-attribute target-label mappings.

Provides the HTTP fetchers + static dicts that translate a source-space label (CoralNet provider
IDs, Coralscapes 1..39 names, MERMAID benthic-attribute names) into MERMAID benthic-attribute target
names, plus the GPU helper ``source_labels_to_target_labels`` used at training time.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
import torch


def fetch_mermaid_target_labels(
    benthicattributes_url: str = "https://api.datamermaid.org/v1/benthicattributes/",
) -> list[str]:
    """Fetch the canonical MERMAID benthic-attribute label names.

    Returns:
        Alphabetically sorted list of unique benthic-attribute names.
    """
    response = requests.get(benthicattributes_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    records = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        records.extend(data["results"])
    return sorted({rec["name"] for rec in records if rec.get("name") is not None})


def fetch_coralnet_to_mermaid() -> dict[str, str | None]:
    """Load the CoralNet provider label -> MERMAID benthic-attribute mapping from local config.

    Returns a dict keyed by stringified CoralNet provider label; values are the mapped MERMAID
    benthic-attribute name(s) (or ``None`` if the CoralNet label is not yet mapped).
    """
    coralnet_to_mermaid_mapping_temporary_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "coralnet_to_mermaid_mapping_temporary.json"
    )

    with open(coralnet_to_mermaid_mapping_temporary_path) as f:
        return json.load(f)


def fetch_catlin_seaview_to_mermaid() -> dict[str, str]:
    """Return the static Catlin Seaview label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Catlin label name; values are the mapped MERMAID benthic-attribute
    name(s). Labels absent from this map collapse to background at training time via
    :class:`SourceLabelRegistry`.
    """
    return {
        "aarcheri": "aplysina archeri",
        "acervicornis": "acropora cervicornis",
        "acompresa": "amphimedon compressa",
        "acropora corymbose/tabular/plate": "acropora",
        "acropora digitate": "acropora",
        "acroporidae other": "acroporidae",
        "acroporidae branching": "acroporidae",
        "acroporidae hispidose": "acroporidae",
        "acroporidae plate/encrusting": "acroporidae",
        "acroporidae table/corymbose/digitate": "acroporidae",
        "afistularis": "aplysina fistularis",
        "agaricia/undaria": "agaricia",
        "aiolochroia crassa": "aiolochroia crassa",
        "alcyoniidae": "alcyoniidae",
        "algae matrix": "epilithic algal matrix",
        "algae: large visible globules": "turf algae",
        "apalmata": "acropora palmata",
        "aplysina fulva_cauliformis": "aplysina",
        "atubulata": "agelas tubulata",
        "bra: acropora bottlebrush": "acropora",
        "bra: acropora branching": "acropora",
        "bra: other": "other",
        "bleached pocillopora": "bleached coral",
        "branching porites": "porites",
        "branching stylophora": "stylophora",
        "branching bleached": "bleached coral",
        "cca": "crustose coralline algae",
        "calcifying algae: halimeda": "halimeda",
        "calcifying algae: padina": "padina",
        "calcifying macroalgae: padina": "padina",
        "cdelitrix": "cliona delitrix",
        "cnatans": "colpophyllia natans",
        "common lrg alcyoniide": "alcyoniidae",
        "cplicifera": "callyspongia plicifera",
        "crinoids": "crinoid",
        "crown of thorns sea star": "acanthaster planci",
        "crustose coralline algae": "crustose coralline algae",
        "cvaginalis": "callyspongia vaginalis",
        "cyanobacteria": "cyanobacteria",
        "cyanobacteria films": "cyanobacteria",
        "cyanobacteria on rock or other substrates": "cyanobacteria",
        "cyanobacteria smothering dead coral": "cyanobacteria",
        "cyanobacteria smothering rubble": "cyanobacteria",
        "diadema/ehinothix": "sea urchin",
        "dictyota": "dictyota",
        "dlabyrinthiformis": "diploria labyrinthiformis",
        "eam: dead hard coral": "dead coral",
        "echinometra": "echinometra",
        "encrusting cliona spp.": "cliona",
        "encrusting gorgonian": "gorgonia",
        "encrusting sponge": "sponge",
        "epilithic algal matrix": "epilithic algal matrix",
        "epilithic algal matirx smotheting rubble": "epilithic algal matrix",
        "epilithic algal matrix on rock or other substrates": "epilithic algal matrix",
        "epillithic algal matrix": "epilithic algal matrix",
        "erect rhodphyta": "rhodophyta",
        "erect gorgonians": "gorgonia",
        "erect sponge": "sponge",
        "eusmilia fastigiata": "eusmilia fastigiata",
        "filamentous macroalgae": "macroalgae",
        "fine branching nonacroporids seriatopora": "seriatopora",
        "fish": "fish",
        "foliose algae: other": "turf algae",
        "foliose fan shaped algae": "turf algae",
        "foliose feathery algae": "turf algae",
        "foliose strap/ branched algae": "turf algae",
        "foliose sheeting macroalage: (e.g ulva)": "macroalgae",
        "halimeda": "halimeda",
        "hydroids feathery types": "hydroid",
        "individual tunicates": "tunicate",
        "ircinia_massive": "ircinia",
        "leathery macrophyte: other": "macroalgae",
        "leptoseris": "leptoseris",
        "loose substrate: rubble": "rubble",
        "loose substrate: sand": "sand",
        "loose substrate: sediment": "bare substrate",
        "lvariegata": "lobophora variegata",
        "mase: isopora": "isopora",
        "mase: lobophyllia": "lobophyllia",
        "mase: porites": "porites",
        "macroalgae": "macroalgae",
        "macroalgae 1": "macroalgae",
        "macroalgae encrusting red": "macroalgae",
        "macrolagae all genera": "macroalgae",
        "madracis": "madracis",
        "massive sponge": "sponge",
        "massive or encrusting sponges": "sponge",
        "mat tunicate": "tunicate",
        "meandrina": "meandrina",
        "millepora": "millepora",
        "millepora bleached": "bleached coral",
        "montastrea cavernosa": "montastraea cavernosa",
        "montipora capitata branching": "montipora capitata",
        "montipora capitata plate": "montipora capitata",
        "montipora flabellata": "montipora flabellata",
        "montipora patula": "montipora patula",
        "mycale laevis": "mycale laevis",
        "niphates digitalis": "niphates digitalis",
        "non hermatypic: millepora": "millepora",
        "non hermatypic: free living (fungia etc)": "fungiidae",
        "non hermatypic: heliopora": "heliopora",
        "ocomplex": "orbicella",
        "other": "other",
        "other acroporidae": "acroporidae",
        "other algae": "turf algae",
        "other branching genus: anacropora/echinopora/monti": "hard coral",
        "other branching genus: anacropora/echinopora/montipora/tubastrea": "hard coral",
        "other hard corals": "hard coral",
        "other sesile invertebrates bryozoa clams": "invertebrate",
        "other sesile invertebrates soft hexacorrallia": "hard coral",
        "other sessile invertebrates": "invertebrate",
        "other soft coral": "soft coral",
        "other soft-corals no common alcyoniidae and erects": "soft coral",
        "other soft-corals no common alcyoniidae and erects/ xeniidae/nepthtydae/tubipora/briareum": "soft coral",
        "pastreoides": "porites astreoides",
        "pavona duerdeni": "pavona duerdeni",
        "pavona maldivensis": "pavona maldivensis",
        "pavona varians": "pavona varians",
        "pocillopora damicornis": "pocillopora damicornis",
        "pocillopora eydouxi": "pocillopora grandis",
        "pocillopora meandrina/ligulata": "pocillopora",
        "pocillopora species": "pocillopora",
        "pocilloporidae": "pocilloporidae",
        "porites compressa fused branches": "porites compressa",
        "porites lichen": "porites lichen",
        "porites lobata/lutea": "porites",
        "porites lobata/lutea bleached": "bleached coral",
        "porites nodular branches": "porites",
        "porites rus/monticulosa": "porites",
        "poritidae branching": "poritidae",
        "poritidae encrusting": "poritidae",
        "poritidae massive": "poritidae",
        "porties compressa fingres": "porites compressa",
        "pporites": "porites porites",
        "pseudodiploria": "pseudodiploria",
        "rope sponge": "sponge",
        "sand": "sand",
        "sea cucumbers/sea urchins/sea stars/lobster": "invertebrate",
        "sea fans and plumes": "gorgoniidae",
        "sea fans/plumes": "gorgoniidae",
        "sea fans/plumes/branching whipes": "gorgoniidae",
        "seagrass": "seagrass",
        "sediment": "bare substrate",
        "siderastrea siderea": "siderastrea siderea",
        "soft coral plumes": "soft coral",
        "sponge": "sponge",
        "sponges": "sponge",
        "sponges: branching/rope forms": "sponge",
        "sponges: fan shaped forms": "sponge",
        "sponges: hollow sponge forms/cups/barrels/tubes/et al": "sponge",
        "tfp bleached": "bleached coral",
        "tfp: porites": "porites",
        "tfp: visible relief structures": "hard coral",
        "tfp: visible round corallites": "hard coral",
        "terrigenous sediment with turf": "turf algae",
        "thin/foliose  and plating bleached": "bleached coral",
        "transect hardware": "transect tools",
        "trash: human origin": "trash",
        "turf": "turf algae",
        "turf algae": "turf algae",
        "turf algae and sand": "turf algae",
        "turf sand": "turf algae",
        "unknown": "unknown",
        "utenuifolia": "agaricia tenuifolia",
        "vase sponge": "sponge",
        "volcanic sediment": "bare substrate",
        "water": "background",
        "xmuta": "xestospongia muta",
        "zoanthid": "zoanthid",
    }


def fetch_moorea_labeled_corals_to_mermaid() -> dict[str, str]:
    """Return the static Moorea Labeled Corals label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Moorea label name (e.g. ``"acrop"`` or ``"turf"``); values are the mapped
    MERMAID benthic-attribute name(s). Labels absent from this map collapse to background at
    training time via :class:`SourceLabelRegistry`.
    """
    return {
        "acan": "acanthastrea",
        "acrop": "acropora",
        "astreo": "astreopora",
        "cca": "crustose coralline algae",
        "cypha": "cyphastrea",
        "favia": "favia",
        "fung": "fungiidae",
        "gardin": "gardineroseris",
        "herpo": "herpolitha",
        "lepta": "leptastrea",
        "lepto": "leptoseris",
        "lobo": "lobophyllia",
        "macro": "macroalgae",
        "mille": "millepora",
        "monta": "montastraea",
        "monti": "montipora",
        "off": "other",
        "p mass": "porites",
        "p. irr": "porites rus",
        "p. rus": "porites rus",
        "pavon": "pavona",
        "pocill": "pocillopora",
        "porit": "porites",
        "psam": "psammocora",
        "sand": "sand",
        "sando": "sandalolitha",
        "soft": "soft coral",
        "stylo": "stylophora",
        "tuba": "tubastraea",
        "turf": "turf algae",
    }


def fetch_pacific_labeled_corals_to_mermaid() -> dict[str, str]:
    """Return the static Pacific Labeled Corals label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Pacific label name from the per-site ``labelmap.txt`` (e.g. ``"acropora"``
    or ``"cca"``); values are the mapped MERMAID benthic-attribute name(s). Labels absent from this
    map collapse to background at training time via :class:`SourceLabelRegistry`.
    """
    return {
        "acropora": "acropora",
        "all other": "other",
        "bare substrate": "bare substrate",
        "cca": "crustose coralline algae",
        "favia": "favia",
        "favites": "favites",
        "macroalgae": "macroalgae",
        "millepora": "millepora",
        "montipora": "montipora",
        "other scleractinians": "hard coral",
        "pavona": "pavona",
        "platygyra": "platygyra",
        "pocillopora": "pocillopora",
        "porites": "porites",
        "sand": "sand",
        "soft coral": "soft coral",
        "sponges": "sponge",
        "transect hardware": "transect tools",
        "turf": "turf algae",
        "unclear": "background",
    }


def fetch_ucsd_mosaics_to_mermaid(
    mapping_endpoint: str = "https://api.datamermaid.org/v1/classification/labelmappings/?provider=UCSD%20Mosaics",
) -> dict[str, str]:
    """Fetch the UCSD Mosaics label-name -> MERMAID benthic-attribute name mapping.

    Pages through the MERMAID
    API label-mappings endpoint filtered to ``provider=UCSD Mosaics`` and
    returns a dict keyed by the UCSD Mosaics ``provider_id`` (which holds the
    original class name from ``classes.json``, e.g. ``"Acropora (branching)"``
    or ``"Porites rus"``) with values equal to the MERMAID benthic-attribute
    name (or ``None`` if the UCSD label is not yet mapped, in which case it
    collapses to background at training time).

    Note: the UCSD Mosaics provider mapping has not yet been populated on the
    MERMAID side at the time of writing -- this fetcher will return an empty
    dict in that case, which causes every UCSD Mosaics source label to fall
    back to background through :class:`SourceLabelRegistry`.
    """
    return {
        "acropora (branching)-branching": "acropora",
        "acropora (corymbose)-corymbose": "acropora",
        "acropora (plating)-tabular": "acropora",
        "astreopora myriophthalma-sub_massive": "astreopora myriophthalma",
        "clavularia-fleshy_invert": "clavularia",
        "corallimorph-corallimorph": "corallimorpharia",
        "dictyosphaeria-encrusting_macroalgae": "dictyosphaeria",
        "favia matthai-massive": "dipsastraea matthaii",
        "favia stelligera-sub_massive": "goniastrea stelligera",
        "favites (encrusting)-encrusting": "favites",
        "favites (submassive)-sub_massive": "favites",
        "fungia-free_living": "fungia",
        "halimeda-erect_macroalgae": "halimeda",
        "halomitra pileus-free_living": "halomitra",
        "leptastrea-encrusting": "leptastrea",
        "lobophyllia-massive": "lobophyllia",
        "montastrea curta-massive": "astrea curta",
        "montipora (encrusting)-encrusting": "montipora",
        "montipora (plating)-plating": "montipora",
        "pasmmocora-encrusting": "psammocora",
        "pavona (submassive)-sub_massive": "pavona",
        "pavona varians-encrusting": "pavona varians",
        "platygyra-massive": "platygyra",
        "pocillopora-corymbose": "pocillopora",
        "pocillopora eydouxi-corymbose": "pocillopora grandis",
        "porites (massive)-massive": "porites",
        "porites rus-massive": "porites rus",
        "porites superfusa-encrusting": "porites superfusa",
        "soft coral-soft": "soft coral",
        "stylophora pistillata-corymbose": "stylophora pistillata",
        "turbinaria reniformis-plating": "turbinaria reniformis",
        "zooanthid-fleshy_invert": "zoanthid",
    }


def fetch_benthos_yuval_to_mermaid() -> dict[str, str]:
    """Return the static Benthos Yuval label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Benthos label name (e.g. ``"algae"`` or ``"sand"``); values are the mapped
    MERMAID benthic-attribute name(s). Labels absent from this map collapse to background at
    training time via :class:`SourceLabelRegistry`.
    """
    return {
        "algae": "turf algae",
        "coral": "hard coral",
        "other": "other",
        "rock": "bare substrate",
        "sand": "sand",
        "soft coral": "soft coral",
        "sponge": "sponge",
    }


def coralscapes_v2_to_mermaid() -> dict[str, str]:
    """Static Coralscapes V2 95-class -> MERMAID benthic-attribute mapping.

    Keyed by the native Coralscapes V2 class name (see ``id2label.json`` in the dataset repo);
    values are the mapped MERMAID benthic-attribute name. Labels absent from this map collapse to
    background at training time via :class:`SourceLabelRegistry`.
    """
    return {
        "acanthaster planci": "acanthaster planci",
        "acropora alive": "acropora",
        "acropora bleached": "bleached coral",
        "acropora dead": "dead coral",
        "algae covered brain coral": "turf algae",
        "algae covered branching coral": "turf algae",
        "algae covered lobophylliidae": "turf algae",
        "algae covered massive coral": "turf algae",
        "algae covered pocillopora": "turf algae",
        "algae covered porites": "turf algae",
        "algae covered substrate": "turf algae",
        "algae covered table acropora": "turf algae",
        "anemone": "anemone",
        "background": "background",
        "brain coral alive": "hard coral",
        "brain coral bleached": "bleached coral",
        "brain coral dead": "dead coral",
        "branching coral alive": "hard coral",
        "branching coral bleached": "bleached coral",
        "branching coral dead": "dead coral",
        "cirrhipathes": "hard coral",
        "dark": "dark",
        "dead giant clam": "tridacna giant clam",
        "feather worm": "sabellidae",
        "fish": "fish",
        "fungiidae alive": "fungiidae",
        "fungiidae bleached": "bleached coral",
        "fungiidae dead": "dead coral",
        "galaxea alive": "galaxea",
        "galaxea bleached": "bleached coral",
        "galaxea dead": "dead coral",
        "giant clam": "tridacna giant clam",
        "goniopora alive": "goniopora",
        "goniopora dead": "dead coral",
        "hard coral alive": "hard coral",
        "hard coral bleached": "bleached coral",
        "hard coral dead": "dead coral",
        "hard substrate": "bare substrate",
        "human": "human",
        "human-made structure": "bare substrate",
        "leather coral alive": "alcyoniidae",
        "lobophylliidae alive": "lobophyllia",
        "lobophylliidae bleached": "bleached coral",
        "lobophylliidae dead": "dead coral",
        "massive coral alive": "hard coral",
        "massive coral bleached": "bleached coral",
        "massive coral dead": "dead coral",
        "meandering coral alive": "hard coral",
        "meandering coral bleached": "bleached coral",
        "meandering coral dead": "dead coral",
        "millepora": "millepora",
        "millepora bleached": "bleached coral",
        "millepora dead": "dead coral",
        "nephtheidae alive": "nephtheidae",
        "other animal": "other invertebrates",
        "other coral alive": "hard coral",
        "other coral bleached": "bleached coral",
        "other coral dead": "dead coral",
        "pavona alive": "pavona",
        "pavona bleached": "bleached coral",
        "pavona dead": "dead coral",
        "pocillopora alive": "pocillopora",
        "pocillopora bleached": "bleached coral",
        "pocillopora dead": "dead coral",
        "porites alive": "porites",
        "porites bleached": "bleached coral",
        "porites dead": "dead coral",
        "rubble": "rubble",
        "sand": "sand",
        "sea cucumber": "sea cucumber",
        "sea urchin": "sea urchin",
        "seagrass": "seagrass",
        "seriatopora alive": "seriatopora",
        "seriatopora bleached": "bleached coral",
        "seriatopora dead": "dead coral",
        "soft coral (malacalcyonacea) alive": "soft coral",
        "soft coral (malacalcyonacea) bleached": "bleached coral",
        "sponge": "sponge",
        "stylophora alive": "stylophora",
        "stylophora bleached": "bleached coral",
        "stylophora dead": "dead coral",
        "table acropora alive": "acropora",
        "table acropora bleached": "bleached coral",
        "table acropora dead": "dead coral",
        "thin plate/encrusting coral alive": "hard coral",
        "thin plate/encrusting coral bleached": "bleached coral",
        "thin plate/encrusting coral dead": "dead coral",
        "transect line": "tape",
        "transect tools": "unknown",
        "trash": "trash",
        "turbinaria": "turbinaria reniformis",
        "turbinaria dead": "dead coral",
        "water column": "background",
        "water surface": "background",
        "xeniidae alive": "xeniidae",
    }


def coralscapes_to_mermaid() -> dict[str, str]:
    """Static Coralscapes 39-class -> MERMAID benthic-attribute mapping.

    Mapping was previously embedded inside ``CoralscapesDataset``. Keyed by the native Coralscapes
    class name; values are the mapped MERMAID benthic-attribute name. Labels absent from this map
    collapse to background at training time via :class:`SourceLabelRegistry`.
    """
    return {
        "human": "human",
        "background": "background",
        "fish": "fish",
        "sand": "sand",
        "rubble": "rubble",
        "unknown hard substrate": "bare substrate",
        "algae covered substrate": "turf algae",
        "dark": "dark",
        "branching bleached": "bleached coral",
        "branching dead": "dead coral",
        "branching alive": "hard coral",
        "stylophora alive": "stylophora",
        "pocillopora alive": "pocillopora",
        "acropora alive": "acropora",
        "table acropora alive": "acropora",
        "table acropora dead": "dead coral",
        "millepora": "millepora",
        "turbinaria": "turbinaria reniformis",
        "other coral dead": "dead coral",
        "other coral alive": "hard coral",
        "other coral bleached": "bleached coral",
        "massive/meandering alive": "hard coral",
        "massive/meandering dead": "dead coral",
        "massive/meandering bleached": "bleached coral",
        "meandering alive": "hard coral",
        "meandering dead": "dead coral",
        "meandering bleached": "bleached coral",
        "transect line": "tape",
        "transect tools": "unknown",
        "sea urchin": "sea urchin",
        "sea cucumber": "sea cucumber",
        "anemone": "anemone",
        "sponge": "sponge",
        "clam": "tridacna giant clam",
        "other animal": "other invertebrates",
        "trash": "trash",
        "seagrass": "seagrass",
        "crown of thorn": "acanthaster planci",
        "dead clam": "tridacna giant clam",
    }


def source_labels_to_target_labels(
    source_labels: torch.Tensor,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Map a global source-label segmentation map to target MERMAID label IDs.

    Pure GPU index op; ``lookup`` must live on the same device as
    ``source_labels``.

    Args:
        source_labels: Integer source-label map of shape ``(B, H, W)`` with
            values in ``[0, N]``.
        lookup: 1-D long tensor of shape ``(N+1,)`` mapping each global source
            label (0 = background) to its MERMAID target label ID
            (0 = background / unmapped).

    Returns:
        Long tensor of shape ``(B, H, W)`` containing target MERMAID label IDs.
    """
    if source_labels.dtype != torch.long:
        source_labels = source_labels.long()
    return lookup[source_labels]
