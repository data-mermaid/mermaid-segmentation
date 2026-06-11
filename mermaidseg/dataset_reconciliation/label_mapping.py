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


def fetch_catlin_seaview_to_mermaid() -> dict[str, list[str]]:
    """Return the static Catlin Seaview label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Catlin label name; values are the mapped MERMAID benthic-attribute
    name(s). Labels absent from this map collapse to background at training time via
    :class:`SourceLabelRegistry`.

    NOTE: values are ``list[str]`` but ``SourceLabelRegistry`` expects scalar targets; see #135.
    """
    return {
        "aarcheri": ["aplysina archeri"],
        "acervicornis": ["acropora cervicornis"],
        "acompresa": ["amphimedon compressa"],
        "acropora corymbose/tabular/plate": ["Acropora"],
        "acropora digitate": ["Acropora"],
        "acroporidae other": ["acroporidae"],
        "acroporidae branching": ["acroporidae"],
        "acroporidae hispidose": ["acroporidae"],
        "acroporidae plate/encrusting": ["acroporidae"],
        "acroporidae table/corymbose/digitate": ["acroporidae"],
        "afistularis": ["aplysina fistularis"],
        "agaricia/undaria": ["agaricia"],
        "aiolochroia crassa": ["aiolochroia crassa"],
        "alcyoniidae": ["alcyoniidae"],
        "algae matrix": ["epilithic algal matrix"],
        "algae: large visible globules": ["Turf algae"],
        "apalmata": ["acropora palmata"],
        "aplysina fulva_cauliformis": ["aplysina"],
        "atubulata": ["agelas tubulata"],
        "bra: acropora bottlebrush": ["Acropora"],
        "bra: acropora branching": ["Acropora"],
        "bra: other": ["other"],
        "branching porites": ["porites"],
        "branching stylophora": ["stylophora"],
        "cca": ["crustose coralline algae"],
        "calcifying algae: halimeda": ["halimeda"],
        "calcifying algae: padina": ["padina"],
        "calcifying macroalgae: padina": ["padina"],
        "cdelitrix": ["cliona delitrix"],
        "cnatans": ["colpophyllia natans"],
        "common lrg alcyoniide": ["alcyoniidae"],
        "coral: meandroid": ["Hard coral"],
        "cplicifera": ["callyspongia plicifera"],
        "crinoids": ["crinoid"],
        "crustose coralline algae": ["crustose coralline algae"],
        "cvaginalis": ["callyspongia vaginalis"],
        "cyanobacteria": ["cyanobacteria"],
        "cyanobacteria films": ["cyanobacteria"],
        "cyanobacteria on rock or other substrates": ["cyanobacteria"],
        "cyanobacteria smothering dead coral": ["cyanobacteria"],
        "cyanobacteria smothering rubble": ["cyanobacteria"],
        "diadema/ehinothix": ["Sea urchin"],
        "dictyota": ["dictyota"],
        "dlabyrinthiformis": ["diploria labyrinthiformis"],
        "echinometra": ["echinometra"],
        "encrusting cliona spp.": ["cliona"],
        "encrusting gorgonian": ["gorgonia"],
        "encrusting sponge": ["Sponge"],
        "epilithic algal matrix": ["epilithic algal matrix"],
        "epilithic algal matirx smotheting rubble": ["epilithic algal matrix"],
        "epilithic algal matrix on rock or other substrates": ["epilithic algal matrix"],
        "epillithic algal matrix": ["epilithic algal matrix"],
        "erect rhodphyta": ["rhodophyta"],
        "erect gorgonians": ["gorgonia"],
        "erect sponge": ["Sponge"],
        "eusmilia fastigiata": ["eusmilia fastigiata"],
        "favidae-mussidae massive/meandroid": ["Hard coral"],
        "filamentous macroalgae": ["macroalgae"],
        "fish": ["Unknown"],
        "foliose algae: other": ["Turf algae"],
        "foliose fan shaped algae": ["Turf algae"],
        "foliose feathery algae": ["Turf algae"],
        "foliose strap/ branched algae": ["Turf algae"],
        "foliose sheeting macroalage: (e.g ulva)": ["macroalgae"],
        "halimeda": ["halimeda"],
        "hydroids feathery types": ["hydroid"],
        "individual tunicates": ["tunicate"],
        "ircinia_massive": ["ircinia"],
        "leathery macrophyte: other": ["macroalgae"],
        "leptoseris": ["leptoseris"],
        "loose substrate: rubble": ["Rubble"],
        "loose substrate: sand": ["Sand"],
        "lvariegata": ["lobophora variegata"],
        "mase: isopora": ["isopora"],
        "mase: lobophyllia": ["lobophyllia"],
        "mase: meandering other": ["Hard coral"],
        "mase: porites": ["porites"],
        "macroalgae": ["macroalgae"],
        "macroalgae 1": ["macroalgae"],
        "macroalgae encrusting red": ["macroalgae"],
        "macrolagae all genera": ["macroalgae"],
        "madracis": ["madracis"],
        "massive sponge": ["Sponge"],
        "massive or encrusting sponges": ["Sponge"],
        "mat tunicate": ["tunicate"],
        "meandrina": ["meandrina"],
        "millepora": ["Milleporidae"],
        "montastrea cavernosa": ["montastraea cavernosa"],
        "montipora capitata branching": ["montipora capitata"],
        "montipora capitata plate": ["montipora capitata"],
        "montipora flabellata": ["montipora flabellata"],
        "montipora patula": ["montipora patula"],
        "mycale laevis": ["mycale laevis"],
        "niphates digitalis": ["niphates digitalis"],
        "non hermatypic: millepora": ["Milleporidae"],
        "non hermatypic: heliopora": ["heliopora"],
        "ocomplex": ["orbicella"],
        "other": ["other"],
        "other acroporidae": ["acroporidae"],
        "other algae": ["Turf algae"],
        "other sesile invertebrates bryozoa clams": ["invertebrate"],
        "other sesile invertebrates soft hexacorrallia": ["Hard coral"],
        "other sessile invertebrates": ["invertebrate"],
        "pastreoides": ["porites astreoides"],
        "pavona duerdeni": ["pavona duerdeni"],
        "pavona maldivensis": ["pavona maldivensis"],
        "pavona varians": ["pavona varians"],
        "pocillopora damicornis": ["pocillopora damicornis"],
        "pocillopora eydouxi": ["pocillopora grandis"],
        "pocillopora meandrina/ligulata": ["Pocillopora"],
        "pocillopora species": ["Pocillopora"],
        "pocilloporidae": ["pocilloporidae"],
        "porites compressa fused branches": ["porites compressa"],
        "porites lichen": ["porites lichen"],
        "porites lobata/lutea": ["porites"],
        "porites nodular branches": ["porites"],
        "porites rus/monticulosa": ["porites"],
        "poritidae branching": ["poritidae"],
        "poritidae encrusting": ["poritidae"],
        "poritidae massive": ["poritidae"],
        "porties compressa fingres": ["porites compressa"],
        "pporites": ["porites porites"],
        "pseudodiploria": ["pseudodiploria"],
        "rope sponge": ["Sponge"],
        "sand": ["Sand"],
        "sea cucumbers/sea urchins/sea stars/lobster": ["invertebrate"],
        "sea fans and plumes": ["gorgoniidae"],
        "sea fans/plumes": ["gorgoniidae"],
        "sea fans/plumes/branching whipes": ["gorgoniidae"],
        "seagrass": ["Seagrass"],
        "siderastrea siderea": ["siderastrea siderea"],
        "sponge": ["Sponge"],
        "sponges": ["Sponge"],
        "sponges: branching/rope forms": ["Sponge"],
        "sponges: fan shaped forms": ["Sponge"],
        "sponges: hollow sponge forms/cups/barrels/tubes/et al": ["Sponge"],
        "tfp: porites": ["porites"],
        "terrigenous sediment with turf": ["turf algae"],
        "transect hardware": ["Unknown"],
        "trash: human origin": ["Trash"],
        "turf": ["turf algae"],
        "turf algae": ["turf algae"],
        "turf algae and sand": ["turf algae"],
        "turf sand": ["turf algae"],
        "unknown": ["unknown"],
        "utenuifolia": ["agaricia tenuifolia"],
        "vase sponge": ["Sponge"],
        "water": ["Unknown", "Obscured"],
        "xmuta": ["xestospongia muta"],
        "zoanthid": ["zoanthid"],
    }


def fetch_moorea_labeled_corals_to_mermaid() -> dict[str, list[str]]:
    """Return the static Moorea Labeled Corals label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Moorea label name (e.g. ``"acrop"`` or ``"turf"``); values are the mapped
    MERMAID benthic-attribute name(s). Labels absent from this map collapse to background at
    training time via :class:`SourceLabelRegistry`.

    NOTE: values are ``list[str]`` but ``SourceLabelRegistry`` expects scalar targets; see #135.
    """
    return {
        "acan": ["acanthastrea"],
        "acrop": ["Acropora"],
        "astreo": ["astreopora"],
        "cca": ["crustose coralline algae"],
        "cypha": ["cyphastrea"],
        "favia": ["favia"],
        "gardin": ["gardineroseris"],
        "herpo": ["herpolitha"],
        "lepta": ["leptastrea"],
        "lepto": ["leptoseris"],
        "lobo": ["lobophyllia"],
        "macro": ["macroalgae"],
        "mille": ["Milleporidae"],
        "monta": ["montastraea"],
        "monti": ["montipora"],
        "off": ["other"],
        "p mass": ["porites"],
        "p. irr": ["porites rus"],
        "p. rus": ["porites rus"],
        "pocill": ["Pocillopora"],
        "porit": ["porites"],
        "psam": ["psammocora"],
        "sand": ["Sand"],
        "stylo": ["stylophora"],
        "tuba": ["tubastraea"],
        "turf": ["turf algae"],
    }


def fetch_pacific_labeled_corals_to_mermaid() -> dict[str, list[str]]:
    """Return the static Pacific Labeled Corals label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Pacific label name from the per-site ``labelmap.txt`` (e.g. ``"acropora"``
    or ``"cca"``); values are the mapped MERMAID benthic-attribute name(s). Labels absent from this
    map collapse to background at training time via :class:`SourceLabelRegistry`.

    NOTE: values are ``list[str]`` but ``SourceLabelRegistry`` expects scalar targets; see #135.
    """
    return {
        "acropora": ["Acropora"],
        "all other": ["other"],
        "cca": ["crustose coralline algae"],
        "favia": ["favia"],
        "favites": ["favites"],
        "macroalgae": ["macroalgae"],
        "millepora": ["Milleporidae"],
        "montipora": ["montipora"],
        "platygyra": ["platygyra"],
        "pocillopora": ["Pocillopora"],
        "porites": ["porites"],
        "sand": ["Sand"],
        "sponges": ["Sponge"],
        "transect hardware": ["Unknown"],
        "turf": ["turf algae"],
        "unclear": ["Unknown", "Obscured"],
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
    response = requests.get(mapping_endpoint, timeout=30)
    response.raise_for_status()
    data = response.json()
    labelset = list(data["results"])
    while data.get("next"):
        response = requests.get(data["next"], timeout=30)
        response.raise_for_status()
        data = response.json()
        labelset.extend(data["results"])
    return {str(label["provider_id"]): label["benthic_attribute_name"] for label in labelset}


def fetch_benthos_yuval_to_mermaid() -> dict[str, list[str]]:
    """Return the static Benthos Yuval label-name -> MERMAID benthic-attribute mapping.

    Keyed by the original Benthos label name (e.g. ``"algae"`` or ``"sand"``); values are the mapped
    MERMAID benthic-attribute name(s). Labels absent from this map collapse to background at
    training time via :class:`SourceLabelRegistry`.

    NOTE: values are ``list[str]`` but ``SourceLabelRegistry`` expects scalar targets; see #135.
    """
    return {"algae": ["Turf algae"], "other": ["other"], "sand": ["Sand"], "sponge": ["Sponge"]}


def coralscapes_to_mermaid() -> dict[str, list[str]]:
    """Static Coralscapes 39-class -> MERMAID benthic-attribute mapping.

    Mapping was previously embedded inside ``CoralscapesDataset``. The first element of each value
    list is treated as the canonical MERMAID label; subsequent elements are alternative
    interpretations not currently used.
    """
    return {
        "human": ["Unknown"],
        "background": ["Unknown", "Obscured"],
        "fish": ["Unknown"],
        "sand": ["Sand"],
        "rubble": ["Rubble"],
        "unknown hard substrate": ["Bare substrate"],
        "algae covered substrate": ["Turf algae"],
        "dark": ["Unknown"],
        "branching bleached": ["Bleached coral"],
        "branching dead": ["Dead coral"],
        "branching alive": ["Hard coral"],
        "stylophora alive": ["Stylophora"],
        "pocillopora alive": ["Pocillopora"],
        "acropora alive": ["Acropora"],
        "table acropora alive": ["Acropora"],
        "table acropora dead": ["Dead coral"],
        "millepora": ["Milleporidae"],
        "turbinaria": ["Turbinaria reniformis"],
        "other coral": ["Bleached coral"],
        "other coral dead": ["Dead coral"],
        "other coral alive": ["Hard coral"],
        "other coral bleached": ["Bleached coral"],
        "massive/meandering alive": ["Hard coral"],
        "massive/meandering dead": ["Dead coral"],
        "massive/meandering bleached": ["Bleached coral"],
        "meandering alive": ["Hard coral"],
        "meandering dead": ["Dead coral"],
        "meandering bleached": ["Bleached coral"],
        "transect line": ["Tape"],
        "transect tools": ["Unknown"],
        "sea urchin": ["Sea urchin"],
        "sea cucumber": ["Sea cucumber"],
        "anemone": ["Anemone"],
        "sponge": ["Sponge"],
        "clam": ["Tridacna giant clam"],
        "other animal": ["Other invertebrates"],
        "trash": ["Trash"],
        "seagrass": ["Seagrass"],
        "crown of thorn": ["Acanthaster planci"],
        "dead clam": ["Unknown"],
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
