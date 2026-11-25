"""Source-dataset to MERMAID benthic-attribute target-label mappings.

Translates a source-space label (CoralNet label ids, Coralscapes V2 names, other datasets' label
names) into a MERMAID benthic-attribute target name. CoralNet comes from the MERMAID LabelMapping
API (or an offline snapshot of it); the other datasets use static dicts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

CORALNET_LABELMAPPINGS_URL = (
    "https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet&limit=1000"
)
# The snapshot sits under the local S3 cache root, beside the cached S3 objects, in a pseudo-bucket
# no real S3 object uses: `<cache root>/_mermaid_api/classification/labelmappings/coralnet.json`.
CORALNET_SNAPSHOT_BUCKET = "_mermaid_api"
CORALNET_SNAPSHOT_KEY = "classification/labelmappings/coralnet.json"
_REQUEST_TIMEOUT_SEC = 30
_RETRY_BACKOFF_SEC = 1


class CoralNetLabelMapping(TypedDict):
    """One CoralNet row from the MERMAID LabelMapping API."""

    id: str | None
    benthic_attribute_id: str | None
    benthic_attribute_name: str | None
    growth_form_id: str | None
    growth_form_name: str | None
    provider: str | None
    provider_id: int
    provider_label: str | None


@dataclass(frozen=True)
class CoralNetLabelMappings:
    """The CoralNet label mappings a run used, keyed by CoralNet label id.

    Attributes:
        records: API rows keyed by integer CoralNet label id (``provider_id``).
        source: ``"api"`` when fetched for this run, ``"snapshot"`` when read offline.
        fetched_at: UTC ISO time the rows were fetched from the API.
    """

    records: dict[int, CoralNetLabelMapping]
    source: Literal["api", "snapshot"]
    fetched_at: str

    @property
    def sha256(self) -> str:
        """Digest of the records, independent of their order."""
        return coralnet_mapping_digest(self.records)

    def provenance(self) -> dict[str, Any]:
        """Small summary for run lineage: source, digest, record count and fetch time."""
        return {
            "source": self.source,
            "sha256": self.sha256,
            "n_records": len(self.records),
            "fetched_at": self.fetched_at,
        }

    def to_json(self) -> dict[str, Any]:
        """JSON form shared by the offline snapshot and the logged run artifact."""
        return {
            "fetched_at": self.fetched_at,
            "sha256": self.sha256,
            "records": [self.records[k] for k in sorted(self.records)],
        }


def _retrying_session() -> requests.Session:
    """A session that retries a GET up to 3 times on connection errors and 502/503/504."""
    retry = Retry(
        total=3,
        backoff_factor=_RETRY_BACKOFF_SEC,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_paginated_results(url: str) -> list[dict]:
    """Page through a MERMAID list endpoint until ``next`` is exhausted.

    Raises:
        requests.RequestException: A request still fails after its retries.
        ValueError: A ``next`` URL repeats a page already fetched.
    """
    records: list[dict] = []
    seen: set[str] = set()
    next_url: str | None = url
    with _retrying_session() as session:
        while next_url:
            if next_url in seen:
                raise ValueError(f"Pagination loop: {next_url} was already fetched")
            seen.add(next_url)
            response = session.get(next_url, timeout=_REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
            records.extend(data["results"])
            next_url = data.get("next")
    return records


def _normalize_provider_id(provider_id: object) -> int | None:
    """Return ``provider_id`` as an int (``82``, ``82.0`` and ``"82.0"`` give 82), else None."""
    try:
        value = float(provider_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not value.is_integer():
        return None
    return int(value)


def parse_coralnet_label_mappings(rows: list[dict]) -> dict[int, CoralNetLabelMapping]:
    """Key raw LabelMapping rows by integer CoralNet label id.

    Raises:
        ValueError: If a row's ``provider_id`` is not an integer, or two rows share a CoralNet
            label id.
    """
    records: dict[int, CoralNetLabelMapping] = {}
    for row in rows:
        provider_id = _normalize_provider_id(row.get("provider_id"))
        if provider_id is None:
            raise ValueError(
                f"CoralNet label mapping has non-integer provider_id {row.get('provider_id')!r}"
            )
        if provider_id in records:
            raise ValueError(f"Duplicate CoralNet label mapping for provider_id {provider_id}")
        records[provider_id] = CoralNetLabelMapping(
            id=row.get("id"),
            benthic_attribute_id=row.get("benthic_attribute_id"),
            benthic_attribute_name=row.get("benthic_attribute_name"),
            growth_form_id=row.get("growth_form_id"),
            growth_form_name=row.get("growth_form_name"),
            provider=row.get("provider"),
            provider_id=provider_id,
            provider_label=row.get("provider_label"),
        )
    return records


def fetch_coralnet_label_mappings(url: str = CORALNET_LABELMAPPINGS_URL) -> CoralNetLabelMappings:
    """Fetch every CoralNet row from the MERMAID LabelMapping API.

    Network and HTTP errors propagate: a run never falls back to a stale mapping silently.
    """
    records = parse_coralnet_label_mappings(_fetch_paginated_results(url))
    logger.info("Fetched %d CoralNet label mappings from the MERMAID API", len(records))
    return CoralNetLabelMappings(
        records=records, source="api", fetched_at=datetime.now(UTC).isoformat()
    )


def coralnet_mapping_digest(records: Mapping[int, CoralNetLabelMapping]) -> str:
    """sha256 of the records as canonical JSON (sorted ids and keys)."""
    canonical = json.dumps([records[k] for k in sorted(records)], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def coralnet_snapshot_path() -> Path:
    """Path of the offline CoralNet mapping snapshot under the local cache root.

    Raises:
        RuntimeError: If no local cache root is configured.
    """
    from mermaidseg.datasets.local_cache import LocalS3Cache

    return LocalS3Cache.get().local_path(CORALNET_SNAPSHOT_BUCKET, CORALNET_SNAPSHOT_KEY)


def write_coralnet_snapshot(mappings: CoralNetLabelMappings, path: Path) -> None:
    """Write ``mappings`` to ``path`` as JSON, for offline runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mappings.to_json(), indent=1, sort_keys=True))
    tmp.replace(path)


def read_coralnet_snapshot(path: Path) -> CoralNetLabelMappings:
    """Read a snapshot written by :func:`write_coralnet_snapshot`.

    Raises:
        ValueError: If the stored ``sha256`` differs from the digest of the records read.
    """
    data = json.loads(path.read_text())
    records = parse_coralnet_label_mappings(data["records"])
    digest = coralnet_mapping_digest(records)
    if digest != data["sha256"]:
        raise ValueError(
            f"CoralNet snapshot {path} is corrupt: stored sha256 {data['sha256']} "
            f"does not match its records ({digest})"
        )
    return CoralNetLabelMappings(records=records, source="snapshot", fetched_at=data["fetched_at"])


def load_coralnet_label_mappings() -> CoralNetLabelMappings:
    """CoralNet label mappings for a run: the API online, the cached snapshot offline.

    Offline (``MERMAIDSEG_S3_OFFLINE`` set) reads only the snapshot that
    ``scripts/prefetch_cache.py`` wrote. Online fetches from the API and returns the rows without
    touching the snapshot: only ``scripts/prefetch_cache.py`` writes it, so concurrent ranks never
    race on one file.

    Raises:
        DataLoadError: Offline and the snapshot is missing (or no cache is configured).
        requests.RequestException: Online and the API call fails.
    """
    from mermaidseg.datasets.local_cache import DataLoadError, LocalS3Cache

    cache = LocalS3Cache.get()
    if cache.offline:
        path = coralnet_snapshot_path() if cache.enabled else None
        if path is None or not path.is_file():
            location = f" at {path}" if path is not None else " (no local cache configured)"
            raise DataLoadError(
                f"no CoralNet label-mapping snapshot{location} in offline mode "
                "(MERMAIDSEG_S3_OFFLINE set); run scripts/prefetch_cache.py "
                "to warm the local cache before running offline."
            )
        mappings = read_coralnet_snapshot(path)
        logger.info(
            "Read %d CoralNet label mappings from the offline snapshot %s (fetched %s)",
            len(mappings.records),
            path,
            mappings.fetched_at,
        )
        return mappings
    return fetch_coralnet_label_mappings()


def coralnet_to_mermaid(
    id2name: Mapping[Any, str | None],
    records: Mapping[int, CoralNetLabelMapping],
    used_names: Collection[str] | None = None,
) -> dict[str, str]:
    """Map lowercased CoralNet label names to MERMAID benthic-attribute names.

    The join is on the CoralNet label id: ``id2name`` (the run's ``coralnet_id2name`` JSON) names
    each id the dataset uses, and ``records`` maps the id to its benthic attribute. A label renamed
    in CoralNet therefore still maps. Ids with no mapping row, or with no string name (``None`` or
    ``NaN``), are left out, so they train as background.

    Args:
        id2name: CoralNet label id -> name.
        records: API rows keyed by integer CoralNet label id.
        used_names: Label names the caller uses (compared lowercased). When given, ids whose name
            is not in it are skipped before the join and the collision check, so a collision
            among unused names does not raise. ``None`` keeps every named id.

    Raises:
        ValueError: If two ids share a lowercased name but map to different benthic attributes.
    """
    used = None if used_names is None else {n.lower() for n in used_names}
    mapping: dict[str, str] = {}
    source_id: dict[str, int] = {}
    named = unmapped = 0
    for raw_id, name in id2name.items():
        if not isinstance(name, str):  # None, or NaN from a JSON null-like value
            continue
        if used is not None and name.lower() not in used:
            continue
        named += 1
        label_id = int(raw_id)
        record = records.get(label_id)
        target = record["benthic_attribute_name"] if record is not None else None
        if target is None:
            unmapped += 1
            continue
        key = name.lower()
        if key in mapping and mapping[key] != target:
            raise ValueError(
                f"CoralNet label name {key!r} has ids {source_id[key]} and {label_id} mapped to "
                f"different benthic attributes ({mapping[key]!r}, {target!r})"
            )
        mapping[key] = target
        source_id[key] = label_id
    logger.info(
        "CoralNet label mapping: %d of %d named label ids have a MERMAID benthic attribute",
        named - unmapped,
        named,
    )
    return mapping


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
    """Return the static Moorea Labeled Corals label-name -> MERMAID benthic-attribute map.

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
    """Return the static Pacific Labeled Corals label-name -> MERMAID benthic-attribute map.

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


def fetch_ucsd_mosaics_to_mermaid() -> dict[str, str]:
    """Return the static UCSD Mosaics label-name -> MERMAID benthic-attribute map.

    Keyed by the original class name from ``classes.json`` (lowercased, e.g.
    ``"acropora (branching)"`` or ``"porites rus"``); values are the mapped MERMAID
    benthic-attribute name(s). Labels absent from this map collapse to background at
    training time via :class:`SourceLabelRegistry`.
    """
    return {
        "acropora (branching)": "acropora",
        "acropora (corymbose)": "acropora",
        "acropora (plating)": "acropora",
        "astreopora myriophthalma": "astreopora myriophthalma",
        "clavularia": "clavularia",
        "corallimorph": "corallimorpharia",
        "dictyosphaeria": "dictyosphaeria",
        "favia matthai": "dipsastraea matthaii",
        "favia stelligera": "goniastrea stelligera",
        "favites (encrusting)": "favites",
        "favites (submassive)": "favites",
        "fungia": "fungia",
        "halimeda": "halimeda",
        "halomitra pileus": "halomitra",
        "leptastrea": "leptastrea",
        "lobophyllia": "lobophyllia",
        "montastrea curta": "astrea curta",
        "montipora (encrusting)": "montipora",
        "montipora (plating)": "montipora",
        "pasmmocora": "psammocora",
        "pavona (submassive)": "pavona",
        "pavona varians": "pavona varians",
        "platygyra": "platygyra",
        "pocillopora": "pocillopora",
        "pocillopora eydouxi": "pocillopora grandis",
        "porites (massive)": "porites",
        "porites rus": "porites rus",
        "porites superfusa": "porites superfusa",
        "soft coral": "soft coral",
        "stylophora pistillata": "stylophora pistillata",
        "turbinaria reniformis": "turbinaria reniformis",
        "zooanthid": "zoanthid",
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
        "massive alive": "hard coral",
        "massive bleached": "bleached coral",
        "massive dead": "dead coral",
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
