"""Fetch public CoralNet source metadata (name, location, owner) from the source pages.

For every source in a split of ``--split-config`` this reads ``https://coralnet.ucsd.edu/source/<id>/``
and keeps name, description, latitude/longitude, creation date, affiliation and members
(``username:role``). Sources that are no longer public get ``public = False``. Requests are
sequential with ``--delay`` seconds between them; already fetched sources are skipped, so the
script can be re-run to resume.

Used by ``mermaidseg.datasets.coralnet.dedup.splits`` to keep sources of one project on the same
side of the split.

Usage:
    python -m mermaidseg.datasets.coralnet.scraper.source_metadata
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from mermaidseg.datasets.coralnet.dedup.common import (
    DEFAULT_OUTPUTS,
    DEFAULT_SPLIT_CONFIG,
    coralnet_splits,
)

URL = "https://coralnet.ucsd.edu/source/{}/"


def parse_source_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    out: dict = {"public": True}
    title = soup.find("title")
    out["name"] = title.get_text(" ", strip=True).split(" | ")[0] if title else None
    for li in soup.select("li"):
        spans = li.find_all("span", recursive=False)
        if len(spans) < 2:
            continue
        key, value = spans[0].get_text(strip=True), " ".join(spans[1].get_text(" ", strip=True).split())
        if key.startswith("Latitude"):
            lat, _, lon = value.partition(",")
            out["latitude"], out["longitude"] = pd.to_numeric(lat, errors="coerce"), pd.to_numeric(lon, errors="coerce")
        elif key.startswith("Created"):
            out["created"] = value
        elif key.startswith("Point generation"):
            out["point_generation"] = value
    desc = soup.find(id="description-box")
    if desc is not None:
        out["description"] = " ".join(desc.get_text(" ", strip=True).removeprefix("Description").split())
    members = soup.find(id="members-column")
    if members is not None:
        aff = members.find("div", class_="center")
        out["affiliation"] = aff.get_text(strip=True) if aff else None
        rows = []
        for tr in members.select("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                rows.append(f"{tds[0].get_text(strip=True)}:{tds[1].get_text(strip=True)}")
        out["members"] = ";".join(rows)
        out["admins"] = ";".join(r.split(":")[0] for r in rows if r.endswith(":Admin"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUTS / "coralnet_source_metadata.csv")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sources = sorted(set().union(*coralnet_splits(args.split_config).values()))
    done = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(columns=["source_id"])
    todo = [s for s in sources if s not in set(done["source_id"])]
    print(f"{len(sources)} sources, {len(todo)} to fetch")

    rows = done.to_dict("records")
    session = requests.Session()
    for n, s in enumerate(todo, 1):
        try:
            r = session.get(URL.format(s), timeout=30)
            if r.status_code == 200 and "members-column" in r.text:
                row = {"source_id": s, **parse_source_page(r.text)}
            else:
                row = {"source_id": s, "public": False, "http_status": r.status_code}
        except requests.RequestException as e:
            print(f"{s}: {e}")
            time.sleep(args.delay)
            continue
        rows.append(row)
        if n % 50 == 0 or n == len(todo):
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{n}/{len(todo)}")
        time.sleep(args.delay)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
