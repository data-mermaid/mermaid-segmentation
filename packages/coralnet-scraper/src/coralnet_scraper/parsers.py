"""Pure HTML parsers for CoralNet pages (offline-testable)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag


def parse_source_access(html: str) -> tuple[bool, str | None]:
    """Interpret a CoralNet source main page HTTP body for accessibility."""
    if "Page could not be found" in html:
        return False, "Source does not exist"
    if "don't have permission" in html:
        return False, "Permission denied"
    return True, None


def _image_status_detail_table(soup: BeautifulSoup) -> Tag | None:
    header = soup.find("h4", string="Image Status")
    if not header:
        return None
    table = header.find_next_sibling("table", class_="detail_box_table")
    return table if isinstance(table, Tag) else None


@dataclass(frozen=True)
class ImageStatusCounts:
    """Parsed from the ``Image Status`` ``detail_box_table`` on the source overview."""

    table_found: bool
    unclassified: int | None = None
    unconfirmed: int | None = None
    confirmed: int | None = None
    total_images: int | None = None


def parse_image_status_counts(html: str) -> ImageStatusCounts:
    """Extract annotation status counts where present; numeric fields ``None`` if missing."""
    soup = BeautifulSoup(html, "html.parser")
    table = _image_status_detail_table(soup)
    if not table:
        return ImageStatusCounts(table_found=False)

    uc: int | None = None
    uq: int | None = None
    cf: int | None = None
    total: int | None = None

    def _linked_int(row: Tag) -> int | None:
        link = row.find("a")
        if not link:
            return None
        try:
            return int(link.get_text().strip().replace(",", ""))
        except ValueError:
            return None

    for tr in table.find_all("tr"):
        row_text = " ".join(td.get_text() for td in tr.find_all("td"))
        val = _linked_int(tr)
        if "Unclassified:" in row_text:
            uc = val
        elif "Unconfirmed:" in row_text:
            uq = val
        elif "Confirmed:" in row_text:
            cf = val
        elif "Total images:" in row_text:
            total = val

    return ImageStatusCounts(
        table_found=True,
        unclassified=uc,
        unconfirmed=uq,
        confirmed=cf,
        total_images=total,
    )


def parse_total_images_from_source_html(html: str) -> int:
    """Extract 'Total images' count from source overview HTML; 0 if unknown."""
    c = parse_image_status_counts(html)
    if c.total_images is None:
        return 0
    return c.total_images


def classifier_plot_data_from_source_html(html: str) -> list[dict] | None:
    """Return classifierPlotData JS array decoded as Python list, or None."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        if script.string and "Classifier overview" in script.string:
            text = script.string
            marker = "let classifierPlotData = "
            idx = text.find(marker)
            if idx == -1:
                continue
            idx += len(marker)
            end = text.find("];", idx)
            if end == -1:
                continue
            blob = text[idx : end + 1].replace("'", '"')
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                return None
    return None


def extract_csrf_token(html: str) -> str | None:
    """Extract ``csrfmiddlewaretoken`` from any page containing a hidden input."""
    soup = BeautifulSoup(html, "html.parser")
    csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})
    return csrf["value"] if csrf else None


def parse_export_annotations_prep_form(html: str) -> str | None:
    """Extract CSRF token from browse images export prep form."""
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", {"id": "export-annotations-prep-form"})
    if not form:
        return None
    csrf = form.find("input", {"name": "csrfmiddlewaretoken"})
    return csrf["value"] if csrf else None
