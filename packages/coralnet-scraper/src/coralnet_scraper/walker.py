"""The browse walk: every image of a CoralNet source, checked against its overview count."""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from bs4 import BeautifulSoup

from coralnet_scraper import s3_io
from coralnet_scraper.errors import (
    ContentError,
    SessionExpired,
    WalkIncomplete,
    _retrying,
    _TotalChanged,
    _WalkCountMismatch,
)
from coralnet_scraper.session import _PAGE_EXPORT_TIMEOUT, CoralNetSession, browse_page_url

logger = logging.getLogger(__name__)

_WALK_CHECKPOINT_EVERY_PAGES = 50  # save the walk's progress to _partial/walk.json every N pages
_BROWSE_PROGRESS_EVERY = 10  # log walk progress every N pages
_WALK_ATTEMPTS = 3
_WALK_FILE = "walk.json"

_IMAGE_HREF_PK = re.compile(r"/image/(\d+)(?:/|$)")


@dataclass(frozen=True)
class ImageRef:
    """One image found by the browse walk.

    Attributes:
        id: CoralNet image id.
        name: Thumbnail title, as listed in ``image_list.csv``.
        page: Browse page the image is listed on, counted from 1.
    """

    id: int
    name: str
    page: int

    @property
    def view_path(self) -> str:
        """Return the relative link of the image's view page."""
        return f"/image/{self.id}/view/"


@dataclass
class _WalkProgress:
    """The images a browse walk has found so far, as saved in ``_partial/walk.json``."""

    refs: dict[int, ImageRef] = field(default_factory=dict)
    last_page: int = 0
    page_size: int = 0
    complete: bool = False

    def add_page(self, thumbs: list[tuple[int, str]]) -> None:
        """Record the next page's thumbnails; an id already seen keeps its first page."""
        self.last_page += 1
        self.page_size = self.page_size or len(thumbs)
        for image_id, name in thumbs:
            self.refs.setdefault(image_id, ImageRef(image_id, name, self.last_page))

    def to_json(self) -> dict[str, Any]:
        """Return the JSON document of this progress."""
        return {
            "last_page": self.last_page,
            "page_size": self.page_size,
            "complete": self.complete,
            "refs": [{"id": r.id, "name": r.name, "page": r.page} for r in self.refs.values()],
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> _WalkProgress:
        """Return the progress saved as ``doc`` by :meth:`to_json`."""
        refs = {r["id"]: ImageRef(r["id"], r["name"], r["page"]) for r in doc["refs"]}
        return cls(refs, doc["last_page"], doc["page_size"], doc["complete"])


def _parse_browse_thumbs(html: str) -> tuple[list[tuple[int, str]], str | None, bool]:
    """Return a browse page's thumbnails, its "Next page" link and whether it has the export form.

    Thumbnails are ``(image id, title)``. A thumbnail without an ``/image/<id>/`` link is
    skipped; one without a title gets ``""``.
    """
    soup = BeautifulSoup(html, "html.parser")
    thumbs = []
    for wrapper in soup.find_all("span", class_="thumb_wrapper"):
        link = wrapper.find("a")
        match = _IMAGE_HREF_PK.search(cast(str, link.get("href", ""))) if link else None
        if match:
            img = wrapper.find("img")
            thumbs.append((int(match.group(1)), cast(str, img.get("title", "")) if img else ""))
    next_anchor = soup.find("a", title="Next page")
    next_href = cast("str | None", next_anchor.get("href")) if next_anchor else None
    has_form = soup.find("form", {"id": "export-annotations-prep-form"}) is not None
    return thumbs, next_href, has_form


def _check_walk_page(
    progress: _WalkProgress, thumbs: list[tuple[int, str]], has_next: bool
) -> None:
    """Raise :class:`WalkIncomplete` if the next page of ``progress`` breaks a walk rule.

    The page must list images, a page with a Next link must list as many as page 1, and
    no image id may repeat one already walked or another on the page.
    """
    page = progress.last_page + 1
    if not thumbs:
        raise WalkIncomplete(f"browse page {page} lists no images")
    if has_next and progress.page_size and len(thumbs) != progress.page_size:
        raise WalkIncomplete(
            f"browse page {page} lists {len(thumbs)} images, page 1 lists {progress.page_size}"
        )
    ids = [image_id for image_id, _ in thumbs]
    if len(set(ids)) != len(ids) or any(image_id in progress.refs for image_id in ids):
        raise WalkIncomplete(f"browse page {page} repeats an image id")


class SourceWalker:
    """Walk the browse pages of a source, checkpointing the walk in its ``_partial/``."""

    def __init__(
        self,
        auth: CoralNetSession,
        s3: Any,
        *,
        reset_partial: Callable[[int, str, str, int], Any],
    ) -> None:
        """Create the walker; no network call is made.

        Args:
            auth: The CoralNet session whose main session walks the pages.
            s3: S3 client holding the sources' ``_partial/``.
            reset_partial: Called as ``(source_id, bucket, prefix, total)`` when the overview
                count changed during the walk; it discards the source's ``_partial/``.
        """
        self.auth = auth
        self.s3 = s3
        self._reset_partial = reset_partial

    def walk_source(
        self, source_id: int, bucket_name: str, s3_prefix: str, *, total_images: int
    ) -> tuple[list[ImageRef], int]:
        """Walk every browse page of a source and return its images, checked against its count.

        Progress is saved to ``_partial/walk.json`` every
        :data:`_WALK_CHECKPOINT_EVERY_PAGES` pages and when the walk ends. A saved walk
        that ended is returned without any request; an unfinished one continues at
        ``?page=<last page + 1>``.

        A walk is retried up to :data:`_WALK_ATTEMPTS` times. A page that breaks a walk rule
        (see :func:`_check_walk_page`) resumes from the last checkpoint; a page without the
        export form logs the session in again first. When the walk ends, the overview is
        read again: a changed count discards the source's ``_partial/`` and walks against the
        new count; a walk that found another number of images starts again at page 1.

        Args:
            source_id: CoralNet source id.
            bucket_name: Bucket holding the source's ``_partial/``.
            s3_prefix: Key prefix of the scrape.
            total_images: Image count of the overview when the source's download began.

        Returns:
            The images in walk order, and the overview count they match.

        Raises:
            Exception: The last attempt's error once no attempt is left or it is not
                worth repeating, e.g. :class:`WalkIncomplete` or ``requests.HTTPError``.
        """
        expected = total_images
        expired = False
        for attempt in _retrying(_WALK_ATTEMPTS):
            with attempt:
                if expired:
                    self.auth._relogin(self.auth.session)
                    expired = False
                try:
                    refs = self._walk_once(source_id, bucket_name, s3_prefix, expected)
                except SessionExpired:
                    expired = True
                    raise
                except _TotalChanged as e:
                    logger.warning("source %s: %s", source_id, e)
                    expected = e.total
                    raise
        return refs, expected

    def _walk_once(
        self, source_id: int, bucket_name: str, s3_prefix: str, expected: int
    ) -> list[ImageRef]:
        """Finish the source's walk from its saved progress and return its images.

        Raises:
            SessionExpired: If a browse page has no export form.
            WalkIncomplete: If a page breaks a walk rule, the overview count is no longer
                ``expected`` (``_partial/`` is discarded), or the walk found another number
                of images (the walk's progress is discarded).
        """
        key = s3_io.partial_key(s3_prefix, source_id, _WALK_FILE)
        saved = s3_io.read_json_s3(self.s3, bucket_name, key)
        progress = _WalkProgress.from_json(saved) if saved is not None else _WalkProgress()
        if progress.complete:
            return list(progress.refs.values())
        if progress.last_page:
            logger.info("source %s: walk resumes after page %d", source_id, progress.last_page)
        self._walk_pages(source_id, bucket_name, key, progress)
        self._check_walk_total(source_id, bucket_name, s3_prefix, len(progress.refs), expected)
        progress.complete = True
        s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
        logger.info("source %s: walk found %d images", source_id, len(progress.refs))
        return list(progress.refs.values())

    def _walk_pages(
        self, source_id: int, bucket_name: str, key: str, progress: _WalkProgress
    ) -> None:
        """Walk the browse pages after ``progress``, checkpointing it to ``key``.

        Raises:
            SessionExpired: If a browse page has no export form.
            WalkIncomplete: If a page breaks a walk rule.
        """
        url: str | None = browse_page_url(self.auth.CORALNET_URL, source_id, progress.last_page + 1)
        while url is not None:
            response = self.auth.session.get(url, timeout=_PAGE_EXPORT_TIMEOUT)
            response.raise_for_status()
            thumbs, next_rel, has_form = _parse_browse_thumbs(response.text)
            if not has_form:
                raise SessionExpired(
                    f"browse page {progress.last_page + 1} of source {source_id} has no export form"
                )
            _check_walk_page(progress, thumbs, has_next=next_rel is not None)
            progress.add_page(thumbs)
            url = urllib.parse.urljoin(url, next_rel) if next_rel else None
            if url is not None and progress.last_page % _WALK_CHECKPOINT_EVERY_PAGES == 0:
                s3_io.write_json_s3(self.s3, bucket_name, key, progress.to_json())
            if progress.last_page % _BROWSE_PROGRESS_EVERY == 0:
                logger.info("source %s: walked %d pages", source_id, progress.last_page)

    def _check_walk_total(
        self, source_id: int, bucket_name: str, s3_prefix: str, found: int, expected: int
    ) -> None:
        """Read the overview again and check the ``found`` images against it.

        Raises:
            ContentError: If the overview lists no image count.
            _TotalChanged: If the count is no longer ``expected``; ``_partial/`` is discarded
                and a fresh state for the new count is saved.
            _WalkCountMismatch: If ``found`` differs from the count; ``walk.json`` is discarded.
        """
        total = self.auth.fetch_total_images(source_id)
        if total is None:
            raise ContentError(f"source {source_id}: overview lists no image count")
        if total != expected:
            self._reset_partial(source_id, bucket_name, s3_prefix, total)
            raise _TotalChanged(total)
        if found != expected:
            key = s3_io.partial_key(s3_prefix, source_id, _WALK_FILE)
            self.s3.delete_object(Bucket=bucket_name, Key=key)
            raise _WalkCountMismatch(found, expected)
