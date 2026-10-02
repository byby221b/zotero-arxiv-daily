from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import math
import random
import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from queue import Empty
from time import sleep, time
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
ARXIV_BATCH_SIZE = 20
ARXIV_REQUEST_DELAY = 10
ARXIV_MAX_ATTEMPTS = 5
ARXIV_BACKOFF_BASE = 30
ARXIV_BACKOFF_CAP = 120
ARXIV_MAX_RETRY_AFTER = 300
ARXIV_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class _ArxivSession(requests.Session):
    """Keep response headers that arxiv 2.4.1's HTTPError discards."""

    def __init__(self):
        super().__init__()
        self.retry_after: str | None = None

    def get(self, url, **kwargs):
        # Reset before every request, including requests that raise a timeout.
        self.retry_after = None
        kwargs.setdefault("timeout", DOWNLOAD_TIMEOUT)
        response = super().get(url, **kwargs)
        self.retry_after = response.headers.get("Retry-After")
        return response


def _create_arxiv_client() -> arxiv.Client:
    # The outer batch loop is the only retry owner. The SDK still serializes
    # requests and enforces spacing, including any pagination within a batch.
    client = arxiv.Client(
        page_size=ARXIV_BATCH_SIZE, num_retries=0, delay_seconds=ARXIV_REQUEST_DELAY
    )
    # arxiv 2.4.1 has no public timeout/session option. Keep this small adapter
    # covered by a real-SDK test when upgrading the dependency.
    client._session.close()
    client._session = _ArxivSession()
    return client


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = max(0.0, retry_at.timestamp() - time())
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class _IncompleteArxivBatchError(RuntimeError):
    pass


def _validate_arxiv_batch(batch: list[ArxivResult], paper_ids: list[str]) -> None:
    """Do not mistake an empty/partial API response for a successful batch."""
    returned_ids = [paper.entry_id.split("arxiv.org/abs/")[-1] for paper in batch]
    requested_ids = set(paper_ids)
    # RSS IDs can be unversioned. Explicit versions must still match exactly.
    matched_ids = [
        result_id if result_id in requested_ids else re.sub(r"v\d+$", "", result_id)
        for result_id in returned_ids
    ]
    if len(matched_ids) != len(paper_ids) or set(matched_ids) != requested_ids:
        raise _IncompleteArxivBatchError(
            f"arXiv returned incomplete or mismatched metadata for {len(paper_ids)} requested papers"
        )


def _retrieve_arxiv_batch(client: arxiv.Client, paper_ids: list[str]) -> list[ArxivResult]:
    search = arxiv.Search(id_list=paper_ids, max_results=len(paper_ids))
    for attempt in range(ARXIV_MAX_ATTEMPTS):
        try:
            # Materialize before committing results, so a generator failure
            # cannot duplicate or leak a partially retrieved batch downstream.
            batch = list(client.results(search))
            _validate_arxiv_batch(batch, paper_ids)
            return batch
        except (
            arxiv.HTTPError,
            arxiv.UnexpectedEmptyPageError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            _IncompleteArxivBatchError,
        ) as exc:
            if isinstance(exc, requests.exceptions.SSLError):
                raise
            if isinstance(exc, arxiv.HTTPError) and exc.status not in ARXIV_RETRYABLE_STATUSES:
                raise
            if attempt == ARXIV_MAX_ATTEMPTS - 1:
                logger.error(f"arXiv metadata batch failed after {ARXIV_MAX_ATTEMPTS} attempts: {exc}")
                raise
            retry_after = (
                _retry_after_seconds(client._session.retry_after)
                if isinstance(exc, arxiv.HTTPError) else None
            )
            if retry_after is not None and retry_after > ARXIV_MAX_RETRY_AFTER:
                # Do not shorten a server's cooldown to fit our retry budget.
                logger.error(f"arXiv requested a {retry_after:.0f}s cooldown; stopping this run")
                raise
            backoff = min(ARXIV_BACKOFF_CAP, ARXIV_BACKOFF_BASE * 2 ** attempt)
            wait = min(ARXIV_BACKOFF_CAP, backoff + random.uniform(0, 10))
            wait = max(wait, retry_after or 0)
            logger.warning(
                f"arXiv metadata batch failed ({exc}); retry {attempt + 2}/{ARXIV_MAX_ATTEMPTS} in {wait:.1f}s"
            )
            sleep(wait)


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = list(dict.fromkeys(
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ))
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
        if not all_paper_ids:
            return []

        # Get full information of each paper from arxiv api
        # Include the preceding RSS request in our conservative pacing.
        sleep(ARXIV_REQUEST_DELAY)
        client = _create_arxiv_client()
        try:
            with tqdm(total=len(all_paper_ids)) as bar:
                for i in range(0, len(all_paper_ids), ARXIV_BATCH_SIZE):
                    batch = _retrieve_arxiv_batch(client, all_paper_ids[i:i + ARXIV_BATCH_SIZE])
                    bar.update(len(batch))
                    raw_papers.extend(batch)
        finally:
            client._session.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
