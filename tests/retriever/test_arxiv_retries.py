"""Offline retry tests, including the actual pinned arxiv SDK and HTTP transport."""

from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import arxiv
import pytest
import requests

import zotero_arxiv_daily.retriever.arxiv_retriever as retriever


def _atom(paper_ids, total=None):
    entries = "".join(
        f"""<entry>
          <id>https://arxiv.org/abs/{paper_id}</id>
          <title>Test paper {paper_id}</title><summary>Test abstract</summary>
          <updated>2026-10-01T00:00:00Z</updated><published>2026-10-01T00:00:00Z</published>
          <author><name>Test Author</name></author>
          <arxiv:primary_category term="cs.AI"/><category term="cs.AI"/>
          <link href="https://arxiv.org/pdf/{paper_id}" title="pdf" type="application/pdf"/>
        </entry>"""
        for paper_id in paper_ids
    )
    return f"""<feed xmlns="http://www.w3.org/2005/Atom"
        xmlns:arxiv="http://arxiv.org/schemas/atom"
        xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>{len(paper_ids) if total is None else total}</opensearch:totalResults>
      {entries}</feed>""".encode()


@pytest.fixture
def transport(monkeypatch):
    """Mock only HTTP transport and time, leaving SDK parsing/retries intact."""
    state = SimpleNamespace(responses=[], requests=[], sleeps=[], now=1790899200.0)

    def sleep(seconds):
        assert seconds >= 0
        state.sleeps.append(seconds)
        state.now += seconds

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(state.now, tz)

    def send(session, request, **kwargs):
        state.requests.append((request, kwargs, state.now))
        assert state.responses, "Unexpected HTTP request (network is mocked)"
        response = state.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, headers, body = response
        result = requests.Response()
        result.status_code = status
        result.headers.update(headers)
        result._content = body
        result.url = request.url
        return result

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(arxiv, "datetime", Clock)
    monkeypatch.setattr(arxiv.time, "sleep", sleep)
    monkeypatch.setattr(retriever, "sleep", sleep)
    monkeypatch.setattr(retriever, "time", lambda: state.now)
    monkeypatch.setattr(retriever.random, "uniform", lambda _a, _b: 0)
    return state


@pytest.fixture
def client():
    client = retriever._create_arxiv_client()
    yield client
    client._session.close()


def test_429_then_503_then_success_uses_one_retry_layer(client, transport):
    transport.responses = [
        (429, {"Retry-After": "100"}, b""),
        (503, {}, b""),
        (200, {}, _atom(["2610.00001v1"])),
    ]
    papers = retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert [p.get_short_id() for p in papers] == ["2610.00001v1"]
    assert client.num_retries == 0
    assert client.delay_seconds >= 3
    assert len(transport.requests) == 3
    assert transport.sleeps == [100, 60]  # Retry-After does not leak to the 503.
    assert all(kwargs["timeout"] == (10, 60) for _, kwargs, _ in transport.requests)
    assert client._session.get_adapter("https://").max_retries.total == 0


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_exhaustion_is_bounded_and_raises(client, transport, status):
    transport.responses = [(status, {}, b"")] * retriever.ARXIV_MAX_ATTEMPTS
    with pytest.raises(arxiv.HTTPError) as caught:
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert caught.value.status == status
    assert len(transport.requests) == 5
    assert transport.sleeps == [30, 60, 120, 120]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501])
def test_nonretryable_status_fails_immediately(client, transport, status):
    transport.responses = [(status, {}, b"")]
    with pytest.raises(arxiv.HTTPError):
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert len(transport.requests) == 1
    assert transport.sleeps == []


@pytest.mark.parametrize("error", [requests.ConnectionError("offline"), requests.Timeout("slow")])
def test_transport_error_can_recover(client, transport, error):
    transport.responses = [error, (200, {}, _atom(["2610.00001v1"]))]
    assert len(retriever._retrieve_arxiv_batch(client, ["2610.00001"])) == 1
    assert len(transport.requests) == 2
    assert transport.sleeps == [30]


def test_ssl_error_is_not_retried(client, transport):
    transport.responses = [requests.exceptions.SSLError("bad certificate")]
    with pytest.raises(requests.exceptions.SSLError):
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert len(transport.requests) == 1
    assert transport.sleeps == []


@pytest.mark.parametrize("header", ["garbage", "NaN", "inf", "-2", "1e500", ""])
def test_invalid_retry_after_uses_backoff(client, transport, header):
    transport.responses = [
        (503, {"Retry-After": header}, b""),
        (200, {}, _atom(["2610.00001v1"])),
    ]
    retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert transport.sleeps == [30]


def test_retry_after_http_date(client, transport):
    retry_at = datetime.fromtimestamp(transport.now + 90, timezone.utc)
    transport.responses = [
        (503, {"Retry-After": format_datetime(retry_at, usegmt=True)}, b""),
        (200, {}, _atom(["2610.00001v1"])),
    ]
    retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert transport.sleeps == [90]


@pytest.mark.parametrize("seconds, expected", [(0, 30), (10, 30), (300, 300)])
def test_retry_after_boundaries(client, transport, seconds, expected):
    transport.responses = [
        (429, {"Retry-After": str(seconds)}, b""),
        (200, {}, _atom(["2610.00001v1"])),
    ]
    retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert transport.sleeps == [expected]


@pytest.mark.parametrize("header", ["301", "999999999999999999999"])
def test_long_retry_after_aborts_without_retrying_early(client, transport, header):
    transport.responses = [(503, {"Retry-After": header}, b"")]
    with pytest.raises(arxiv.HTTPError):
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert len(transport.requests) == 1
    assert transport.sleeps == []


def test_jitter_is_added_and_backoff_is_capped(client, transport, monkeypatch):
    monkeypatch.setattr(retriever.random, "uniform", lambda _a, _b: 7)
    transport.responses = [(503, {}, b"")] * 5
    with pytest.raises(arxiv.HTTPError):
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert transport.sleeps == [37, 67, 120, 120]


@pytest.mark.parametrize("returned", [[], ["2610.00002v1"], ["2610.00001v2"]])
def test_empty_wrong_or_wrong_version_batch_is_not_accepted(client, transport, returned):
    transport.responses = [
        (200, {}, _atom(returned)),
        (200, {}, _atom(["2610.00001v1"])),
    ]
    result = retriever._retrieve_arxiv_batch(client, ["2610.00001v1"])
    assert [paper.get_short_id() for paper in result] == ["2610.00001v1"]
    assert len(transport.requests) == 2
    assert transport.sleeps == [30]


def test_incomplete_batch_exhaustion_raises(client, transport):
    transport.responses = [(200, {}, _atom([]))] * 5
    with pytest.raises(retriever._IncompleteArxivBatchError):
        retriever._retrieve_arxiv_batch(client, ["2610.00001"])
    assert len(transport.requests) == 5


def test_duplicate_results_cannot_hide_missing_papers(client, transport):
    transport.responses = [
        (200, {}, _atom(["2610.00001v1", "2610.00001v1"])),
        (200, {}, _atom(["2610.00001v1", "2610.00002v1"])),
    ]
    result = retriever._retrieve_arxiv_batch(client, ["2610.00001", "2610.00002"])
    assert len({paper.entry_id for paper in result}) == 2
    assert transport.sleeps == [30]


def test_partial_generator_failure_does_not_duplicate_results(client, transport):
    transport.responses = [
        (200, {}, _atom(["2610.00001v1"], total=2)),
        (503, {}, b""),
        (200, {}, _atom(["2610.00001v1", "2610.00002v1"])),
    ]
    result = retriever._retrieve_arxiv_batch(client, ["2610.00001", "2610.00002"])
    assert [paper.get_short_id() for paper in result] == ["2610.00001v1", "2610.00002v1"]
    assert len(transport.requests) == 3
    assert transport.sleeps == [10, 30]


def test_unexpected_empty_later_page_can_recover(client, transport):
    transport.responses = [
        (200, {}, _atom(["2610.00001v1"], total=2)),
        (200, {}, _atom([], total=2)),
        (200, {}, _atom(["2610.00001v1", "2610.00002v1"])),
    ]
    result = retriever._retrieve_arxiv_batch(client, ["2610.00001", "2610.00002"])
    assert len(result) == 2
    assert len(transport.requests) == 3
    assert transport.sleeps == [10, 30]


def test_successive_batches_are_serial_and_rate_limited(config, transport, monkeypatch):
    paper_ids = [f"2610.{i:05d}" for i in range(21)]
    feed = SimpleNamespace(
        feed=SimpleNamespace(title="arXiv"),
        entries=[dict(id=f"oai:arXiv.org:{pid}", arxiv_announce_type="new") for pid in paper_ids],
    )
    # Match feedparser's dict + attribute interface.
    feed.entries = [arxiv.feedparser.FeedParserDict(entry) for entry in feed.entries]
    original_parse = retriever.feedparser.parse
    monkeypatch.setattr(
        retriever.feedparser, "parse",
        lambda source: feed if isinstance(source, str) else original_parse(source),
    )
    transport.responses = [
        (200, {}, _atom([pid + "v1" for pid in paper_ids[:20]])),
        (200, {}, _atom([paper_ids[-1] + "v1"])),
    ]
    papers = retriever.ArxivRetriever(config)._retrieve_raw_papers()
    assert len(papers) == 21
    assert len(transport.requests) == 2
    assert transport.requests[1][2] - transport.requests[0][2] >= 3
    assert transport.sleeps == [10, 10]  # RSS to API, then API to API.
    ids_requested = [parse_qs(urlparse(req.url).query)["id_list"][0].split(",")
                     for req, _, _ in transport.requests]
    assert ids_requested == [paper_ids[:20], paper_ids[20:]]


def test_progress_and_session_close_when_a_batch_fails(config, mock_feedparser, monkeypatch):
    closed = []
    monkeypatch.setattr(retriever, "sleep", lambda _: None)

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append("progress")

        def update(self, _):
            pytest.fail("Failed batch must not update progress")

    client = SimpleNamespace(_session=SimpleNamespace(close=lambda: closed.append("session")))
    monkeypatch.setattr(retriever, "_create_arxiv_client", lambda: client)
    monkeypatch.setattr(retriever, "tqdm", lambda **_: Progress())

    def fail(*_):
        raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 503)

    monkeypatch.setattr(retriever, "_retrieve_arxiv_batch", fail)
    with pytest.raises(arxiv.HTTPError):
        retriever.ArxivRetriever(config)._retrieve_raw_papers()
    assert closed == ["progress", "session"]


def test_empty_rss_does_not_start_api_client(config, transport, monkeypatch):
    feed = SimpleNamespace(feed=SimpleNamespace(title="arXiv"), entries=[])
    monkeypatch.setattr(retriever.feedparser, "parse", lambda _: feed)
    monkeypatch.setattr(retriever, "_create_arxiv_client", lambda: pytest.fail("No API client needed"))
    assert retriever.ArxivRetriever(config)._retrieve_raw_papers() == []
    assert transport.requests == []
    assert transport.sleeps == []
