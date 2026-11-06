"""The shared bounded downloader and its three callers."""

from __future__ import annotations

import hashlib
import http.client
import io
import urllib.request

import pytest

import bounded_download
import build_deepfilter

HOSTS = frozenset({"allowed.example"})
URL = "https://allowed.example/file.bin"


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str = URL) -> None:
        super().__init__(payload)
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def geturl(self) -> str:
        return self._url


def _serve(monkeypatch, payload: bytes, url: str = URL) -> list[object]:
    handlers: list[object] = []

    class Opener:
        def open(self, request, timeout):
            return _Response(payload, url)

    def build_opener(*given):
        handlers.extend(given)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    return handlers


def _download(destination, **overrides):
    options = {
        "hosts": HOSTS, "max_bytes": 100, "timeout": 5,
        "refusal": "host refused", "too_large": "payload too large",
    }
    options.update(overrides)
    return bounded_download.download(URL, destination, **options)


def test_download_returns_the_byte_count_and_sha256_of_what_it_wrote(tmp_path, monkeypatch):
    _serve(monkeypatch, b"payload")
    destination = tmp_path / "file.bin"

    assert _download(destination) == (7, hashlib.sha256(b"payload").hexdigest())
    assert destination.read_bytes() == b"payload"


@pytest.mark.parametrize(
    "url",
    [
        "http://allowed.example/file.bin",
        "https://other.example/file.bin",
        "https://user:secret@allowed.example/file.bin",
        "https://allowed.example:8443/file.bin",
        "https://allowed.example/file.bin#fragment",
    ],
)
def test_download_refuses_a_url_outside_the_allowed_hosts_before_connecting(
    tmp_path, monkeypatch, url
):
    def refuse_to_connect(*_):
        raise AssertionError("must not connect")

    monkeypatch.setattr(urllib.request, "build_opener", refuse_to_connect)
    destination = tmp_path / "file.bin"

    with pytest.raises(ValueError, match="host refused"):
        bounded_download.download(
            url, destination, hosts=HOSTS, max_bytes=100, timeout=5,
            refusal="host refused", too_large="payload too large",
        )
    assert not destination.exists()


def test_download_stops_at_the_cap_and_leaves_nothing_behind(tmp_path, monkeypatch):
    _serve(monkeypatch, b"x" * 101)
    destination = tmp_path / "file.bin"

    with pytest.raises(RuntimeError, match="payload too large"):
        _download(destination)
    assert not destination.exists()


def test_download_accepts_exactly_the_cap(tmp_path, monkeypatch):
    _serve(monkeypatch, b"x" * 100)

    assert _download(tmp_path / "file.bin")[0] == 100


def test_download_without_redirects_installs_the_refusing_handler(tmp_path, monkeypatch):
    handlers = _serve(monkeypatch, b"payload")
    _download(tmp_path / "file.bin")

    assert handlers == [bounded_download.NoRedirectHandler]


def test_download_that_follows_redirects_refuses_a_final_url_on_another_host(
    tmp_path, monkeypatch
):
    _serve(monkeypatch, b"payload", url="https://other.example/file.bin")
    destination = tmp_path / "file.bin"

    with pytest.raises(ValueError, match="host refused"):
        _download(destination, follow_redirects=True)
    assert not destination.exists()


def test_the_redirect_handler_follows_only_allowed_hosts():
    handler = bounded_download.allowed_host_redirects(HOSTS, "host refused")()
    request = urllib.request.Request(URL)

    with pytest.raises(ValueError, match="host refused"):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), "https://evil.example/x"
        )
    followed = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), "https://allowed.example/other.bin"
    )
    assert followed is not None and followed.full_url == "https://allowed.example/other.bin"


def test_build_deepfilter_downloads_the_tract_archive_through_the_bounded_downloader(
    tmp_path, monkeypatch
):
    payload = b"crate-bytes"
    _serve(monkeypatch, payload, url="https://static.crates.io/crates/tract-linalg/x.crate")
    archive = tmp_path / "cache" / "tract.crate"

    build_deepfilter.ensure_tract_archive(
        archive, tmp_path / "cargo-home", "tract.crate", hashlib.sha256(payload).hexdigest(),
        "https://crates.io/api/v1/crates/tract-linalg/0.21.17/download",
    )

    assert archive.read_bytes() == payload


def test_build_deepfilter_refuses_a_tract_archive_url_on_another_host(tmp_path, monkeypatch):
    _serve(monkeypatch, b"crate-bytes", url="https://evil.example/x.crate")

    with pytest.raises(ValueError, match="crates.io"):
        build_deepfilter.ensure_tract_archive(
            tmp_path / "tract.crate", tmp_path / "cargo-home", "tract.crate", "0" * 64,
            "https://evil.example/x.crate",
        )
    assert not (tmp_path / "tract.crate").exists()


def test_build_deepfilter_caps_the_tract_archive_size(tmp_path, monkeypatch):
    monkeypatch.setattr(build_deepfilter, "TRACT_ARCHIVE_MAX_BYTES", 4)
    _serve(monkeypatch, b"too-large", url="https://static.crates.io/x.crate")

    with pytest.raises(RuntimeError, match="exceeded 4 bytes"):
        build_deepfilter.ensure_tract_archive(
            tmp_path / "tract.crate", tmp_path / "cargo-home", "tract.crate", "0" * 64,
            "https://crates.io/api/v1/crates/tract-linalg/0.21.17/download",
        )
    assert not (tmp_path / "tract.crate").exists()
