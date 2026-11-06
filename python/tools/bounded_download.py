"""Download one HTTPS file from an allowed set of hosts, with a hard size cap."""

from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request
from collections.abc import Collection
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
USER_AGENT = "AudioForge-release-assets"


def is_allowed_https_url(url: str, hosts: Collection[str]) -> bool:
    """Whether *url* is plain HTTPS to one of *hosts*, without credentials or a fragment."""

    parsed = urllib.parse.urlsplit(url)
    return not (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    )


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def allowed_host_redirects(hosts: Collection[str], refusal: str) -> type[urllib.request.HTTPRedirectHandler]:
    """A redirect handler that follows only redirects to *hosts* and raises *refusal* otherwise."""

    class AllowedHostRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            target = urllib.parse.urljoin(req.full_url, newurl)
            if not is_allowed_https_url(target, hosts):
                raise ValueError(refusal)
            return super().redirect_request(req, fp, code, msg, headers, target)

    return AllowedHostRedirectHandler


def download(
    url: str,
    destination: Path,
    *,
    hosts: Collection[str],
    max_bytes: int,
    timeout: float,
    refusal: str,
    too_large: str,
    follow_redirects: bool = False,
    exclusive: bool = False,
) -> tuple[int, str]:
    """Write *url* to *destination* and return its byte count and SHA-256.

    The URL, and with *follow_redirects* every redirect and the final URL, must
    be HTTPS on one of *hosts*; without it a redirect is an error. More than
    *max_bytes* raises RuntimeError(*too_large*). With *exclusive* the
    destination must not exist yet. Nothing is left at *destination* on failure. Callers still verify the bytes against a pinned hash.
    """

    if not is_allowed_https_url(url, hosts):
        raise ValueError(refusal)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    handler = allowed_host_redirects(hosts, refusal) if follow_redirects else NoRedirectHandler
    opener = urllib.request.build_opener(handler)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with opener.open(request, timeout=timeout) as response, destination.open("xb" if exclusive else "wb") as output:
            if follow_redirects and not is_allowed_https_url(response.geturl(), hosts):
                raise ValueError(refusal)
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    raise RuntimeError(too_large)
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return byte_count, digest.hexdigest()
