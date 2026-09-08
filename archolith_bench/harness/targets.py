"""Allow-list for benchmark targets.

Memory benchmarks ingest, recall and reset against their target. Pointed at a
real instance they would write junk into, and reset, the operator's actual
memories -- so the target check is a safety boundary, not a convenience.

This used to be a deny-list of known-production markers ("prod", "menhir.",
a hardcoded LAN IP). Deny-lists fail OPEN: anything unrecognised is permitted.
That is exactly what happened when production moved to
`https://memory.ctharvey.me` -- a hostname containing none of the markers, so
the guard allowed the real system.

The rule is now inverted. Only loopback is allowed by default, plus hosts
explicitly opted in through ARCHOLITH_BENCH_ALLOW_HOSTS. Everything else is
refused, including hosts nobody has heard of. When production moves again --
and it will -- this fails CLOSED.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

# Loopback only. A throwaway reachable from another machine is not a throwaway.
LOOPBACK_HOSTS: frozenset[str] = frozenset({
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
})

ALLOW_HOSTS_ENV = "ARCHOLITH_BENCH_ALLOW_HOSTS"

# Retained as a second layer. An allow-listed host is still refused if its name
# looks like production -- opting a host in must not be able to opt out of this.
PROD_NAME_MARKERS: tuple[str, ...] = (
    "prod", "production", "menhir.", "staging.", "preprod", "preview", "release",
)


class TargetRefused(RuntimeError):
    """Raised when a benchmark target is not on the allow-list."""


def extra_allowed_hosts() -> frozenset[str]:
    """Hosts opted in via ARCHOLITH_BENCH_ALLOW_HOSTS (comma-separated)."""
    raw = os.environ.get(ALLOW_HOSTS_ENV, "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def parse_host(uri: str) -> str:
    """Extract the hostname from *uri*, with or without a scheme.

    Bare `host:port` has no scheme, so urlsplit would read it as a path; the
    `//` prefix forces netloc parsing. Returns "" when no host can be found,
    which callers must treat as refusable rather than as loopback.
    """
    text = (uri or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "//" + text
    try:
        return (urlsplit(text).hostname or "").lower()
    except ValueError:
        return ""


def assert_allowed_target(uri: str, *, what: str = "target") -> None:
    """Refuse *uri* unless its host is loopback or explicitly opted in.

    Raises TargetRefused with the reason and the opt-in mechanism, so the
    failure explains how to proceed deliberately rather than just blocking.
    """
    host = parse_host(uri)
    if not host:
        raise TargetRefused(
            f"{what} {uri!r} has no parseable host; refusing. "
            f"Use an explicit URL such as http://localhost:8098."
        )

    lowered = (uri or "").lower()
    for marker in PROD_NAME_MARKERS:
        if marker in lowered:
            raise TargetRefused(
                f"{what} {uri!r} contains the production marker {marker!r}; refusing "
                f"even if the host is allow-listed."
            )

    allowed = LOOPBACK_HOSTS | extra_allowed_hosts()
    if host not in allowed:
        raise TargetRefused(
            f"{what} {uri!r} resolves to host {host!r}, which is not on the "
            f"throwaway allow-list ({', '.join(sorted(LOOPBACK_HOSTS))}). "
            f"Memory benchmarks ingest and reset their target, so only loopback is "
            f"permitted by default. To use another throwaway deliberately, set "
            f"{ALLOW_HOSTS_ENV}={host}"
        )
