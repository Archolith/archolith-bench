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
ALLOW_PORTS_ENV = "ARCHOLITH_BENCH_ALLOW_PORTS"

# Default port per scheme. A URI without an explicit port still connects
# somewhere: the neo4j driver dials 7687. Checking the typed string instead of
# the effective address is how `bolt://localhost` slipped past a guard whose
# entire purpose was to refuse port 7687.
SCHEME_DEFAULT_PORTS: dict[str, int] = {
    "bolt": 7687, "bolt+s": 7687, "bolt+ssc": 7687,
    "neo4j": 7687, "neo4j+s": 7687, "neo4j+ssc": 7687,
    "http": 80, "https": 443,
}

# Ports a throwaway is expected on. Allow-listed, like hosts, so an unfamiliar
# port fails closed rather than being assumed safe.
DEFAULT_ALLOWED_PORTS: frozenset[int] = frozenset({
    7688,  # throwaway Neo4j bolt (docker-compose.throwaway-neo4j.yml)
    8098,  # throwaway menhir HTTP
})

# Real services. Refused on every host, and no opt-in can lift them -- the
# escape hatch exists for unfamiliar throwaways, not for production.
RESERVED_REAL_PORTS: dict[int, str] = {
    7687: "the default Neo4j bolt port (a real Menhir graph)",
    8090: "the real local Menhir HTTP API",
}

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


def parse_effective_port(uri: str) -> int | None:
    """Return the port a client will actually dial for *uri*.

    Falls back to the scheme's default when no port is written, because that is
    what the driver does. Returns None only when neither an explicit port nor a
    known scheme default exists -- an unknown destination, which callers must
    refuse rather than wave through.
    """
    text = (uri or "").strip()
    if not text:
        return None
    scheme = ""
    if "://" in text:
        scheme = text.split("://", 1)[0].strip().lower()
    else:
        text = "//" + text
    try:
        parsed = urlsplit(text)
        # .port raises ValueError on a non-numeric or out-of-range port; a
        # target we cannot resolve is a target we must not connect to.
        explicit = parsed.port
    except ValueError:
        return None
    if explicit is not None:
        return explicit
    return SCHEME_DEFAULT_PORTS.get(scheme)


def extra_allowed_ports() -> frozenset[int]:
    """Ports opted in via ARCHOLITH_BENCH_ALLOW_PORTS (comma-separated)."""
    raw = os.environ.get(ALLOW_PORTS_ENV, "")
    ports = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            ports.add(int(chunk))
    return frozenset(ports)


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

    # Host alone is not the destination. Resolve the port a client will really
    # dial, since an omitted port still connects (bolt -> 7687).
    port = parse_effective_port(uri)
    if port is None:
        raise TargetRefused(
            f"{what} {uri!r} has no explicit port and no known default for its "
            f"scheme, so the destination cannot be determined; refusing."
        )

    if port in RESERVED_REAL_PORTS:
        raise TargetRefused(
            f"{what} {uri!r} resolves to port {port}, which is "
            f"{RESERVED_REAL_PORTS[port]}. This is refused on every host and "
            f"cannot be opted out of."
        )

    if port not in (DEFAULT_ALLOWED_PORTS | extra_allowed_ports()):
        raise TargetRefused(
            f"{what} {uri!r} resolves to port {port}, which is not on the "
            f"throwaway port allow-list ({', '.join(str(p) for p in sorted(DEFAULT_ALLOWED_PORTS))}). "
            f"To use another throwaway port deliberately, set "
            f"{ALLOW_PORTS_ENV}={port}"
        )
