#!/usr/bin/env python3
"""Fail-closed liveness gate for partial resume.

A partial resume keeps a stopped run's episodes in Neo4j, which is only safe if no
Menhir from that run is still writing. Neo4j is reached via a local ssh -L tunnel, so
a live Menhir holds an ESTABLISHED connection to 127.0.0.1:<bolt port>. Take the proof
BEFORE `menhir serve` starts (startup resets expired episode leases database-wide).

Exit 0 clear, 1 offender found, 2 state could not be determined.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

TIMEOUT_S = 30


def _run(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_S, encoding="utf-8", errors="replace"
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _powershell(script: str) -> str | None:
    return _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])


_SENTINEL = "END-OF-LIVENESS-OUTPUT"


def _parse_tab_lines(out: str) -> list[tuple[int, str]] | None:
    """Rows before the sentinel; None when the sentinel is missing (partial or failed output)."""
    lines = [line.strip("\r\n") for line in out.splitlines()]
    if _SENTINEL not in (line.strip() for line in lines):
        return None
    rows: list[tuple[int, str]] = []
    for line in lines:
        if line.strip() == _SENTINEL:
            break
        if not line.strip():
            continue
        pid_s, _, rest = line.partition("\t")
        try:
            rows.append((int(pid_s.strip()), rest.strip()))
        except ValueError:
            continue
    return rows


def bolt_clients(bolt_port: int) -> list[tuple[int, str]] | None:
    """(pid, process name) with an ESTABLISHED connection to remote port bolt_port."""
    port = int(bolt_port)
    if os.name == "nt":
        # Errors exit 3 (-> None); only "no matching connection" (ObjectNotFound) means empty.
        script = (
            "$ErrorActionPreference = 'Stop'; "
            f"try {{ $c = @(Get-NetTCPConnection -State Established -RemotePort {port}) }} "
            "catch { if ($_.CategoryInfo.Category -eq 'ObjectNotFound') { $c = @() } else { exit 3 } }; "
            "$ids = @($c | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique); "
            "foreach ($i in $ids) { "
            "$p = Get-Process -Id $i -ErrorAction SilentlyContinue; "
            "$n = if ($p) { $p.ProcessName } else { 'unknown' }; "
            "Write-Output (\"$i`t$n\") }; "
            f"Write-Output '{_SENTINEL}'"
        )
        out = _powershell(script)
        if out is None:
            return None
        return _parse_tab_lines(out)
    out = _run(["ss", "-Htnp", "state", "established", f"( dport = :{port} )"])
    if out is None:
        return None
    rows: list[tuple[int, str]] = []
    for m in re.finditer(r'\("([^"]*)",pid=(\d+)', out):
        rows.append((int(m.group(2)), m.group(1)))
    return rows


def serve_processes(serve_port: int) -> list[tuple[int, str]] | None:
    """(pid, command line) of a Menhir 'serve --port <serve_port>', or of an ingest.py driver
    targeting localhost:<serve_port> (a surviving driver would post to the NEW server)."""
    needle = f"serve --port {int(serve_port)}"
    target = f"localhost:{int(serve_port)}"
    if os.name == "nt":
        # Needles are assembled at runtime so this powershell's own command line cannot match them.
        script = (
            "$ErrorActionPreference = 'Stop'; "
            f"try {{ $n = 'serve --port ' + '{int(serve_port)}'; "
            f"$t = 'localhost:' + '{int(serve_port)}'; $d = 'ingest' + '.py'; "
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -and ("
            "$_.CommandLine.Contains($n) -or ($_.CommandLine.Contains($d) -and $_.CommandLine.Contains($t))) } | "
            "ForEach-Object { Write-Output (\"$($_.ProcessId)`t$($_.CommandLine)\") } } "
            "catch { exit 3 }; "
            f"Write-Output '{_SENTINEL}'"
        )
        out = _powershell(script)
        if out is None:
            return None
        return _parse_tab_lines(out)
    out = _run(["ps", "-eo", "pid=,args="])
    if out is None:
        return None
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and (
            needle in parts[1] or ("ingest.py" in parts[1] and target in parts[1])
        ):
            rows.append((int(parts[0]), parts[1]))
    return rows


def _is_ssh(name: str) -> bool:
    return name.strip().lower() in ("ssh", "ssh.exe")


def evaluate(clients, serves, *, own_pid: int) -> tuple[int, str]:
    """Pure decision: (exit code, message)."""
    if clients is None:
        return 2, "[liveness-gate] cannot determine bolt clients; refusing (fail closed)"
    if serves is None:
        return 2, "[liveness-gate] cannot determine serve processes; refusing (fail closed)"
    bad_clients = [(p, n) for p, n in clients if p != own_pid and not _is_ssh(n)]
    bad_serves = [(p, c) for p, c in serves if p != own_pid]
    if not bad_clients and not bad_serves:
        return 0, "[liveness-gate] ok: no client of bolt, no serve on port"
    lines = ["[liveness-gate] REFUSED: a previous Menhir may still be live"]
    for p, n in bad_clients:
        lines.append(f"  bolt client: pid={p} name={n}")
    for p, c in bad_serves:
        lines.append(f"  serve process: pid={p} cmd={c}")
    return 1, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bolt-port", type=int, required=True)
    ap.add_argument("--serve-port", type=int, required=True)
    args = ap.parse_args(argv)
    code, msg = evaluate(
        bolt_clients(args.bolt_port),
        serve_processes(args.serve_port),
        own_pid=os.getpid(),
    )
    if code == 0:
        msg = f"[liveness-gate] ok: no client of bolt :{args.bolt_port}, no serve on :{args.serve_port}"
    print(msg, file=sys.stdout if code == 0 else sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
