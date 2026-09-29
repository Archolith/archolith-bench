"""Live benchmark dashboard for Mode-B memory runs.

A full run takes hours; tailing a log is not real tracking. This reads the same
append-only checkpoint the run writes (one record per completed (arm, item)) for
progress + running accuracy, and optionally probes the throwaway menhir for live
activity (queue depth, enrichment) so you can see it is actually working between
item completions. Read-only: it never touches the running benchmark or its store.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

# Known full-dataset item counts (per arm) so progress shows a % without a network
# load. Override with --total-items when running a subset.
_KNOWN_VARIANT_ITEMS = {"oracle": 500, "s": 500, "m": 500}

# First-seen wall-clock per (checkpoint, arm, task_id). Used when a checkpoint record has
# no `ts` (older runs predating the timestamped format): the dashboard stamps an item the
# first refresh it observes it, so a live run still gets per-item times.
_FIRST_SEEN: dict[tuple[str, str, str], float] = {}


@dataclass
class ArmAgg:
    n: int = 0
    correct: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def score(self) -> float:
        return (self.correct / self.n) if self.n else 0.0


@dataclass
class RunSnapshot:
    checkpoint: Path
    benchmark: str
    variant: str
    model: str
    arms: dict[str, ArmAgg] = field(default_factory=dict)
    mtime: float = 0.0
    source: str = ""  # the checkpoint's folder, distinguishes same-named runs (e.g. ext-deepseek)
    items: list[dict] = field(default_factory=list)  # per-item feed in completion order

    @property
    def total_done(self) -> int:
        return sum(a.n for a in self.arms.values())

    @property
    def lift(self) -> float | None:
        """menhir_recall score minus the no_memory floor, when both arms are present."""
        base = self.arms.get("no_memory")
        mem = self.arms.get("menhir_recall")
        if base is None or mem is None or base.n == 0 or mem.n == 0:
            return None
        return mem.score - base.score


@dataclass
class IngestSnapshot:
    """Completed LongMemEval graph-ingest items recorded in a run manifest."""

    manifest: Path
    source: str
    items: list[dict] = field(default_factory=list)
    mtime: float = 0.0

    @property
    def completed(self) -> int:
        return len(self.items)


def _parse_checkpoint_name(path: Path) -> tuple[str, str, str]:
    """`.checkpoint_<benchmark>_<variant>_<model>.jsonl` -> (benchmark, variant, model)."""
    stem = path.name
    if stem.startswith(".checkpoint_"):
        stem = stem[len(".checkpoint_"):]
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    parts = stem.split("_")
    if len(parts) >= 3:
        # benchmark may itself contain no underscore (longmemeval-menhir); variant is
        # the second-to-last segment, model the last, benchmark the rest.
        model = parts[-1]
        variant = parts[-2]
        benchmark = "_".join(parts[:-2])
        return benchmark, variant, model
    return stem, "?", "?"


def read_checkpoint(path: Path) -> RunSnapshot:
    benchmark, variant, model = _parse_checkpoint_name(path)
    snap = RunSnapshot(checkpoint=path, benchmark=benchmark, variant=variant, model=model,
                       mtime=path.stat().st_mtime if path.exists() else 0.0,
                       source=path.parent.name)
    if not path.exists():
        return snap
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                arm = str(rec["arm"])
                res = rec["result"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # tolerate a torn final line
            agg = snap.arms.setdefault(arm, ArmAgg())
            agg.n += 1
            agg.correct += 1 if res.get("correct") else 0
            agg.input_tokens += int(res.get("input_tokens") or 0)
            agg.output_tokens += int(res.get("output_tokens") or 0)
            task_id = str(res.get("task_id") or rec.get("task_id") or "")
            ts = rec.get("ts")
            if ts is None:
                key = (str(path), arm, task_id)
                ts = _FIRST_SEEN.setdefault(key, time.time())
            snap.items.append({
                "arm": arm,
                "task_id": task_id,
                "correct": bool(res.get("correct")),
                "resp": (res.get("response_text") or "").strip().replace("\n", " "),
                "out_tok": int(res.get("output_tokens") or 0),
                "ts": float(ts),
                "question": (res.get("question") or "").strip(),
                "recalled": (res.get("recalled") or "").strip(),
                "gold": (res.get("gold") or "").strip(),
            })
    return snap


def read_ingest_manifest(path: Path) -> IngestSnapshot:
    snap = IngestSnapshot(
        manifest=path,
        source=path.parent.name,
        mtime=path.stat().st_mtime if path.exists() else 0.0,
    )
    if not path.exists():
        return snap
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return snap
    if isinstance(rows, list):
        snap.items = [row for row in rows if isinstance(row, dict)]
    return snap


def scan_ingests(results_dir: Path, *, active_within_s: float | None = None) -> list[IngestSnapshot]:
    if not results_dir.exists():
        return []
    paths = {p for p in results_dir.glob("manifest.json")}
    paths |= {p for p in results_dir.rglob("manifest.json")}
    snaps = [read_ingest_manifest(p) for p in sorted(paths)]
    if active_within_s:
        cutoff = time.time() - active_within_s
        snaps = [s for s in snaps if s.mtime >= cutoff]
    snaps.sort(key=lambda s: s.mtime, reverse=True)
    return snaps


def scan_runs(results_dir: Path, *, active_within_s: float | None = None) -> list[RunSnapshot]:
    if not results_dir.exists():
        return []
    # rglob so isolated per-config runs in subfolders (results/ext-deepseek/...) show too.
    paths = {p for p in results_dir.glob(".checkpoint_*.jsonl")}
    paths |= {p for p in results_dir.rglob(".checkpoint_*.jsonl")}
    snaps = [read_checkpoint(p) for p in sorted(paths)]
    if active_within_s:
        # "Active" = checkpoint written within the window (a run currently producing items).
        cutoff = time.time() - active_within_s
        snaps = [s for s in snaps if s.mtime >= cutoff]
    snaps.sort(key=lambda s: s.mtime, reverse=True)
    return snaps


def probe_menhir(url: str, timeout: float = 3.0) -> dict | None:
    """Best-effort live menhir state (health + queue/enrichment). None if unreachable."""
    try:
        import httpx
    except ImportError:
        return None
    base = url.rstrip("/")
    out: dict = {}
    try:
        with httpx.Client(timeout=timeout) as c:
            h = c.get(f"{base}/api/health")
            out["health"] = h.status_code == 200
            try:
                s = c.get(f"{base}/api/stats", params={"since_hours": 1})
                if s.status_code == 200:
                    d = s.json()
                    out["queue_depth"] = d.get("queue_depth")
                    out["startup_mode"] = d.get("startup_mode")
                    enr = d.get("enrichment") or {}
                    out["enrichment"] = enr
            except httpx.HTTPError:
                pass
    except httpx.HTTPError:
        return None
    return out


def _bar(done: int, total: int | None, width: int = 28) -> str:
    if not total or total <= 0:
        return f"[{'?' * width}] {done}"
    frac = min(1.0, done / total)
    filled = int(frac * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({frac * 100:.1f}%)"


def render(
    snaps: list[RunSnapshot],
    menhir: dict | None,
    *,
    total_items: int | None,
    rate_per_min: float | None,
    eta_min: float | None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"  archolith-bench memory dashboard   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)
    if not snaps:
        lines.append("  (no checkpoints found -- start a run with --resume)")
        return "\n".join(lines)

    for s in snaps:
        per_arm_total = total_items or _KNOWN_VARIANT_ITEMS.get(s.variant)
        n_arms = max(1, len(s.arms))
        grand_total = per_arm_total * n_arms if per_arm_total else None
        lines.append("")
        src = f"  [{s.source}]" if s.source and s.source != "results" else ""
        lines.append(f"  {s.benchmark}{src}  variant={s.variant}  answer-model={s.model}")
        lines.append(f"  progress {_bar(s.total_done, grand_total)}")
        if rate_per_min:
            eta_txt = f"  ETA ~{eta_min:.0f} min" if eta_min is not None else ""
            lines.append(f"  rate {rate_per_min:.1f} items/min{eta_txt}")
        lines.append(f"  {'arm':<16}{'done':>6}{'acc':>9}{'in_tok':>12}{'out_tok':>10}")
        for arm in sorted(s.arms):
            a = s.arms[arm]
            done_txt = f"{a.n}/{per_arm_total}" if per_arm_total else str(a.n)
            lines.append(f"  {arm:<16}{done_txt:>6}{a.score:>9.3f}{a.input_tokens:>12,}{a.output_tokens:>10,}")
        if s.lift is not None:
            lines.append(f"  memory lift (menhir_recall - no_memory): {s.lift:+.3f}")

    lines.append("")
    if menhir is not None:
        if menhir.get("health"):
            qd = menhir.get("queue_depth")
            mode = menhir.get("startup_mode")
            enr = menhir.get("enrichment") or {}
            rate = enr.get("episodes_per_min") or enr.get("rate") or enr.get("per_min")
            extra = f"  enrich~{rate}/min" if rate else ""
            lines.append(f"  menhir: UP  mode={mode}  queue_depth={qd}{extra}")
        else:
            lines.append("  menhir: probe returned non-200 (starting or degraded)")
    else:
        lines.append("  menhir: not probed (pass --menhir-url for live activity)")
    lines.append("  note: token/cost columns are ANSWER-model only; menhir ingestion")
    lines.append("        (OpenAI extraction+embedding) spend is not tracked by the bench.")
    lines.append("=" * 64)
    return "\n".join(lines)


def _esc(s: object) -> str:
    import html
    return html.escape(str(s))


def _slug_id(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts)
    return "d_" + "".join(c if c.isalnum() else "_" for c in raw)[:80]


def _task_path(task_id: object) -> str:
    """Stable dashboard path for a benchmark item or graph namespace."""
    raw = str(task_id)
    namespace = raw if raw.startswith("lme-") else f"lme-{raw}"
    return f"/tasks/{quote(namespace, safe='')}"


def _feed_rows_html(s: RunSnapshot, items_n: int) -> str:
    """Newest-first per-item feed (turn-by-turn) for one run."""
    if not items_n or not s.items:
        return ""
    runkey = f"{s.model}_{s.variant}_{s.source}"
    recent = list(reversed(s.items))[:items_n]
    cells = ""
    for it in recent:
        mark = "&#10003;" if it["correct"] else "&#10007;"
        cls = "ok" if it["correct"] else "no"
        ts = it.get("ts")
        tstr = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
        full = it["resp"]
        q = it.get("question", "")
        recalled = it.get("recalled", "")
        gold = it.get("gold", "")
        task_path = _task_path(it["task_id"])
        task_link = (
            f"<a class='task-link' href='{_esc(task_path)}' "
            f"title='Open task page'>{_esc(it['task_id'][:24])}<span aria-hidden='true'> ↗</span></a>"
        )
        # If we captured question/retrieval (newer runs), make the answer cell expandable
        # to show question -> retrieved memory -> gold -> response. Older runs (no capture)
        # just show the answer text.
        if q or recalled:
            # Summary shows the QUESTION (visible at a glance); expanding reveals what was
            # retrieved, the gold answer, and the model's response.
            did = _slug_id(runkey, it["arm"], it["task_id"])
            snips = [ln for ln in recalled.split("\n") if ln.strip()]
            if snips:
                snip_html = "".join(
                    f"<div class='snip'><span class='snum'>{i + 1}</span>{_esc(ln)}</div>"
                    for i, ln in enumerate(snips)
                )
                recalled_block = (
                    f"<span class='lbl'>retrieved memory ({len(snips)})</span>"
                    f"<div id='{did}_pre' class='keepscroll snips'>{snip_html}</div>"
                )
            else:
                recalled_block = (
                    "<span class='lbl'>retrieved memory</span>"
                    "<div class='muted'><em>(nothing recalled)</em></div>"
                )
            detail = (
                f"<summary>{_esc(q) or _esc(full)}</summary>"
                f"<div class='blk'>{recalled_block}</div>"
                f"<div class='blk'><span class='lbl'>gold answer</span><div>{_esc(gold)}</div></div>"
                f"<div class='blk'><span class='lbl'>llm response</span><div>{_esc(full)}</div></div>"
            )
            ans_cell = f"<details id='{did}'>{detail}</details>"
        else:
            ans_cell = f"<span title='{_esc(full)}'>{_esc(full)}</span>"
        cells += (
            f"<tr><td class='muted'>{tstr}</td><td class='{cls}'>{mark}</td><td>{_esc(it['arm'])}</td>"
            f"<td class='muted'>{task_link}</td>"
            f"<td class='ans'>{ans_cell}</td></tr>"
        )
    return (
        "<table class='feed'><thead><tr><th>time</th><th></th><th>arm</th><th>item ↗ task</th>"
        "<th>question &middot; click to expand retrieval &rarr; gold &rarr; response</th>"
        "</tr></thead><tbody>" + cells + "</tbody></table>"
    )


def _ingest_rows_html(ingests: list[IngestSnapshot], total_items: int | None) -> str:
    rows: list[str] = []
    for ingest in ingests:
        done = ingest.completed
        pct = (done / total_items * 100.0) if total_items else None
        progress = (
            f'<div class="bar"><div class="fill" style="width:{min(100.0, pct):.1f}%"></div></div>'
            f'<span class="muted">{done}/{total_items} ({pct:.1f}%)</span>'
            if pct is not None
            else f'<span class="muted">{done} completed</span>'
        )
        ready = sum(int(item.get("ready") or 0) for item in ingest.items)
        failed = sum(int(item.get("failed_remaining") or 0) for item in ingest.items)
        turn_evidence = sum(int(item.get("turn_evidence") or 0) for item in ingest.items)
        scalar_views = sum(int(item.get("scalar_views") or 0) for item in ingest.items)
        latest = ingest.items[-1] if ingest.items else {}
        latest_question = latest.get("question") or latest.get("question_id") or "none yet"
        if total_items and done >= total_items:
            phase_note = "Graph ingest complete; recall/QA scoring should begin next."
        else:
            phase_note = "Graph ingest is active; accuracy appears after recall/QA checkpointing begins."
        rows.append(
            f"<div class='run'><h2>LongMemEval graph ingest "
            f"<span class='src'>[{_esc(ingest.source)}]</span></h2>"
            f"<div class='prog'>{progress}</div>"
            "<table><thead><tr><th>completed</th><th>ready episodes</th><th>failed</th>"
            "<th>turn evidence</th><th>scalar views</th></tr></thead>"
            f"<tbody><tr><td>{done}</td><td>{ready}</td><td>{failed}</td>"
            f"<td>{turn_evidence}</td><td>{scalar_views}</td></tr></tbody></table>"
            f"<div class='muted'>latest completed: {_esc(latest_question)}</div>"
            f"<div class='muted'>{phase_note}</div></div>"
        )
    return "".join(rows)


EPISODE_PROGRESS_FILE = "ama_ingest_progress.json"


def scan_episode_progress(results_dir: Path) -> dict | None:
    """Newest per-episode ingest progress file (written by scripts/ama/progress_ama.py)."""
    if not results_dir.exists():
        return None
    paths = sorted(results_dir.rglob(EPISODE_PROGRESS_FILE), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("episodes"), list):
            return data
    return None


def _episode_progress_html(progress: dict | None) -> str:
    if not progress:
        return ""
    episodes = [e for e in progress["episodes"] if isinstance(e, dict)]
    total = sum(int(e.get("steps_total") or 0) for e in episodes)
    done = sum(int(e.get("ready") or 0) + int(e.get("failed") or 0) for e in episodes)
    failed = sum(int(e.get("failed") or 0) for e in episodes)
    finished = sum(
        1 for e in episodes
        if int(e.get("ready") or 0) + int(e.get("failed") or 0) >= int(e.get("steps_total") or 0) > 0
    )
    pct = (done / total * 100.0) if total else 0.0
    rows = ""
    for e in episodes:
        steps = int(e.get("steps_total") or 0)
        ep_done = int(e.get("ready") or 0) + int(e.get("failed") or 0)
        ep_pct = (ep_done / steps * 100.0) if steps else 0.0
        rows += (
            f"<tr><td>{_esc(e.get('namespace'))}</td><td>{_esc(e.get('domain'))}</td>"
            f"<td><div class='bar'><div class='fill' style='width:{min(100.0, ep_pct):.1f}%'></div></div></td>"
            f"<td class='num'>{ep_done}/{steps}</td><td class='num'>{int(e.get('failed') or 0)}</td>"
            f"<td class='num'>{int(e.get('in_flight') or 0)}</td></tr>"
        )
    return (
        f"<div class='run'><h2>Episode ingest <span class='src'>[{_esc(progress.get('run'))}]</span></h2>"
        f"<div class='prog'><div class='bar'><div class='fill' style='width:{min(100.0, pct):.1f}%'></div></div>"
        f"<span class='muted'>{done}/{total} steps ({pct:.1f}%) &middot; {finished}/{len(episodes)} episodes "
        f"finished &middot; {failed} failed steps</span></div>"
        "<table><thead><tr><th>namespace</th><th>domain</th><th>progress</th><th>steps</th>"
        f"<th>failed</th><th>in flight</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<div class='muted'>graph snapshot {_esc(progress.get('generated_at'))}</div></div>"
    )


def _task_directory_html(tasks: list[dict]) -> str:
    """Searchable index of every completed manifest task and its score state."""
    rows: list[str] = []
    status_counts = {"correct": 0, "incorrect": 0, "unscored": 0}
    for task in tasks:
        scores = {
            str(score.get("arm") or ""): score
            for score in task.get("scoring") or []
        }
        memory_score = scores.get("menhir_recall")
        baseline_score = scores.get("no_memory")
        if memory_score is None:
            score_status = "unscored"
        else:
            score_status = "correct" if memory_score.get("correct") else "incorrect"
        status_counts[score_status] += 1

        def score_badge(label: str, score: dict | None) -> str:
            if score is None:
                return f"<span class='score-pill pending-score'>{label} pending</span>"
            if score.get("correct"):
                return f"<span class='score-pill score-correct'>{label} &#10003;</span>"
            return f"<span class='score-pill score-incorrect'>{label} &#10007;</span>"

        namespace = str(task.get("namespace") or "")
        question_id = str(task.get("question_id") or namespace)
        question = str(task.get("question") or namespace)
        question_type = str(task.get("question_type") or "task")
        search_text = " ".join((
            namespace,
            question_id,
            question,
            question_type,
            str(task.get("answer") or ""),
        )).lower()
        graph_available = task.get("graph_available")
        graph_badge = ""
        if graph_available is True:
            graph_badge = "<span class='directory-chip graph-ready'>graph ready</span>"
        elif graph_available is False:
            graph_badge = "<span class='directory-chip graph-missing'>graph missing</span>"
        rows.append(
            f"<article class='task-directory-row' data-task-row data-score='{score_status}' "
            f"data-search='{_esc(search_text)}'>"
            f"<div class='directory-id'><a href='{_esc(_task_path(namespace))}'>{_esc(question_id)} ↗</a>"
            f"<span>{_esc(namespace)}</span></div>"
            f"<div class='directory-question'><span>{_esc(question_type)}</span>"
            f"<h3><a href='{_esc(_task_path(namespace))}'>{_esc(question)}</a></h3>"
            f"<div class='directory-counts'><span>{int(task.get('turns') or 0)} turns</span>"
            f"<span>{int(task.get('typed_assertions') or 0)} assertions</span>"
            f"<span>{int(task.get('scalar_views') or 0)} scalar views</span>{graph_badge}</div></div>"
            f"<div class='directory-scores'>{score_badge('memory', memory_score)}"
            f"{score_badge('no memory', baseline_score)}</div></article>"
        )
    total = len(tasks)
    return (
        "<section class='task-directory'>"
        "<div class='directory-top'><div><a class='back-link' href='/'>← run dashboard</a>"
        f"<h2>All tasks <span>{total}</span></h2>"
        "<p class='muted'>Open any task to inspect its evidence, assertions, Views, content memory, "
        "derivation, and answer path.</p></div>"
        "<div class='directory-summary'>"
        f"<span class='score-correct'>{status_counts['correct']} correct</span>"
        f"<span class='score-incorrect'>{status_counts['incorrect']} incorrect</span>"
        f"<span class='pending-score'>{status_counts['unscored']} unscored</span></div></div>"
        "<div class='directory-controls'>"
        "<label>find a task<input id='task-search' type='search' placeholder='question, ID, type, or answer' "
        "oninput='window.filterTaskDirectory()'></label>"
        "<label>memory score<select id='task-score-filter' onchange='window.filterTaskDirectory()'>"
        "<option value='all'>all</option><option value='correct'>correct</option>"
        "<option value='incorrect'>incorrect</option><option value='unscored'>unscored</option>"
        "</select></label>"
        f"<span id='task-visible-count'>{total} of {total} tasks</span></div>"
        f"<div id='task-list' class='task-directory-list'>{''.join(rows)}</div>"
        "<p id='task-directory-empty' class='empty' hidden>No tasks match this search.</p>"
        "</section>"
    )


def _scalar_viewer_shell(
    default_namespace: str | None,
    *,
    detail_page: bool = False,
) -> str:
    """Interactive shell; task data is loaded on demand from read-only JSON routes."""
    default_json = (
        json.dumps(default_namespace or "")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    detail_json = "true" if detail_page else "false"
    return """
<section id="scalar-viewer" class="viewer">
  <div class="viewer-head">
    <div>
      <div class="eyebrow">READ-ONLY GRAPH EXPLORER</div>
      <h2>How one scalar view is made</h2>
      <p class="muted viewer-copy">Walk one completed task from transcript evidence through the
      k-sample vote, durable assertions, deterministic fold, and benchmark answer.</p>
    </div>
    <div class="viewer-picker">
      <label for="scalar-task">completed task <span id="scalar-task-count"></span></label>
      <select id="scalar-task"></select>
      <button id="scalar-load" type="button">inspect task</button>
      <div class="task-links"><a href="/tasks/">all tasks</a>
      <a id="scalar-open" href="/tasks/">open task page ↗</a></div>
    </div>
  </div>
  <div id="scalar-status" class="muted">Loading completed tasks…</div>
  <div id="scalar-body"></div>
</section>
<script>
(() => {
  const preferred = __DEFAULT_NAMESPACE__;
  const detailPage = __DETAIL_PAGE__;
  const state = {data: null, stage: 0, onlyFounded: false, catalogFingerprint: ""};
  const stageMeta = [
    ["1", "Evidence", "preserved source turns"],
    ["2", "2-of-3 gate", "stochastic extraction vote"],
    ["3", "Assertions", "durable typed events"],
    ["4", "Scalar view", "deterministic fold"],
    ["5", "Memory map", "view, derivation, or content"],
    ["6", "Answer path", "recall and scoring"],
  ];
  const h = value => String(value == null ? "" : value)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  const when = value => value ? h(String(value).replace("T", " ").replace("Z", " UTC")) : "—";
  const list = value => Array.isArray(value) ? value : [];
  const foldLabels = {
    current: "CURRENT VIEW",
    historical: "HISTORICAL VIEW",
    recorded: "RECORDED",
    abstained: "ABSTAINED",
    expired: "EXPIRED",
    not_folded: "NOT FOLDED",
    not_materialized: "NO VIEW",
    superseded: "SUPERSEDED",
    write_failed: "WRITE FAILED",
  };
  const foldProblemStatuses = new Set([
    "abstained", "expired", "not_folded", "not_materialized", "write_failed",
  ]);
  const status = document.getElementById("scalar-status");
  const body = document.getElementById("scalar-body");
  const picker = document.getElementById("scalar-task");
  const taskCount = document.getElementById("scalar-task-count");
  const openTask = document.getElementById("scalar-open");

  function currentViews() {
    return list(state.data && state.data.views).filter(v => v.current);
  }

  function stageCounts() {
    const d = state.data || {};
    const gates = list(d.audit).filter(x => x.event === "gate");
    const scores = list(d.scoring);
    const assertions = list(d.assertions);
    const foldProblems = assertions.filter(a => {
      const outcome = a.fold_outcome || {};
      return [outcome.state, outcome.history].some(x => x && foldProblemStatuses.has(x.status));
    }).length;
    const histCount = list(d.history_views).filter(v => v.current).length;
    const viewLabel = currentViews().length + " state"
      + (histCount ? ", " + histCount + " history" : "");
    return [
      list(d.evidence).filter(x => list(x.founds).length).length + "/" + list(d.evidence).length,
      gates.length,
      assertions.length + (foldProblems ? " · " + foldProblems + " blocked" : ""),
      viewLabel,
      list(d.memory_inventory).filter(x => x.memory_type === "view").length + " views · "
        + list(d.memory_inventory).filter(x => x.memory_type === "content").length + " content",
      scores.length ? scores.length + " scored" : "pending",
    ];
  }

  function renderNav() {
    const counts = stageCounts();
    return `<div class="stage-nav">${stageMeta.map((s, i) =>
      `<button type="button" class="stage ${i === state.stage ? "active" : ""}"
        onclick="window.scalarStage(${i})"><span class="stage-num">${s[0]}</span>
        <span><strong>${s[1]}</strong><small>${h(counts[i])}</small></span></button>`
    ).join('<span class="stage-arrow">→</span>')}</div>`;
  }

  function renderEvidence(d) {
    const assertions = new Map(list(d.assertions).map(a => [a.id, a]));
    const rows = list(d.evidence).filter(t => !state.onlyFounded || list(t.founds).length);
    const transcript = rows.map((t, i) => {
      const founded = list(t.founds).map(id => assertions.get(id)).filter(Boolean);
      return `<article class="turn ${founded.length ? "founded" : ""}">
        <div class="turn-meta"><span>${i + 1}</span><b>${h(t.role)}</b>
          <span>${t.occurred_at ? "source " + when(t.occurred_at) : "source time unavailable"}</span>
          <span>ingested ${when(t.recorded_at)}</span></div>
        <div class="turn-text">${h(t.text)}</div>
        ${founded.length ? `<div class="founds">FOUNDS ${founded.map(a =>
          `<span>${h(a.attribute)} = ${h(a.value)}${a.unit ? " " + h(a.unit) : ""}</span>`
        ).join("")}</div>` : ""}
      </article>`;
    }).join("");
    const facts = list(d.facts).map(f =>
      `<li><b>${h(f.subject)}</b> → ${h(f.object)}<span>${h(f.fact)}</span></li>`
    ).join("");
    return `<div class="stage-panel"><div class="stage-intro">
      <div><h3>Source boundary</h3><p>Every user turn is preserved as <code>TurnEvidence</code>.
      Source time says when it happened; ingested time says when Menhir recorded it. A
      <code>FOUNDS</code> edge marks the exact turn that grounded a committed assertion. Older
      graphs may show source time as unavailable because that field was not captured yet.</p></div>
      <label class="toggle"><input type="checkbox" ${state.onlyFounded ? "checked" : ""}
        onchange="window.scalarFounded(this.checked)"> only founding turns</label></div>
      <div class="transcript">${transcript || '<p class="empty">No evidence rows in this graph.</p>'}</div>
      <details><summary>Context graph facts (${list(d.facts).length})</summary>
        <p class="muted">These aid ordinary recall, but they are not the typed assertion log folded
        into the scalar view.</p><ul class="fact-list">${facts}</ul></details></div>`;
  }

  function voteDistribution(details) {
    const dist = details.distribution || {};
    const k = Number(details.k || 0);
    return Object.entries(dist).map(([label, votes]) => {
      const n = Number(votes || 0);
      const pct = k ? (n / k * 100) : 0;
      return `<div class="vote-row"><div><span>${h(label)}</span><b>${n}/${k || "?"}</b></div>
        <div class="vote-bar"><i style="width:${pct}%"></i></div></div>`;
    }).join("");
  }

  function renderGate(d) {
    const gates = list(d.audit).filter(x => x.event === "gate");
    const cards = gates.map(g => {
      const x = g.details || {};
      const accepted = g.state === "commit";
      return `<article class="gate-card ${accepted ? "accepted" : "rejected"}">
        <div class="gate-title"><span class="pill">${accepted ? "ACCEPTED" : "ABSTAINED"}</span>
          <b>${Math.round(Number(x.agreement || 0) * 100)}% agreement</b>
          <span>${h(x.reason || g.state)}</span></div>
        ${voteDistribution(x)}
        <div class="mono-id">source ${h(String(x.source_key || "").slice(0, 16))}…</div>
      </article>`;
    }).join("");
    const warning = d.audit_warning ? `<div class="warning">${h(d.audit_warning)}</div>` : "";
    return `<div class="stage-panel"><div class="stage-intro"><div><h3>Probabilistic boundary</h3>
      <p>The extractor runs <em>k</em> times. Each sample gets one vote per grounded source claim.
      At a 2-of-3 threshold, two matching readings commit; scattered readings abstain.</p></div>
      <div class="receipt">${d.audit_pass_id ? "receipt " + h(d.audit_pass_id) : "no receipt"}</div>
      </div>${warning}${cards || '<p class="empty">No matching gate receipt.</p>'}</div>`;
  }

  function renderAssertions(d) {
    const evidence = list(d.evidence);
    const evidenceById = new Map(evidence.map(turn => [String(turn.id || ""), turn]));
    const sourceTurnFor = assertion => {
      const direct = evidenceById.get(String(assertion.evidence_id || ""));
      if (direct) return direct;
      return evidence.find(turn =>
        list(turn.founds).some(assertionId => String(assertionId) === String(assertion.id))) || null;
    };
    const badge = (projection, outcome) => {
      const value = outcome || {};
      const foldStatus = String(value.status || "unknown").replace(/[^a-z_]/g, "");
      const label = foldLabels[foldStatus] || "UNKNOWN";
      return `<span class="fold-badge ${foldStatus}" title="${h(value.reason || "")}">
        <b>${h(projection)}</b>${h(label)}</span>`;
    };
    const cards = list(d.assertions).map((a, i) => {
      const outcome = a.fold_outcome || {};
      const projections = [["state", outcome.state], ["history", outcome.history]];
      const problems = projections.filter(([, value]) =>
        value && foldProblemStatuses.has(value.status));
      const reasons = problems.map(([projection, value]) =>
        `<b>${h(projection)}</b>: ${h(String(value.reason || value.status).replaceAll("_", " "))}`
      ).join(" · ");
      const sourceTurn = sourceTurnFor(a);
      const sourceQuote = problems.length
        ? sourceTurn
          ? `<div class="assertion-source">
              <div class="assertion-source-meta"><b>ORIGINAL SOURCE TURN</b>
                <span>${h(sourceTurn.role)} · source ${when(sourceTurn.occurred_at)}</span></div>
              <blockquote>“${h(sourceTurn.text)}”</blockquote>
            </div>`
          : `<div class="assertion-source unavailable">Original source turn unavailable in this graph.</div>`
        : "";
      return `<article class="assertion ${problems.length ? "fold-blocked" : ""}">
        <div class="assertion-num">${i + 1}</div><div>
          <div class="assertion-value"><b>${h(a.subject)}</b><span>→</span>
            <b>${h(a.attribute)}</b><span>=</span><strong>${h(a.value)}${a.unit ? " " + h(a.unit) : ""}</strong></div>
          <div class="extracted-span"><span>EXTRACTED SPAN</span><blockquote>“${h(a.stated_span)}”</blockquote></div>
          <div class="chips"><span>${h(a.value_kind)}</span><span>${h(a.operation)}</span>
            <span>${h(a.evidence_tier)} evidence</span><span>${a.binding_pending ? "binding pending" : "subject bound"}</span></div>
          <div class="fold-outcomes">${projections.map(([projection, value]) =>
            badge(projection, value)).join("")}</div>
          ${reasons ? `<div class="fold-reasons">${reasons}</div>` : ""}
          ${sourceQuote}
          <div class="muted">world time ${when(a.valid_at)} · learned ${when(a.learned_at)}</div>
        </div>
      </article>`;
    }).join("");
    return `<div class="stage-panel"><div class="stage-intro"><div><h3>Durable event log</h3>
      <p>A winning interpretation becomes an immutable <code>TypedAssertion</code>. It records
      subject, slot, typed value, world time, source quote, and binding state. Projection badges
      show whether each assertion reached current state, advisory history, safely abstained, or
      could not be folded. Gate rejections remain in stage 2 because no assertion was created.</p></div></div>
      <div class="assertions">${cards || '<p class="empty">No TypedAssertions were emitted.</p>'}</div></div>`;
  }

  function scalarViewLaneKey(view) {
    return String(view.view_key || JSON.stringify([
      view.subject_uuid,
      view.attribute,
      view.scope,
      view.value_kind,
      view.unit,
    ]));
  }

  function renderViews(d) {
    const stateLanes = new Map();
    list(d.views).forEach(view => {
      const key = scalarViewLaneKey(view);
      if (!stateLanes.has(key)) stateLanes.set(key, []);
      stateLanes.get(key).push(view);
    });
    const cards = Array.from(stateLanes.values()).map(lane =>
      `<div class="views view-lane">${lane.map((v, i) =>
        `<article class="view-card ${v.current ? "current" : "history"}">
          <div class="view-state">${v.current ? "CURRENT" : "SUPERSEDED"}</div>
          <div class="view-value">${h(v.display || v.value)}${v.unit ? " " + h(v.unit) : ""}</div>
          <div><b>${h(v.subject)}</b> · ${h(v.attribute)}${v.scope ? " · " + h(v.scope) : ""}</div>
          <p>${h(v.summary)}</p>
          <div class="chips"><span>${h(v.value_kind)}</span><span>${h(v.effective_tier)} effective tier</span>
            <span>${list(v.contributor_ids).length} contributor</span></div>
          <div class="muted">valid ${when(v.valid_at)}</div>
        </article>${i < lane.length - 1 ? '<div class="supersede">→ later world time →</div>' : ""}`
      ).join("")}</div>`
    ).join("");
    // scalar_history Views: advisory, ordered delta/assertion history per slot.
    const hvs = list(d.history_views).filter(v => v.current);
    const hCards = hvs.map(hv => {
      const ops = hv.op_counts || {};
      const deltaOnly = Object.keys(ops).length > 0 && Object.keys(ops).every(k => k === "delta");
      const entries = list(hv.entries).map(e =>
        `<tr><td>${when(e.valid_at)}</td><td>${h(e.operation)}</td>
         <td>${h(e.value)}</td><td>${h(e.stated_span)}</td></tr>`
      ).join("");
      return `<article class="view-card advisory">
        <div class="view-state">ADVISORY HISTORY</div>
        <div><b>${h(hv.subject)}</b> · ${h(hv.attribute)}${hv.scope ? " · " + h(hv.scope) : ""}</div>
        <div class="chips"><span>${hv.entry_count || 0} entries</span>
          <span>${Object.entries(ops).map(([k,v]) => v + " " + k).join(", ")}</span></div>
        ${deltaOnly ? '<div class="warning">advisory scalar history — not an absolute current total</div>' : ""}
        ${entries ? `<table class="history-entries"><thead><tr>
          <th>source time</th><th>operation</th><th>value</th><th>stated span</th>
        </tr></thead><tbody>${entries}</tbody></table>` : ""}
        <div class="muted">${when(hv.first_valid_at)} to ${when(hv.last_valid_at)}</div>
      </article>`;
    }).join("");
    const stateEmpty = !cards;
    const histEmpty = !hCards;
    const noAbstain = stateEmpty && !histEmpty
      ? '<p class="muted">scalar_state: abstained (no anchor). Advisory history is available below.</p>'
      : "";
    return `<div class="stage-panel"><div class="stage-intro"><div><h3>Deterministic projection</h3>
      <p>The fold replays assertions by slot and world time. Views are rebuildable projections:
      old values remain visible for provenance, while exactly one value is current.</p></div></div>
      <div class="view-lanes">${cards || '<p class="empty">No scalar_state view materialized.</p>'}</div>
      ${noAbstain}
      ${hCards ? '<h4>Advisory history</h4><div class="views">' + hCards + '</div>' : ""}</div>`;
  }

  function renderMemoryMap(d) {
    const inventory = list(d.memory_inventory);
    const views = inventory.filter(item => item.memory_type === "view");
    const content = inventory.filter(item => item.memory_type === "content");
    const viewCards = views.map(item => {
      const derivation = item.derivation === "delta"
        ? "DELTA-DERIVED"
        : item.derivation === "absolute"
          ? "ABSOLUTE"
          : item.derivation === "mixed"
            ? "MIXED DERIVATION"
            : "DERIVATION UNKNOWN";
      const kind = item.view_kind === "scalar_history" ? "HISTORY VIEW" : "STATE VIEW";
      return `<article class="memory-card view-memory">
        <div class="memory-labels"><span class="memory-kind view">VIEW</span>
          <span>${h(kind)}</span><span class="derivation ${h(item.derivation)}">${h(derivation)}</span></div>
        <h4>${h(item.subject)} · ${h(item.attribute)}${item.scope ? " · " + h(item.scope) : ""}</h4>
        ${item.value ? `<div class="memory-value">${h(item.value)}</div>` : ""}
        <p>${h(item.content)}</p>
        <div class="muted">${item.current ? "current rebuildable projection" : "superseded projection"}
          · operations ${list(item.operations).map(h).join(", ") || "unavailable"}</div>
      </article>`;
    }).join("");
    const contentCards = content.map(item =>
      `<article class="memory-card content-memory">
        <div class="memory-labels"><span class="memory-kind content">CONTENT</span>
          <span>ordinary graph fact</span></div>
        <h4>${h(item.subject)} → ${h(item.object)}</h4>
        <p>${h(item.content)}</p>
        <div class="muted">${h(item.relation)} · ${list(item.episode_ids).length} source episode(s)</div>
      </article>`
    ).join("");
    return `<div class="stage-panel"><div class="stage-intro"><div><h3>Task memory inventory</h3>
      <p><b>VIEW</b> means a rebuildable scalar projection derived from durable assertions.
      Its derivation badge says whether the contributing assertions were absolute observations,
      deltas, or a mix. <b>CONTENT</b> means an ordinary relationship fact extracted from the
      conversation; it is not part of the scalar fold.</p></div></div>
      <div class="memory-summary"><span><b>${views.length}</b> Views</span>
        <span><b>${content.length}</b> content facts</span></div>
      <h4>Derived Views</h4><div class="memory-grid">${viewCards ||
        '<p class="empty">No scalar Views materialized for this task.</p>'}</div>
      <h4>Content memory</h4><div class="memory-grid">${contentCards ||
        '<p class="empty">No ordinary relationship facts are available.</p>'}</div></div>`;
  }

  function renderAnswer(d) {
    const task = d.task || {};
    const views = currentViews();
    const scores = list(d.scoring);
    const viewRows = views.map(v =>
      `<div class="answer-source"><span>current scalar view</span><b>${h(v.attribute)} = ${h(v.display || v.value)}${v.unit ? " " + h(v.unit) : ""}</b></div>`
    ).join("");
    // Advisory history as fallback state when scalar_state abstains
    const hvs = list(d.history_views).filter(v => v.current);
    const histRows = hvs.map(hv => {
      const latest = list(hv.entries).slice(-1)[0];
      const latestVal = latest ? h(latest.value) + " (" + h(latest.operation) + ", not absolute)" : "—";
      return `<div class="answer-source advisory"><span>advisory history (latest delta)</span><b>${h(hv.attribute)} = ${latestVal}</b></div>`;
    }).join("");
    const stateBlock = viewRows || (histRows
      ? '<p class="muted">scalar_state: abstained (no anchor)</p>' + histRows
      : '<p class="empty">No current scalar view.</p>');
    const scoring = scores.length ? scores.map(s =>
      `<article class="score ${s.correct ? "okbox" : "nobox"}"><b>${h(s.arm)}</b>
       <span>${s.correct ? "correct" : "incorrect"}</span><p>${h(s.response)}</p>
       <details><summary>recalled memory</summary><pre>${h(s.recalled)}</pre></details></article>`
    ).join("") : `<div class="pending">Recall/QA has not written a checkpoint for this task yet.
      The graph build can be inspected now; the final answer path becomes measurable during scoring.</div>`;
    return `<div class="stage-panel answer-grid">
      <div class="answer-question"><span>BENCHMARK QUESTION</span><h3>${h(task.question)}</h3>
        <div class="gold"><span>gold answer</span><b>${h(task.answer)}</b></div></div>
      <div><h3>Available state</h3>${stateBlock}</div>
      <div class="score-wrap"><h3>Recall → model answer → scorer</h3>${scoring}</div></div>`;
  }

  function render() {
    const d = state.data;
    if (!d) return;
    const task = d.task || {};
    const panels = [renderEvidence, renderGate, renderAssertions, renderViews, renderMemoryMap, renderAnswer];
    body.innerHTML = `<div class="task-hero"><div><span>${h(task.question_type || "task")}</span>
      <h3>${h(task.question || d.namespace)}</h3></div><div class="task-id">${h(d.namespace)}</div></div>
      ${renderNav()}${panels[state.stage](d)}
      <div class="stage-controls"><button type="button" onclick="window.scalarStage(${Math.max(0, state.stage - 1)})"
        ${state.stage === 0 ? "disabled" : ""}>← previous</button>
      <span>${state.stage + 1} / ${stageMeta.length}</span>
      <button type="button" onclick="window.scalarStage(${Math.min(stageMeta.length - 1, state.stage + 1)})"
        ${state.stage === stageMeta.length - 1 ? "disabled" : ""}>next →</button></div>`;
  }

  window.scalarStage = i => { state.stage = Number(i); render(); body.scrollIntoView({behavior: "smooth", block: "start"}); };
  window.scalarFounded = checked => { state.onlyFounded = Boolean(checked); render(); };

  async function loadTask() {
    const namespace = picker.value;
    if (!namespace) return;
    status.textContent = "Reading " + namespace + "…";
    body.innerHTML = "";
    try {
      const response = await fetch("/api/scalar-task?namespace=" + encodeURIComponent(namespace), {cache: "no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "viewer request failed");
      const taskPath = "/tasks/" + encodeURIComponent(namespace);
      openTask.href = taskPath;
      if (detailPage) history.replaceState(null, "", taskPath);
      state.data = data;
      state.stage = 0;
      status.textContent = data.graph_warning || (list(data.evidence).length
        ? "Loaded from Neo4j" + (data.audit_pass_id ? " + vote receipt" : "")
        : "Manifest task loaded; scalar graph evidence is not available yet.");
      render();
      if (detailPage && !location.hash) {
        document.getElementById("scalar-viewer").scrollIntoView({block: "start"});
      }
    } catch (error) {
      status.textContent = "Could not load task: " + error.message;
    }
  }

  async function refreshTaskCatalog(initial = false) {
    try {
      const response = await fetch("/api/scalar-tasks", {cache: "no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "task catalog failed");
      const tasks = list(data.tasks);
      const selected = picker.value;
      const fingerprint = JSON.stringify(tasks.map(task => [
        task.namespace,
        task.question,
        task.graph_available,
      ]));
      if (fingerprint !== state.catalogFingerprint) {
        picker.replaceChildren(...tasks.map(task => {
          const option = document.createElement("option");
          option.value = task.namespace;
          option.textContent = task.question_id + " · " + (task.question || task.namespace)
            + (task.graph_available === false ? " · graph unavailable" : "");
          return option;
        }));
        state.catalogFingerprint = fingerprint;
      }
      taskCount.textContent = "(" + tasks.length + ")";
      const wanted = initial ? (preferred || data.default_namespace) : selected;
      if (wanted && data.tasks.some(task => task.namespace === wanted)) picker.value = wanted;
      if (initial) {
        if (picker.value) await loadTask();
        else status.textContent = "No completed manifest tasks are available.";
      } else if (!state.data && picker.value) {
        await loadTask();
      }
    } catch (error) {
      if (initial || !state.data) {
        status.textContent = "Scalar viewer unavailable: " + error.message;
      }
    }
  }

  async function init() {
    document.getElementById("scalar-load").addEventListener("click", loadTask);
    picker.addEventListener("change", loadTask);
    window.addEventListener("dashboard:refresh", () => refreshTaskCatalog(false));
    await refreshTaskCatalog(true);
  }
  init();
})();
</script>
""".replace("__DEFAULT_NAMESPACE__", default_json).replace("__DETAIL_PAGE__", detail_json)


def render_html(
    snaps: list[RunSnapshot],
    menhir: dict | None,
    *,
    total_items: int | None,
    refresh_s: int = 5,
    items_n: int = 20,
    ingests: list[IngestSnapshot] | None = None,
    scalar_viewer_enabled: bool = False,
    scalar_default_namespace: str | None = None,
    scalar_detail_page: bool = False,
    task_directory: list[dict] | None = None,
    episode_progress: dict | None = None,
) -> str:
    """Self-contained auto-refreshing HTML page for the same data as render()."""
    rows: list[str] = []
    for s in snaps:
        per_arm_total = total_items or _KNOWN_VARIANT_ITEMS.get(s.variant)
        n_arms = max(1, len(s.arms))
        grand_total = per_arm_total * n_arms if per_arm_total else None
        pct = (s.total_done / grand_total * 100.0) if grand_total else None
        bar = (
            f'<div class="bar"><div class="fill" style="width:{min(100.0, pct):.1f}%"></div></div>'
            f'<span class="muted">{s.total_done}{"/" + str(grand_total) if grand_total else ""}'
            f'{f" ({pct:.1f}%)" if pct is not None else ""}</span>'
        ) if True else ""
        arm_rows = ""
        for arm in sorted(s.arms):
            a = s.arms[arm]
            done = f"{a.n}/{per_arm_total}" if per_arm_total else str(a.n)
            arm_rows += (
                f"<tr><td>{_esc(arm)}</td><td class='num'>{_esc(done)}</td>"
                f"<td class='num'>{a.score:.3f}</td><td class='num'>{a.input_tokens:,}</td>"
                f"<td class='num'>{a.output_tokens:,}</td></tr>"
            )
        lift = f"<span class='lift'>memory lift: {s.lift:+.3f}</span>" if s.lift is not None else ""
        src = f" <span class='src'>[{_esc(s.source)}]</span>" if s.source and s.source != "results" else ""
        feed = _feed_rows_html(s, items_n)
        feed_wrap_id = _slug_id(s.model, s.variant, s.source, "feedwrap")
        feed_block = (
            f"<details open id='{feed_wrap_id}'><summary class='muted'>turn-by-turn (latest {items_n})</summary>{feed}</details>"
            if feed else ""
        )
        rows.append(
            f"<div class='run'><h2>{_esc(s.benchmark)}{src} "
            f"<span class='muted'>variant={_esc(s.variant)} &middot; answer-model={_esc(s.model)}</span></h2>"
            f"<div class='prog'>{bar}</div>"
            f"<table><thead><tr><th>arm</th><th>done</th><th>acc</th><th>in_tok</th><th>out_tok</th></tr></thead>"
            f"<tbody>{arm_rows}</tbody></table>{lift}{feed_block}</div>"
        )
    ingest_rows = _ingest_rows_html(ingests or [], total_items)
    body = _episode_progress_html(episode_progress) + ingest_rows + "".join(rows)
    if task_directory is not None:
        body = _task_directory_html(task_directory)
    if not body:
        body = "<p class='muted'>No ingest manifest or scoring checkpoints yet.</p>"

    if menhir is None:
        mh = "<span class='muted'>menhir: not probed</span>"
    elif menhir.get("health"):
        enr = menhir.get("enrichment") or {}
        rate = enr.get("episodes_per_min") or enr.get("rate") or enr.get("per_min")
        mh = (f"<span class='up'>&#9679; menhir UP</span> "
              f"<span class='muted'>mode={_esc(menhir.get('startup_mode'))} &middot; "
              f"queue_depth={_esc(menhir.get('queue_depth'))}"
              f"{f' &middot; enrich~{_esc(rate)}/min' if rate else ''}</span>")
    else:
        mh = "<span class='down'>&#9679; menhir starting/degraded</span>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>archolith-bench dashboard</title>
<style>
 body{{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:24px}}
 h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:15px;margin:18px 0 8px}}
 .muted{{color:#8b949e;font-weight:normal}} .up{{color:#3fb950}} .down{{color:#f85149}}
 .src{{color:#d29922}}
 .lift{{color:#58a6ff}} table{{border-collapse:collapse;margin:6px 0}}
 th,td{{padding:4px 14px 4px 0;text-align:left}} th{{color:#8b949e;font-weight:normal;border-bottom:1px solid #30363d}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 .bar{{display:inline-block;width:320px;height:12px;background:#21262d;border-radius:6px;overflow:hidden;vertical-align:middle;margin-right:8px}}
 .fill{{height:100%;background:linear-gradient(90deg,#1f6feb,#3fb950)}}
 .run{{border:1px solid #30363d;border-radius:8px;padding:10px 16px;margin:12px 0;max-width:900px}}
 .ok{{color:#3fb950;font-weight:bold}} .no{{color:#f85149;font-weight:bold}}
 details{{margin-top:8px}} summary{{cursor:pointer}}
 table.feed{{width:100%}}
 table.feed td{{vertical-align:top}}
 table.feed td.ans{{white-space:normal;overflow-wrap:anywhere;color:#c9d1d9;max-width:0;width:99%}}
 table.feed td:nth-child(1),table.feed td:nth-child(2),table.feed td:nth-child(3),table.feed td:nth-child(4){{white-space:nowrap}}
 td.ans details summary{{cursor:pointer;color:#c9d1d9}}
 a.task-link{{color:#58a6ff;text-decoration:none;font-weight:600}}
 a.task-link:hover,a.task-link:focus{{text-decoration:underline}}
 td.ans details[open] summary{{color:#58a6ff;margin-bottom:6px}}
 .blk{{margin:5px 0 5px 10px;border-left:2px solid #30363d;padding-left:10px}}
 .lbl{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b949e}}
 .blk pre{{margin:2px 0;white-space:pre-wrap;overflow-wrap:anywhere;color:#79c0ff;font-size:12.5px;max-height:220px;overflow:auto}}
 .snips{{max-height:260px;overflow:auto;margin-top:3px}}
 .snip{{padding:4px 8px;margin:3px 0;border-left:3px solid #2f5a8f;background:#0f1620;border-radius:3px;
   font-size:12.5px;color:#adbac7;white-space:pre-wrap;overflow-wrap:anywhere}}
 .snip:nth-child(even){{background:#131c28}}
 .snum{{display:inline-block;min-width:22px;color:#d29922;font-weight:bold}}
 .viewer{{border:1px solid #30363d;border-radius:12px;padding:18px;margin:28px 0 12px;max-width:1060px;
   background:linear-gradient(145deg,#101722,#0d1117 48%);box-shadow:0 12px 40px #0005}}
 .viewer-head{{display:flex;justify-content:space-between;gap:28px;align-items:end}}
 .viewer-head h2{{font-size:20px;margin:2px 0 4px}} .viewer-copy{{max-width:650px;margin:0}}
 .eyebrow{{color:#58a6ff;font-size:11px;letter-spacing:.12em;font-weight:bold}}
 .viewer-picker{{display:grid;grid-template-columns:auto auto;gap:6px;min-width:390px}}
 .viewer-picker label{{grid-column:1/-1;color:#8b949e;font-size:11px;text-transform:uppercase}}
 select,button{{font:inherit;color:#c9d1d9;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:7px 9px}}
 button{{cursor:pointer}} button:hover:not(:disabled){{border-color:#58a6ff;background:#1b2636}}
 button:disabled{{cursor:default;opacity:.4}} .viewer-picker select{{min-width:280px;max-width:420px}}
 .task-links{{grid-column:1/-1;display:flex;justify-content:space-between;gap:12px;font-size:11px}}
 .task-links a{{color:#58a6ff;text-decoration:none}} .task-links a:hover{{text-decoration:underline}}
 .task-directory{{max-width:1100px;margin-top:18px}} .directory-top{{display:flex;justify-content:space-between;
   gap:20px;align-items:end;border-bottom:1px solid #30363d;padding-bottom:14px}}
 .directory-top h2{{font-size:24px;margin:5px 0 2px}} .directory-top h2 span{{color:#8b949e;font-size:15px}}
 .back-link,.directory-id a,.directory-question a{{color:#58a6ff;text-decoration:none}}
 .back-link:hover,.directory-id a:hover,.directory-question a:hover{{text-decoration:underline}}
 .directory-summary{{display:flex;gap:8px;flex-wrap:wrap}} .directory-summary span,.score-pill{{border:1px solid #30363d;
   border-radius:12px;padding:3px 8px;font-size:10px;white-space:nowrap}}
 .score-correct{{color:#3fb950}} .score-incorrect{{color:#f85149}} .pending-score{{color:#8b949e}}
 .directory-controls{{display:flex;gap:12px;align-items:end;margin:16px 0}}
 .directory-controls label{{display:grid;gap:4px;color:#8b949e;font-size:10px;text-transform:uppercase}}
 .directory-controls input{{font:inherit;color:#c9d1d9;background:#161b22;border:1px solid #30363d;
   border-radius:6px;padding:8px 10px;min-width:360px}} .directory-controls>span{{margin-left:auto;color:#8b949e}}
 .task-directory-list{{display:grid;gap:8px}} .task-directory-row{{display:grid;grid-template-columns:130px 1fr auto;
   gap:14px;align-items:center;border:1px solid #30363d;border-radius:8px;padding:12px 14px;background:#0d1117}}
 .task-directory-row:hover{{border-color:#484f58;background:#101722}} .directory-id a{{font-weight:bold}}
 .directory-id span,.directory-question>span{{display:block;color:#8b949e;font-size:9px;text-transform:uppercase;
   letter-spacing:.07em;margin-top:3px}} .directory-question h3{{font:600 14px/1.45 ui-sans-serif,system-ui;margin:3px 0}}
 .directory-question h3 a{{color:#e6edf3}} .directory-counts{{display:flex;gap:6px;flex-wrap:wrap}}
 .directory-counts span,.directory-chip{{color:#8b949e;border:1px solid #30363d;border-radius:10px;padding:1px 6px;font-size:9px}}
 .directory-chip.graph-ready{{color:#3fb950;border-color:#2ea04366}} .directory-chip.graph-missing{{color:#e3b341;border-color:#d2992266}}
 .directory-scores{{display:grid;gap:5px;justify-items:end}}
 #scalar-status{{margin:14px 0 8px}} .task-hero{{display:flex;justify-content:space-between;gap:20px;
   align-items:start;padding:14px;border:1px solid #30363d;border-radius:8px;background:#0d1117}}
 .task-hero span,.answer-question>span,.gold span,.answer-source span{{display:block;color:#8b949e;
   font-size:10px;letter-spacing:.09em;text-transform:uppercase}}
 .task-hero h3{{font:600 17px/1.4 ui-sans-serif,system-ui;margin:4px 0}} .task-id{{color:#d29922;white-space:nowrap}}
 .stage-nav{{display:flex;align-items:stretch;margin:16px 0;overflow-x:auto}}
 .stage{{display:flex;text-align:left;gap:8px;align-items:center;flex:1;min-width:118px;padding:10px;border-radius:8px}}
 .stage.active{{border-color:#58a6ff;background:#12233a;box-shadow:inset 0 0 0 1px #58a6ff44}}
 .stage-num{{display:grid;place-items:center;width:24px;height:24px;border-radius:50%;background:#21262d;color:#58a6ff}}
 .stage strong,.stage small{{display:block}} .stage small{{color:#8b949e;margin-top:2px}} .stage-arrow{{padding:15px 4px;color:#484f58}}
 .stage-panel{{min-height:300px;border-top:1px solid #30363d;padding-top:14px}}
 .stage-intro{{display:flex;justify-content:space-between;gap:20px;align-items:start;margin-bottom:12px}}
 .stage-intro h3,.answer-grid h3{{margin:0 0 4px;font-size:15px}} .stage-intro p{{margin:0;max-width:720px;color:#8b949e}}
 code{{color:#79c0ff}} .toggle{{white-space:nowrap;color:#8b949e}} .toggle input{{accent-color:#1f6feb}}
 .transcript{{display:grid;gap:7px;max-height:520px;overflow:auto;padding-right:5px}}
 .turn{{border:1px solid #21262d;border-left:3px solid #30363d;border-radius:6px;padding:8px 10px;background:#0d1117}}
 .turn.founded{{border-left-color:#3fb950;background:#0e1b15}} .turn-meta{{display:flex;gap:10px;color:#8b949e;font-size:11px}}
 .turn-meta b{{color:#d2a8ff;text-transform:uppercase}} .turn-text{{font:13px/1.5 ui-sans-serif,system-ui;margin-top:3px}}
 .founds{{margin-top:7px;color:#3fb950;font-size:10px;letter-spacing:.06em}} .founds span{{margin-left:8px;
   padding:2px 6px;border:1px solid #2ea04366;border-radius:10px;letter-spacing:0}}
 .fact-list{{max-height:260px;overflow:auto;padding-left:20px}} .fact-list li{{margin:5px 0}}
 .fact-list span{{display:block;color:#8b949e}} .receipt,.mono-id{{color:#8b949e;font-size:11px}}
 .gate-card{{border:1px solid #30363d;border-left:4px solid #f85149;border-radius:8px;padding:12px;margin:8px 0;background:#0d1117}}
 .gate-card.accepted{{border-left-color:#3fb950}} .gate-title{{display:flex;gap:10px;align-items:center;margin-bottom:9px}}
 .gate-title>span:last-child{{color:#8b949e}} .pill{{font-size:9px;letter-spacing:.08em;padding:2px 6px;border-radius:10px;background:#21262d}}
 .accepted .pill{{color:#3fb950}} .rejected .pill{{color:#f85149}} .vote-row{{max-width:760px;margin:7px 0}}
 .vote-row>div:first-child{{display:flex;justify-content:space-between;gap:12px}} .vote-bar{{height:5px;background:#21262d;border-radius:3px;overflow:hidden}}
 .vote-bar i{{display:block;height:100%;background:#58a6ff}} .warning,.pending{{border:1px solid #d2992266;background:#2d220d;
   color:#e3b341;border-radius:6px;padding:10px}} .assertions{{display:grid;gap:9px}}
  .assertion{{display:grid;grid-template-columns:34px 1fr;gap:8px;border:1px solid #30363d;border-radius:8px;padding:12px;background:#0d1117}}
  .assertion.fold-blocked{{border-left:4px solid #d29922}}
  .assertion-num{{display:grid;place-items:center;width:26px;height:26px;border-radius:50%;background:#1f6feb;color:white}}
  .assertion-value{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .assertion-value strong{{color:#79c0ff;font-size:16px}}
  blockquote{{margin:7px 0;color:#e6edf3;font:italic 14px/1.5 ui-serif,Georgia}}
  .extracted-span>span,.assertion-source-meta b{{font-size:9px;letter-spacing:.08em;color:#8b949e}}
  .assertion-source{{margin:9px 0;padding:9px 11px;border:1px solid #d2992266;border-radius:6px;background:#2d220d}}
  .assertion-source.unavailable{{color:#8b949e;font-size:11px}}
  .assertion-source-meta{{display:flex;gap:10px;justify-content:space-between;color:#8b949e;font-size:10px}}
  .assertion-source blockquote{{margin-bottom:0;color:#fff8c5}}
  .chips{{display:flex;gap:5px;flex-wrap:wrap;margin:7px 0}}
  .chips span{{font-size:10px;padding:2px 6px;border:1px solid #30363d;border-radius:10px;color:#8b949e}}
  .fold-outcomes{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 5px}}
  .fold-badge{{display:inline-flex;gap:6px;border:1px solid #30363d;border-radius:5px;padding:3px 7px;
    color:#8b949e;font-size:10px}} .fold-badge b{{color:#e6edf3;text-transform:uppercase}}
  .fold-badge.current,.fold-badge.recorded{{color:#3fb950;border-color:#2ea04366;background:#0e1b15}}
  .fold-badge.abstained,.fold-badge.expired,.fold-badge.not_folded,.fold-badge.not_materialized,
  .fold-badge.write_failed{{color:#e3b341;border-color:#d2992266;background:#2d220d}}
  .fold-reasons{{color:#e3b341;font-size:11px;margin-bottom:7px}}
  .view-lanes{{display:grid;gap:10px}} .views{{display:flex;align-items:stretch;overflow-x:auto;padding-bottom:6px}}
  .view-card{{min-width:240px;flex:1;border:1px solid #30363d;
   border-top:4px solid #6e7681;border-radius:8px;padding:14px;background:#0d1117}} .view-card.current{{border-top-color:#3fb950;background:#0e1b15}}
 .view-state{{color:#8b949e;font-size:10px;letter-spacing:.1em}} .current .view-state{{color:#3fb950}}
 .view-value{{font-size:34px;color:#79c0ff;font-weight:bold;margin:4px 0}} .view-card p{{color:#8b949e}}
 .supersede{{display:grid;place-items:center;min-width:135px;color:#d29922;text-align:center}}
 .memory-summary{{display:flex;gap:10px;margin:12px 0}} .memory-summary span{{border:1px solid #30363d;
   border-radius:7px;padding:7px 10px;color:#8b949e}} .memory-summary b{{color:#e6edf3}}
 .memory-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:9px;margin-bottom:18px}}
 .memory-card{{border:1px solid #30363d;border-radius:8px;padding:12px;background:#0d1117}}
 .memory-card.view-memory{{border-left:4px solid #58a6ff}} .memory-card.content-memory{{border-left:4px solid #8b949e}}
 .memory-card h4{{margin:8px 0 4px}} .memory-card p{{color:#adbac7;margin:5px 0 9px}}
 .memory-labels{{display:flex;gap:6px;align-items:center;flex-wrap:wrap;color:#8b949e;font-size:9px;letter-spacing:.07em}}
 .memory-labels span{{border:1px solid #30363d;border-radius:10px;padding:2px 6px}}
 .memory-kind.view{{color:#79c0ff;border-color:#1f6feb88}} .memory-kind.content{{color:#c9d1d9}}
 .derivation.absolute{{color:#3fb950;border-color:#2ea04366}} .derivation.delta{{color:#d2a8ff;border-color:#8957e566}}
 .derivation.mixed{{color:#e3b341;border-color:#d2992266}} .memory-value{{font-size:20px;color:#79c0ff;font-weight:bold}}
 .answer-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .answer-question{{border:1px solid #30363d;border-radius:8px;padding:16px;background:#0d1117}}
 .answer-question h3{{font:600 19px/1.5 ui-sans-serif,system-ui;margin:6px 0 16px}} .gold{{border-top:1px solid #30363d;padding-top:10px}}
 .gold b{{font-size:22px;color:#3fb950}} .answer-source{{border:1px solid #2ea04366;border-radius:7px;padding:10px;margin:6px 0;background:#0e1b15}}
 .answer-source b{{display:block;color:#79c0ff;margin-top:3px}} .score-wrap{{grid-column:1/-1}} .score{{border:1px solid #30363d;border-radius:7px;padding:10px}}
 .score.okbox{{border-left:4px solid #3fb950}} .score.nobox{{border-left:4px solid #f85149}} .score pre{{white-space:pre-wrap}}
 .empty{{color:#8b949e;font-style:italic}} .stage-controls{{display:flex;justify-content:space-between;align-items:center;margin-top:12px}}
 @media (max-width:800px){{.viewer-head,.stage-intro,.directory-top{{display:block}}.viewer-picker{{margin-top:12px;min-width:0}}
   .stage-arrow{{display:none}}.answer-grid{{grid-template-columns:1fr}}.viewer-picker select{{min-width:0}}
   .directory-controls{{align-items:stretch;flex-direction:column}}.directory-controls input{{min-width:0;width:100%;box-sizing:border-box}}
   .directory-controls>span{{margin-left:0}}.task-directory-row{{grid-template-columns:1fr}}.directory-scores{{display:flex;justify-items:start}}}}
 footer{{color:#8b949e;margin-top:18px;max-width:900px}}
</style></head><body>
<h1>archolith-bench &mdash; memory benchmark</h1>
<div id="content">
<div class="muted">{time.strftime('%Y-%m-%d %H:%M:%S')} &middot; live (every {refresh_s}s, in place)</div>
<div style="margin:10px 0">{mh}</div>
{body}
</div>
{_scalar_viewer_shell(scalar_default_namespace, detail_page=scalar_detail_page) if scalar_viewer_enabled else ""}
<footer>Token columns are ANSWER-model only; menhir ingestion (OpenAI extraction+embedding)
spend is not tracked by the bench. Progress is by item-count across arms.</footer>
<script>
window.filterTaskDirectory = function() {{
  const search = document.getElementById('task-search');
  const filter = document.getElementById('task-score-filter');
  const list = document.getElementById('task-list');
  if (!search || !filter || !list) return;
  const query = search.value.trim().toLowerCase();
  const score = filter.value;
  const rows = Array.from(list.querySelectorAll('[data-task-row]'));
  let visible = 0;
  rows.forEach(row => {{
    const show = (!query || row.dataset.search.includes(query))
      && (score === 'all' || row.dataset.score === score);
    row.hidden = !show;
    if (show) visible += 1;
  }});
  document.getElementById('task-visible-count').textContent = visible + ' of ' + rows.length + ' tasks';
  document.getElementById('task-directory-empty').hidden = visible !== 0;
}};
// In-place refresh: fetch the page, swap only #content, and preserve which <details>
// are open plus the scroll position -- so expanding a question is NOT reset every tick.
const RS = {refresh_s} * 1000;
async function tick() {{
  try {{
    const html = await (await fetch(location.href, {{cache: 'no-store'}})).text();
    const fresh = new DOMParser().parseFromString(html, 'text/html').getElementById('content');
    if (!fresh) return;
    // preserve: which questions are open, page scroll, and each retrieved-memory box's scroll
    const openIds = new Set(
      Array.from(document.querySelectorAll('details[open]')).map(d => d.id).filter(Boolean)
    );
    const innerScroll = {{}};
    document.querySelectorAll('.keepscroll[id]').forEach(el => {{
      if (el.scrollTop > 0) innerScroll[el.id] = el.scrollTop;
    }});
    const taskSearch = document.getElementById('task-search')?.value || '';
    const taskScore = document.getElementById('task-score-filter')?.value || 'all';
    const sy = window.scrollY;
    document.getElementById('content').replaceWith(fresh);
    fresh.querySelectorAll('details[id]').forEach(d => {{ d.open = openIds.has(d.id); }});
    fresh.querySelectorAll('.keepscroll[id]').forEach(el => {{
      if (innerScroll[el.id] != null) el.scrollTop = innerScroll[el.id];
    }});
    const freshTaskSearch = document.getElementById('task-search');
    const freshTaskScore = document.getElementById('task-score-filter');
    if (freshTaskSearch) freshTaskSearch.value = taskSearch;
    if (freshTaskScore) freshTaskScore.value = taskScore;
    window.filterTaskDirectory();
    window.scrollTo(0, sy);
    window.dispatchEvent(new Event('dashboard:refresh'));
  }} catch (e) {{ /* transient fetch error; try again next tick */ }}
}}
setInterval(tick, RS);
</script>
</body></html>"""


def serve_dashboard(
    results_dir: Path,
    *,
    menhir_url: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8200,
    total_items: int | None = None,
    refresh_s: int = 5,
    items_n: int = 20,
    active_within_s: float | None = None,
    scalar_reader: object | None = None,
    scalar_default_namespace: str | None = None,
) -> None:
    """Serve the dashboard as an auto-refreshing web page until interrupted."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlparse

    from .scalar_viewer import catalog_with_graph_availability, scoring_rows, task_catalog

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence access logs
            pass

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, payload: dict) -> None:
            self._send(
                status,
                json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self):  # noqa: N802
            route = urlparse(self.path)
            snaps = scan_runs(results_dir, active_within_s=active_within_s)
            ingests = scan_ingests(results_dir, active_within_s=active_within_s)
            catalog = task_catalog(ingests)
            if route.path.rstrip("/") == "/tasks":
                available = None
                if scalar_reader is not None:
                    try:
                        available = scalar_reader.available_namespaces()
                    except Exception:  # noqa: BLE001 - directory remains useful while graph is offline
                        pass
                directory = [
                    {
                        **task,
                        "scoring": scoring_rows(snaps, task["question_id"]),
                    }
                    for task in catalog_with_graph_availability(catalog, available)
                ]
                menhir = probe_menhir(menhir_url) if menhir_url else None
                page = render_html(
                    snaps,
                    menhir,
                    total_items=total_items,
                    refresh_s=refresh_s,
                    items_n=items_n,
                    ingests=ingests,
                    task_directory=directory,
                )
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route.path.startswith("/tasks/"):
                namespace = unquote(route.path.removeprefix("/tasks/")).strip("/")
                task = next((item for item in catalog if item["namespace"] == namespace), None)
                if task is None:
                    self._json(404, {"error": "Unknown or incomplete manifest namespace."})
                    return
                menhir = probe_menhir(menhir_url) if menhir_url else None
                page = render_html(
                    snaps,
                    menhir,
                    total_items=total_items,
                    refresh_s=refresh_s,
                    items_n=items_n,
                    ingests=ingests,
                    scalar_viewer_enabled=True,
                    scalar_default_namespace=namespace,
                    scalar_detail_page=True,
                )
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route.path == "/api/scalar-tasks":
                graph_warning = None
                available = None
                if scalar_reader is not None:
                    try:
                        available = scalar_reader.available_namespaces()
                    except Exception as exc:  # noqa: BLE001 - catalog remains useful if graph is down
                        graph_warning = f"Could not annotate tasks against the scalar graph: {exc}"
                catalog = catalog_with_graph_availability(catalog, available)
                default = scalar_default_namespace
                namespaces = {item["namespace"] for item in catalog}
                if default not in namespaces:
                    default = catalog[0]["namespace"] if catalog else None
                self._json(200, {
                    "tasks": catalog,
                    "default_namespace": default,
                    "reader_enabled": scalar_reader is not None,
                    "graph_warning": graph_warning,
                })
                return
            if route.path == "/api/scalar-task":
                namespace = (parse_qs(route.query).get("namespace") or [""])[0]
                task = next((item for item in catalog if item["namespace"] == namespace), None)
                if task is None:
                    self._json(404, {"error": "Unknown or incomplete manifest namespace."})
                    return
                if scalar_reader is None:
                    payload = {
                        "namespace": namespace,
                        "evidence": [],
                        "assertions": [],
                        "views": [],
                        "history_views": [],
                        "facts": [],
                        "memory_inventory": [],
                        "audit_pass_id": None,
                        "audit": [],
                        "audit_warning": None,
                        "graph_warning": "Scalar graph access is not configured for this dashboard.",
                    }
                else:
                    try:
                        payload = scalar_reader.read(namespace)
                    except Exception as exc:  # noqa: BLE001 - explorer keeps manifest/scoring usable
                        payload = {
                            "namespace": namespace,
                            "evidence": [],
                            "assertions": [],
                            "views": [],
                            "history_views": [],
                            "facts": [],
                            "memory_inventory": [],
                            "audit_pass_id": None,
                            "audit": [],
                            "audit_warning": None,
                            "graph_warning": f"Scalar graph is unavailable: {exc}",
                        }
                payload["task"] = task
                payload["scoring"] = scoring_rows(snaps, task["question_id"])
                self._json(200, payload)
                return
            if route.path not in ("/", ""):
                self._json(404, {"error": "Not found."})
                return
            menhir = probe_menhir(menhir_url) if menhir_url else None
            page = render_html(
                snaps,
                menhir,
                total_items=total_items,
                refresh_s=refresh_s,
                items_n=items_n,
                ingests=ingests,
                scalar_viewer_enabled=bool(catalog),
                scalar_default_namespace=scalar_default_namespace,
                episode_progress=scan_episode_progress(results_dir),
            )
            data = page.encode("utf-8")
            self._send(200, data, "text/html; charset=utf-8")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard serving at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n(dashboard stopped)")
        server.shutdown()
    finally:
        close = getattr(scalar_reader, "close", None)
        if close:
            close()


def run_dashboard(
    results_dir: Path,
    *,
    menhir_url: str | None = None,
    interval: float = 5.0,
    once: bool = False,
    total_items: int | None = None,
    active_within_s: float | None = None,
) -> None:
    prev_done: int | None = None
    prev_t: float | None = None
    rate_per_min: float | None = None
    eta_min: float | None = None
    while True:
        snaps = scan_runs(results_dir, active_within_s=active_within_s)
        menhir = probe_menhir(menhir_url) if menhir_url else None

        active = snaps[0] if snaps else None
        if active is not None:
            now = time.time()
            done = active.total_done
            if prev_done is not None and prev_t is not None and now > prev_t:
                delta = done - prev_done
                dt_min = (now - prev_t) / 60.0
                if delta > 0 and dt_min > 0:
                    rate_per_min = delta / dt_min
                    per_arm_total = total_items or _KNOWN_VARIANT_ITEMS.get(active.variant)
                    grand = (per_arm_total * max(1, len(active.arms))) if per_arm_total else None
                    if grand:
                        eta_min = max(0.0, (grand - done) / rate_per_min)
            prev_done, prev_t = done, now

        out = render(snaps, menhir, total_items=total_items, rate_per_min=rate_per_min, eta_min=eta_min)
        if once:
            print(out)
            return
        # clear screen + repaint
        print("\033[2J\033[H" + out, flush=True)
        time.sleep(interval)
