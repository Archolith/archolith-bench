# Bench progress harness

`archolith_bench/progress.py` — a small, reusable progress reporter for the long
bench loops. Use it whenever a bench iterates over more than a few dozen items so a
run shows a live heartbeat instead of looking hung.

**Why it exists:** a live R1 recall run is 6 conditions × 155 queries of real
graph+embedding calls (~10 minutes). The old scripts printed nothing until the end, so
the run looked frozen — and piping stdout through `tail` hid even the little that was
printed. This module fixes that with one shared, well-behaved primitive.

## Design (what makes it "sane")

- **Progress goes to STDERR.** stdout stays clean for the JSON artifact + result table,
  and progress stays visible even when stdout is redirected to a file or `tee`.
- **Throttled + flushed.** At most one update per `min_interval` seconds (default 0.5s),
  always flushed — a heartbeat, not a flood.
- **TTY-aware.** A terminal gets a single `\r`-updated line; a pipe/redirect gets one
  line per tick (so `tee` / a logfile captures the trail).
- **Rate + ETA + detail.** `i/total pct%  elapsed  rate/s  eta  <detail>`.
- **Stdlib only, sync + async.** Imports in CI with no deps; drop-in for `await` loops.

Example output (piped/non-TTY):

```
[R1 recall] 45/155  29%  12.3s  3.7/s  eta 29.7s  E_hybrid_a0
[R1 recall] 90/155  58%  24.1s  3.7/s  eta 17.5s  E_hybrid_a0.5
[R1 recall] 155/155 100%  41.6s  3.7/s  eta 0.0s  E_hybrid_a1
```

> **Do not pipe the run's stdout through `tail`.** `tail` buffers to EOF, so you see
> nothing until the run finishes. Progress is on **stderr** — just run the script (or
> `... 2>&1 | tee run.log` if you want both streams in a file). The reporter also takes
> an optional `logfile=` to append its summary lines durably.

## The three primitives

### 1. `track(iterable, ...)` — tqdm-style, sync loops

```python
from archolith_bench.progress import track

for query in track(fixture.queries, label="facet", min_interval=0.5):
    run_one(query)
```

`total` is inferred from `len(iterable)` when available; pass `total=` for generators.

### 2. `ProgressReporter` — the engine (sync **or async**)

The R1/facet recall loops are `await`-based, so drive the reporter directly:

```python
from archolith_bench.progress import ProgressReporter

progress = ProgressReporter(len(conditions) * len(queries), label="R1 recall")
for name, tuning in conditions.items():
    for q in queries:
        await recall_service.recall(q.text, tuning=tuning, ...)
        progress.advance(detail=name)   # detail shows the current condition
progress.close()                        # emits the final 100% line

# or as a context manager (sync):
with ProgressReporter(total, label="x") as progress:
    for item in items:
        ...
        progress.advance()
```

### 3. `run_ladder(conditions, items, run_one, ...)` — the common shape

For a plain sync `conditions × items` ladder, skip the boilerplate:

```python
from archolith_bench.progress import run_ladder

results = run_ladder(conditions, items, run_one, label="ladder")
# -> {condition_name: [run_one(ctx, item) for item in items]}
# heartbeat detail is the current condition name; condition-major order.
```

## Adopting it in a new bench

1. `from archolith_bench.progress import ProgressReporter` (or `track` / `run_ladder`).
2. Set `total` to the real unit count (conditions × items, or just items).
3. Call `advance(detail=...)` once per completed unit; `close()` (or a `with` block /
   `track`) at the end.
4. Run the script **without** a `tail` pipe. Add `logfile="results/<run>.progress.log"`
   if you want a durable trail (mirrors `build --check --watch`).

## API reference

`ProgressReporter(total, *, label="bench", stream=sys.stderr, min_interval=0.5,
enabled=True, logfile=None)`

- `advance(n=1, *, detail="")` — record `n` completed units; emit a throttled update.
- `close()` — emit the final line exactly once (also runs on `__exit__`).
- `enabled=False` — make every call a no-op (quiet / CI).

`track(iterable, *, total=None, label="bench", **reporter_kwargs) -> Iterator`

`run_ladder(conditions, items, run_one, *, label="ladder", **reporter_kwargs) -> dict`

`format_duration(seconds) -> str` — `4.2s` / `1m03s` / `2h05m` (used internally; exported
for consistent duration formatting elsewhere).

Tests: `tests/test_progress.py` (11 offline cases). Wired into `scripts/run_r1_dummy.py`.

> **Name note:** it is `archolith_bench.progress`, **not** `archolith_bench.harness` —
> the latter is the existing package of *external* benchmark adapters (LongBench,
> SWE-bench, MTEB, ...). This is only the run-progress primitive.
