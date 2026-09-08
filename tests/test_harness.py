"""Offline tests for the external benchmark harness (no network, no API spend)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from archolith_bench.core.api import DIRECT_URL
from archolith_bench.harness import (
    ABResult,
    AgentDojoAdapter,
    BigCodeBenchHardAdapter,
    CyberSecEvalAdapter,
    MtebAdapter,
    SweBenchAdapter,
    get_adapter,
    is_external,
    run_ab,
    run_external_ab,
    write_harness_evidence,
)
from archolith_bench.harness.external import _build_subprocess_env, _openai_env
from archolith_bench.harness.longbench_v2 import LongBenchV2Adapter, _extract_choice

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE = FIXTURES / "longbench_v2_sample.json"


def _make_send_fn(answer_by_task: dict[str, str], proxy_input_tokens: int | None = None):
    """Build a deterministic send_fn keyed by the gold letter in the prompt.

    Returns the mapped answer per task. When proxy_input_tokens is set, proxy arms
    report fewer input tokens than direct so token-reduction deltas are non-zero.
    """

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        # Identify task by its question text (present in the user message).
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        ans = "A"
        for marker, letter in answer_by_task.items():
            if marker in user:
                ans = letter
                break
        is_proxy = base_url != DIRECT_URL and proxy_input_tokens is not None
        inp = proxy_input_tokens if is_proxy else 1000
        return ans, 1.0, {"prompt_tokens": inp, "completion_tokens": 2}

    return send_fn


def test_longbench_load_tasks_from_fixture():
    adapter = LongBenchV2Adapter()
    tasks = adapter.load_tasks(fixture_path=FIXTURE)
    assert len(tasks) == 4
    assert tasks[0].answer == "A"
    assert tasks[1].answer == "B"
    assert any("ARENA_MAX" in m["content"] for m in tasks[2].prompt_messages)


def test_longbench_subset_and_limit():
    adapter = LongBenchV2Adapter()
    tasks = adapter.load_tasks(subset="single_document_qa", fixture_path=FIXTURE)
    assert len(tasks) == 2
    limited = adapter.load_tasks(limit=1, fixture_path=FIXTURE)
    assert len(limited) == 1


def test_extract_choice():
    assert _extract_choice("The answer is B.") == "B"
    assert _extract_choice("answer: C") == "C"
    assert _extract_choice("(D)") == "D"
    assert _extract_choice("A") == "A"
    assert _extract_choice("I considered A and B. Final answer: C.") == "C"
    assert _extract_choice("I considered A and B. Final choice C.") == "C"
    assert _extract_choice("I am not sure") == ""


def test_score_letter_matching():
    adapter = LongBenchV2Adapter()
    tasks = adapter.load_tasks(fixture_path=FIXTURE)
    assert adapter.score(tasks[1], "The answer is B") is True
    assert adapter.score(tasks[1], "A") is False


def test_run_ab_offline_scores_and_deltas(tmp_path):
    adapter = LongBenchV2Adapter()
    # Map each task's question to a returned letter: get 3/4 right on every arm.
    answers = {
        "satellite uplink operate": "A",   # gold A -> correct
        "silt-corrected reservoir": "B",   # gold B -> correct
        "raises the heap arena cap": "A",  # gold A -> correct
        "launch date slip": "A",           # gold C -> wrong
    }
    ab = run_ab(
        adapter,
        arms=("direct", "proxy_only", "proxy_plus_filter"),
        fixture_path=FIXTURE,
        send_fn=_make_send_fn(answers, proxy_input_tokens=600),
        configure_proxy=False,
    )
    assert isinstance(ab, ABResult)
    assert ab.arms["direct"].n == 4
    assert ab.arms["direct"].score == 0.75
    # Proxy arms send fewer input tokens -> positive reduction delta vs direct.
    assert ab.deltas["proxy_only"]["input_token_reduction_pct"] > 0
    assert ab.deltas["proxy_only"]["score_delta"] == 0.0

    out = write_harness_evidence(ab, tmp_path / "ev.json", output_format="json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["benchmark_id"] == "longbench-v2"
    assert data["arms"]["direct"]["score"] == 0.75


def test_get_adapter_unknown_raises():
    try:
        get_adapter("does-not-exist")
    except KeyError as e:
        assert "available" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown benchmark_id")


def test_build_subprocess_env_filters_secret_keys(monkeypatch):
    monkeypatch.setenv("UPSTREAM_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")

    env = _build_subprocess_env({})

    assert "UPSTREAM_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env


def test_build_subprocess_env_allows_path(monkeypatch):
    monkeypatch.setenv("PATH", "local-path")

    env = _build_subprocess_env({})

    assert env["PATH"] == "local-path"


def test_build_subprocess_env_overrides_win(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")

    env = _build_subprocess_env({"OPENAI_API_KEY": "adapter-secret"})

    assert env["OPENAI_API_KEY"] == "adapter-secret"


def test_openai_env_helper_exposes_expected_keys():
    assert _openai_env("http://example.test/v1", "secret") == {
        "OPENAI_BASE_URL": "http://example.test/v1",
        "OPENAI_API_BASE": "http://example.test/v1",
        "OPENAI_API_KEY": "secret",
    }


def test_harness_cli_offline(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "archolith_bench", "harness", "longbench-v2",
            "--offline-fixture", str(FIXTURE),
            "--format", "json",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tmp_path / "harness_longbench-v2.json").exists()


def test_stub_menhir_client_roundtrip():
    from archolith_bench.harness import StubMenhirClient
    c = StubMenhirClient()
    g = c.new_group()
    c.ingest(g, "user", "My dog is named Biscuit")
    c.ingest(g, "assistant", "Nice name!")
    out = c.recall(g, "what is my dog named", limit=10)
    assert any("Biscuit" in s for s in out)
    c.reset(g)
    assert c.recall(g, "dog", limit=10) == []


def test_stub_menhir_client_recall_preserves_insertion_order_for_ties():
    from archolith_bench.harness import StubMenhirClient
    c = StubMenhirClient()
    g = c.new_group()
    c.ingest(g, "user", "alpha")
    c.ingest(g, "assistant", "beta")
    c.ingest(g, "user", "alpha")

    out = c.recall(g, "unmatched", limit=3)

    assert out == ["user: alpha", "assistant: beta", "user: alpha"]


def test_longmemeval_answer_prompt_explains_authority_precedence():
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    memory_context = (
        "[AUTHORITATIVE CURRENT MEMORY]\n"
        "current fact: user — stars needed (gold level) = 120\n"
        "[RELATED semantic MEMORY | non-authoritative] user needs 125 stars."
    )

    messages = LongMemEvalMemoryAdapter().build_messages(
        memory_context,
        "How many stars do I need to reach gold?",
    )

    assert "canonical current value" in messages[0]["content"]
    assert "Safe veto policy" in messages[0]["content"]
    assert "subject, attribute, scope, unit, or time" in messages[0]["content"]
    assert "veto the conflicting set and say you don't know" in messages[0]["content"]
    assert "Never veto a matching, supported authoritative record" in messages[0]["content"]
    assert "compare those source times" in messages[0]["content"]
    assert "never infer chronology from retrieval/list order" in messages[0]["content"]
    assert "'superseded belief' label means the fact is historical" in messages[0]["content"]
    assert "If source time is unknown, do not invent an ordering" in messages[0]["content"]
    assert "outside knowledge suggests something else" in messages[0]["content"]
    assert "falling back to a conflicting stale value" in messages[0]["content"]
    assert messages[1]["content"].index("= 120") < messages[1]["content"].index("125 stars")


def test_assert_not_production_guards():
    """Allow-list policy: loopback passes, everything else is refused.

    Previously a deny-list of name markers, which failed open -- the real
    production endpoint (https://memory.ctharvey.me) contains none of them and
    was permitted. Only loopback passes now.
    """
    from archolith_bench.harness import assert_not_production
    assert_not_production("http://localhost:7999")  # ok
    assert_not_production("http://localhost:9800/v1")  # proxy port alone is not production.
    assert_not_production("http://127.0.0.1:8098")  # ok
    for bad in (
        "https://menhir.example.com",
        "http://prod-neo4j:7687",
        "https://staging.menhir.example.com",
        "https://preprod-memory.example.com",
        "https://preview-memory.example.com",
        "https://release-memory.example.com",
        # The endpoint the old deny-list let through:
        "https://memory.ctharvey.me",
        "https://memory.ctharvey.me/mcp-http",
        # Any other unrecognised host, which is the point of failing closed:
        "http://147.93.132.141:8090",
        "http://some-box.lan:8098",
        "",
    ):
        try:
            assert_not_production(bad)
        except SystemExit:
            continue
        raise AssertionError(f"expected refusal for {bad}")


def test_run_memory_ab_lift_offline():
    from archolith_bench.harness import StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        # Answer only if the fact is present in the (recalled) memory context.
        user = messages[-1]["content"]
        text = "I don't know"
        if "Biscuit" in user:
            text = "Biscuit"
        elif "Denver" in user:
            text = "Denver"
        return text, 1.0, {"prompt_tokens": max(1, len(user) // 4), "completion_tokens": 2}

    ab = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory", "menhir_recall"),
        fixture_path=LME_FIXTURE,
        client=StubMenhirClient(),
        send_fn=send_fn,
    )
    # no_memory: only the abstention item is right (1/3). menhir_recall: all 3.
    assert ab.arms["no_memory"].score < ab.arms["menhir_recall"].score
    assert ab.arms["menhir_recall"].score == 1.0
    assert ab.deltas["menhir_recall"]["score_delta"] > 0


def test_parse_query_list_lenient():
    from archolith_bench.harness.memory_ab import _parse_query_list
    assert _parse_query_list('["a", "b"]') == ["a", "b"]
    assert _parse_query_list('sure: ["x","y"] done') == ["x", "y"]
    assert _parse_query_list("not json at all") == []
    assert _parse_query_list("") == []


def test_agentic_recall_arm_plans_and_runs():
    """The agentic arm asks the LLM for sub-queries, recalls each, and answers."""
    from archolith_bench.harness import StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    planned: list[str] = []

    def send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "memory-search queries" in system:
            # planner call: decompose (echo keeps the stub's token-overlap recall working)
            planned.append(user)
            return f'["{user}"]', 1.0, {"prompt_tokens": 5, "completion_tokens": 5}
        text = "I don't know"
        if "Biscuit" in user:
            text = "Biscuit"
        elif "Denver" in user:
            text = "Denver"
        return text, 1.0, {"prompt_tokens": max(1, len(user) // 4), "completion_tokens": 2}

    ab = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory", "menhir_agentic_recall"),
        fixture_path=LME_FIXTURE,
        client=StubMenhirClient(),
        send_fn=send_fn,
    )
    assert "menhir_agentic_recall" in ab.arms
    assert ab.arms["menhir_agentic_recall"].n == 3  # all fixture items ran
    assert planned  # the planner was actually invoked
    # agentic recall surfaces the facts -> beats the no-memory floor
    assert ab.arms["menhir_agentic_recall"].score > ab.arms["no_memory"].score


def test_run_memory_ab_recall_only_uses_prebuilt_namespaces():
    """Recall-only reads stable lme-<question_id> namespaces in place: no ingest, no reset,
    no --confirm-menhir-reset needed."""
    from archolith_bench.harness import StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    client = StubMenhirClient()
    # Pre-build the graph once, as _ingest_lme.py leaves it: stable per-question namespaces.
    prebuilt = {
        "lme-lme-sample-001": ["user: my dog is named Biscuit"],
        "lme-lme-sample-002": ["user: I now live in Denver"],
        "lme-lme-sample-003": ["user: I went hiking with my sister"],
    }
    client._groups.update({k: list(v) for k, v in prebuilt.items()})

    def send_fn(client_, base_url, api_key, messages, model, **kwargs):
        user = messages[-1]["content"]
        text = "Biscuit" if "Biscuit" in user else ("Denver" if "Denver" in user else "I don't know")
        return text, 1.0, {"prompt_tokens": max(1, len(user) // 4), "completion_tokens": 2}

    ab = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory", "menhir_recall"),
        fixture_path=LME_FIXTURE,
        client=client,
        send_fn=send_fn,
        recall_only=True,  # reads the pre-built graph; no reset_confirmed required
    )
    # Recalled the pre-built facts -> menhir_recall beats the no_memory floor.
    assert ab.arms["menhir_recall"].score > ab.arms["no_memory"].score
    # The pre-built namespaces survive untouched: recall-only never resets or ingests,
    # so no fresh per-item group is ever created.
    assert client._groups == prebuilt


def test_run_memory_ab_recall_only_honors_namespace_template():
    """A custom namespace template is formatted with {question_id} and used verbatim."""
    from archolith_bench.harness import StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    client = StubMenhirClient()
    client._groups["q::lme-sample-001"] = ["user: my dog is named Biscuit"]

    def send_fn(client_, base_url, api_key, messages, model, **kwargs):
        user = messages[-1]["content"]
        return ("Biscuit" if "Biscuit" in user else "I don't know"), 1.0, {
            "prompt_tokens": 1, "completion_tokens": 1,
        }

    ab = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("menhir_recall",),
        fixture_path=LME_FIXTURE,
        client=client,
        send_fn=send_fn,
        recall_only=True,
        namespace_template="q::{question_id}",
        limit=1,
    )
    assert ab.arms["menhir_recall"].results[0].correct
    # Only the custom-templated namespace was touched; nothing created or reset.
    assert client._groups == {"q::lme-sample-001": ["user: my dog is named Biscuit"]}


def test_run_memory_ab_checkpoint_resumes_and_skips_done(tmp_path):
    """A second run with the same checkpoint reuses recorded items and re-issues no calls."""
    from archolith_bench.harness import MemoryCheckpoint, StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    calls = {"n": 0}

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        calls["n"] += 1
        user = messages[-1]["content"]
        text = "Biscuit" if "Biscuit" in user else ("Denver" if "Denver" in user else "I don't know")
        return text, 1.0, {"prompt_tokens": max(1, len(user) // 4), "completion_tokens": 2}

    class _Scorer:
        last_usage: dict = {}

        def __call__(self, item, response_text):
            self.last_usage = {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
            }
            return True

    ckpt_path = tmp_path / "ckpt.jsonl"

    first = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory", "menhir_recall"),
        fixture_path=LME_FIXTURE,
        client=StubMenhirClient(),
        send_fn=send_fn,
        checkpoint=MemoryCheckpoint(ckpt_path),
        score_fn=_Scorer(),
    )
    calls_after_first = calls["n"]
    assert calls_after_first > 0
    assert ckpt_path.exists()

    # Fresh checkpoint object loads the persisted results; a rerun must answer no calls.
    second = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory", "menhir_recall"),
        fixture_path=LME_FIXTURE,
        client=StubMenhirClient(),
        send_fn=send_fn,
        checkpoint=MemoryCheckpoint(ckpt_path),
        score_fn=_Scorer(),
    )
    assert calls["n"] == calls_after_first, "resume should not re-issue any answer calls"
    # Identical aggregates from the checkpoint as from the live run.
    for arm in ("no_memory", "menhir_recall"):
        assert second.arms[arm].n == first.arms[arm].n
        assert second.arms[arm].score == first.arms[arm].score
        assert second.arms[arm].input_tokens == first.arms[arm].input_tokens
        assert second.arms[arm].results[0].scorer_input_tokens == 7
        assert second.arms[arm].results[0].scorer_output_tokens == 1
        assert second.arms[arm].results[0].scorer_raw_usage["total_tokens"] == 8


def test_run_memory_ab_no_memory_arm_receives_chat_client():
    from archolith_bench.harness import run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    seen_clients = []

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        seen_clients.append(client)
        return "I don't know", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    ab = run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory",),
        fixture_path=LME_FIXTURE,
        send_fn=send_fn,
    )

    assert ab.arms["no_memory"].n == 3
    assert seen_clients
    assert all(client is not None for client in seen_clients)


def test_run_memory_ab_closes_chat_client(monkeypatch):
    import archolith_bench.harness.memory_ab as memory_ab_module
    from archolith_bench.harness import run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    class FakeChatClient:
        def __init__(self, timeout):
            self.timeout = timeout
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True

    fake_client = FakeChatClient(timeout=300)
    monkeypatch.setattr(memory_ab_module.httpx, "Client", lambda timeout: fake_client)

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        assert client is fake_client
        return "I don't know", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory",),
        fixture_path=LME_FIXTURE,
        send_fn=send_fn,
    )

    assert fake_client.closed is True


def test_run_memory_ab_closes_http_menhir_client():
    from archolith_bench.harness import HttpMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    client = HttpMenhirClient("http://throwaway-menhir.local")
    closed = False

    def close():
        nonlocal closed
        closed = True

    client.close = close

    def send_fn(chat_client, base_url, api_key, messages, model, **kwargs):
        return "I don't know", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("no_memory",),
        fixture_path=LME_FIXTURE,
        client=client,
        send_fn=send_fn,
    )

    assert closed is True


def test_run_memory_ab_requires_confirmation_before_real_menhir_reset():
    from archolith_bench.harness import HttpMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    client = HttpMenhirClient("http://throwaway-menhir.local")
    try:
        try:
            run_memory_ab(
                LongMemEvalMemoryAdapter(),
                arms=("menhir_recall",),
                fixture_path=LME_FIXTURE,
                client=client,
                send_fn=lambda *args, **kwargs: ("I don't know", 1.0, {}),
            )
        except ValueError as e:
            assert "--confirm-menhir-reset" in str(e)
        else:
            raise AssertionError("expected reset confirmation refusal")
    finally:
        client.close()


def test_run_memory_ab_dry_run_skips_real_menhir_reset():
    from archolith_bench.harness import HttpMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    class FakeHttpMenhirClient(HttpMenhirClient):
        def __init__(self):
            self.reset_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def new_group(self):
            return "group-1"

        def ingest(self, group_id, role, content):
            return None

        def recall(self, group_id, query, limit=10):
            return ["Biscuit"]

        def reset(self, group_id):
            self.reset_calls += 1

    client = FakeHttpMenhirClient()

    def send_fn(chat_client, base_url, api_key, messages, model, **kwargs):
        return "Biscuit", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    run_memory_ab(
        LongMemEvalMemoryAdapter(),
        arms=("menhir_recall",),
        fixture_path=LME_FIXTURE,
        client=client,
        send_fn=send_fn,
        dry_run_reset=True,
    )

    assert client.reset_calls == 0


def test_harness_cli_memory_offline(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "archolith_bench", "harness", "longmemeval-menhir",
            "--offline-fixture", str(LME_FIXTURE),
            "--format", "json", "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tmp_path / "harness_longmemeval-menhir.json").exists()


def test_harness_cli_list():
    result = subprocess.run(
        [sys.executable, "-m", "archolith_bench", "harness", "--list"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    for bid in ("longbench-v2", "bigcodebench-hard", "longmemeval", "longmemeval-menhir",
                "swe-bench", "cyberseceval-4", "agentdojo", "mteb-retrieval"):
        assert bid in result.stdout


# ---- BigCodeBench (in-process, real code execution) ----

BCB_FIXTURE = FIXTURES / "bigcodebench_hard_sample.json"


def test_bigcodebench_load_and_score():
    adapter = BigCodeBenchHardAdapter()
    tasks = adapter.load_tasks(fixture_path=BCB_FIXTURE)
    assert len(tasks) == 1
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert adapter.score(tasks[0], good) is True
    assert adapter.score(tasks[0], bad) is False


def test_bigcodebench_run_ab_executes(tmp_path):
    adapter = BigCodeBenchHardAdapter()

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        return "```python\ndef add(a, b):\n    return a + b\n```", 1.0, {
            "prompt_tokens": 100, "completion_tokens": 20,
        }

    ab = run_ab(adapter, arms=("direct",), fixture_path=BCB_FIXTURE,
                send_fn=send_fn, configure_proxy=False)
    assert ab.arms["direct"].score == 1.0


# ---- External-harness wrappers (parser + A/B over fixtures) ----

def test_external_adapters_are_external():
    for a in (SweBenchAdapter(), CyberSecEvalAdapter(), AgentDojoAdapter(), MtebAdapter()):
        assert is_external(a)


def test_swebench_parse_summary():
    adapter = SweBenchAdapter()
    s = adapter.parse_summary({"total_instances": 10, "resolved_instances": 4})
    assert s["n"] == 10 and s["score"] == 0.4


def test_mteb_parse_summary_mean():
    adapter = MtebAdapter()
    s = adapter.parse_summary({"results": [{"main_score": 0.7}, {"main_score": 0.5}]})
    assert s["n"] == 2 and abs(s["score"] - 0.6) < 1e-9


def test_run_external_ab_with_per_arm_fixtures(tmp_path):
    adapter = SweBenchAdapter()
    # direct uses more input tokens than the proxy arm -> positive reduction delta.
    direct_f = tmp_path / "direct.json"
    proxy_f = tmp_path / "proxy.json"
    direct_f.write_text(json.dumps(
        {"total_instances": 10, "resolved_instances": 4, "input_tokens": 100000, "output_tokens": 2000}))
    proxy_f.write_text(json.dumps(
        {"total_instances": 10, "resolved_instances": 4, "input_tokens": 60000, "output_tokens": 2000}))
    ab = run_external_ab(
        adapter,
        arms=("direct", "proxy_only"),
        configure_proxy=False,
        results_fixtures={"direct": direct_f, "proxy_only": proxy_f},
    )
    assert ab.arms["direct"].score == 0.4
    assert ab.deltas["proxy_only"]["input_token_reduction_pct"] == 40.0
    assert ab.deltas["proxy_only"]["score_delta"] == 0.0


# ---- LongMemEval (in-process memory QA) ----

LME_FIXTURE = FIXTURES / "longmemeval_sample.json"


def test_longmemeval_load_and_history():
    from archolith_bench.harness.longmemeval import LongMemEvalAdapter
    adapter = LongMemEvalAdapter()
    tasks = adapter.load_tasks(fixture_path=LME_FIXTURE)
    assert len(tasks) == 3
    # History flattened into chat turns + final question.
    assert tasks[0].prompt_messages[0]["role"] == "system"
    assert any("Biscuit" in m["content"] for m in tasks[0].prompt_messages)
    assert tasks[0].prompt_messages[-1]["content"] == "What is the name of my dog?"


def test_longmemeval_scoring():
    from archolith_bench.harness.longmemeval import LongMemEvalAdapter
    adapter = LongMemEvalAdapter()
    tasks = adapter.load_tasks(fixture_path=LME_FIXTURE)
    assert adapter.score(tasks[0], "Your dog's name is Biscuit.") is True
    assert adapter.score(tasks[0], "I think it was Rex.") is False
    # knowledge-update: must use the latest fact.
    assert adapter.score(tasks[1], "You live in Denver now.") is True
    assert adapter.score(tasks[1], "You live in Boston.") is False
    # abstention: declining is correct.
    assert adapter.score(tasks[2], "I don't know that.") is True
    assert adapter.score(tasks[2], "It is Marie.") is False
    assert adapter.score(tasks[0], "The answer is not Biscuit.") is False


def test_longmemeval_run_ab_offline():
    from archolith_bench.harness.longmemeval import LongMemEvalAdapter
    adapter = LongMemEvalAdapter()

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        q = messages[-1]["content"]
        text = "I don't know"
        if "dog" in q:
            text = "Biscuit"
        elif "city" in q:
            text = "Denver"
        return text, 1.0, {"prompt_tokens": 200, "completion_tokens": 3}

    ab = run_ab(adapter, arms=("direct",), fixture_path=LME_FIXTURE,
                send_fn=send_fn, configure_proxy=False)
    # dog + city correct, abstention correct -> 3/3.
    assert ab.arms["direct"].score == 1.0


def test_harness_cli_external_offline(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "archolith_bench", "harness", "swe-bench",
            "--arms", "direct,proxy_only",
            "--offline-fixture", str(FIXTURES / "swebench_report_sample.json"),
            "--format", "json", "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (tmp_path / "harness_swe-bench.json").exists()


# ---- Menhir bearer key resolution ----

def test_resolve_menhir_bearer_key_agent_wins(monkeypatch):
    from archolith_bench import cli as cli_module
    monkeypatch.setenv("MENHIR_AGENT_KEY", "agent-value")
    monkeypatch.setenv("MENHIR_API_KEY", "api-value")
    monkeypatch.setattr(cli_module, "API_KEY", "upstream-value")
    assert cli_module._resolve_menhir_bearer_key() == "agent-value"


def test_resolve_menhir_bearer_key_api_wins_when_agent_missing(monkeypatch):
    from archolith_bench import cli as cli_module
    monkeypatch.delenv("MENHIR_AGENT_KEY", raising=False)
    monkeypatch.setenv("MENHIR_API_KEY", "api-value")
    monkeypatch.setattr(cli_module, "API_KEY", "upstream-value")
    assert cli_module._resolve_menhir_bearer_key() == "api-value"


def test_resolve_menhir_bearer_key_fallback_to_upstream(monkeypatch):
    from archolith_bench import cli as cli_module
    monkeypatch.delenv("MENHIR_AGENT_KEY", raising=False)
    monkeypatch.delenv("MENHIR_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "API_KEY", "upstream-value")
    assert cli_module._resolve_menhir_bearer_key() == "upstream-value"


def test_resolve_menhir_bearer_key_empty_when_none_set(monkeypatch):
    from archolith_bench import cli as cli_module
    monkeypatch.delenv("MENHIR_AGENT_KEY", raising=False)
    monkeypatch.delenv("MENHIR_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "API_KEY", "")
    assert cli_module._resolve_menhir_bearer_key() == ""


def test_generic_memory_harness_passes_resolved_key(monkeypatch, tmp_path):
    from archolith_bench import cli as cli_module
    import argparse

    monkeypatch.setenv("MENHIR_AGENT_KEY", "menhir-bearer-token")

    captured_key = [None]

    class FakeHttpMenhirClient:
        def __init__(self, url, *, api_key="", **kwargs):
            captured_key[0] = api_key
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def close(self):
            pass
        def new_group(self):
            return "g-1"
        def ingest(self, *args, **kwargs):
            pass
        def recall(self, *args, **kwargs):
            return []
        def reset(self, *args, **kwargs):
            pass

    monkeypatch.setattr("archolith_bench.harness.HttpMenhirClient", FakeHttpMenhirClient)
    monkeypatch.setattr("archolith_bench.harness.assert_not_production", lambda url: None)
    monkeypatch.setattr(cli_module, "send_chat", lambda *a, **kw: ("x", 1.0, {}))

    class MockABResult:
        model = "test"
        arms = {}
        deltas = {}

    monkeypatch.setattr("archolith_bench.harness.run_memory_ab", lambda *a, **kw: MockABResult())
    monkeypatch.setattr("archolith_bench.harness.write_harness_evidence", lambda *a, **kw: None)
    monkeypatch.setattr("archolith_bench.harness.ab_result_to_dict", lambda *a, **kw: {})
    monkeypatch.setattr("archolith_bench.cli._publish_cli_evidence", lambda *a, **kw: None)

    args = argparse.Namespace(
        benchmark_id="longmemeval-menhir",
        list_adapters=False,
        arms="menhir_recall",
        offline_fixture=None,
        menhir_url="http://localhost:8099",
        model="test-model",
        recall_limit=10,
        confirm_menhir_reset=True,
        dry_run_menhir_reset=False,
        recall_only=False,
        namespace_template="lme-{question_id}",
        format="json",
        output_dir=tmp_path,
        out=None,
        publish_evidence=None,
        public_copy=False,
        command_text="test",
        scorer="containment",
        judge_api_key="",
        judge_model="gpt-4o-mini",
        judge_url=None,
        resume=False,
        subset=None,
        limit=None,
    )

    cli_module._run_harness(args)

    assert captured_key[0] == "menhir-bearer-token"
