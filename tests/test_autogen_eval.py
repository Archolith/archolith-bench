"""Tests for the auto-generated eval harness (pure logic; stubbed I/O)."""

from __future__ import annotations

from archolith_bench.r1.autogen_eval import (
    CorpusNode,
    build_eval_set,
    build_paraphrase_prompt,
    looks_like_leak,
    stratified_sample,
)


def _nodes(n: int, *, ns: str = "default", prefix: str = "u") -> list[CorpusNode]:
    return [CorpusNode(uuid=f"{prefix}{i}", name=f"name{i}", text=f"text {i}", namespace=ns) for i in range(n)]


# --- stratified_sample -----------------------------------------------------


def test_sample_is_deterministic_for_a_seed() -> None:
    pool = _nodes(50)
    a = stratified_sample(pool, 10, seed=7)
    b = stratified_sample(pool, 10, seed=7)
    assert [x.uuid for x in a] == [x.uuid for x in b]


def test_different_seeds_generally_differ() -> None:
    pool = _nodes(50)
    a = stratified_sample(pool, 10, seed=1)
    b = stratified_sample(pool, 10, seed=2)
    assert [x.uuid for x in a] != [x.uuid for x in b]


def test_sample_returns_all_when_n_exceeds_pool() -> None:
    pool = _nodes(5)
    assert len(stratified_sample(pool, 10, seed=1)) == 5


def test_sample_empty_and_nonpositive() -> None:
    assert stratified_sample([], 10, seed=1) == []
    assert stratified_sample(_nodes(5), 0, seed=1) == []


def test_allocation_is_proportional_across_strata() -> None:
    # 80 in 'big', 20 in 'small'; sample 10 -> ~8 / ~2 by largest-remainder.
    pool = _nodes(80, ns="big", prefix="b") + _nodes(20, ns="small", prefix="s")
    sample = stratified_sample(pool, 10, seed=3)
    by_ns = {"big": 0, "small": 0}
    for node in sample:
        by_ns[node.namespace] += 1
    assert by_ns == {"big": 8, "small": 2}
    assert len(sample) == 10


def test_allocation_totals_exactly_n_with_awkward_ratios() -> None:
    # Three strata that don't divide evenly; leftover seats must still sum to n.
    pool = _nodes(7, ns="a", prefix="a") + _nodes(7, ns="b", prefix="b") + _nodes(7, ns="c", prefix="c")
    for n in (1, 4, 5, 10, 13):
        assert len(stratified_sample(pool, n, seed=n)) == n


# --- prompt + leak detection ----------------------------------------------


def test_prompt_contains_note_text_and_instruction() -> None:
    node = CorpusNode(uuid="u", name="PUBLIC_GA_ID", text="Set as the Google Analytics tracking id")
    prompt = build_paraphrase_prompt(node)
    assert "Google Analytics tracking id" in prompt
    assert "without reusing" in prompt.lower() or "without" in prompt.lower()


def test_leak_detection_flags_distinctive_token_reuse() -> None:
    node = CorpusNode(uuid="u", name="lifecycle_decay_interval_s", text="the decay interval setting")
    # Query reuses the distinctive name token -> a weak (leaky) case.
    assert looks_like_leak("what is lifecycle_decay_interval_s", node) is True


def test_leak_detection_checks_summary_identifiers_not_only_name() -> None:
    node = CorpusNode(
        uuid="u",
        name="decay schedule",
        text="Configured by MENHIR_DECAY_INTERVAL_SECONDS.",
    )
    assert looks_like_leak("what does MENHIR_DECAY_INTERVAL_SECONDS control", node) is True


def test_leak_detection_passes_genuine_paraphrase() -> None:
    node = CorpusNode(uuid="u", name="lifecycle_decay_interval_s", text="the decay interval setting")
    # Genuine paraphrase shares no distinctive (long) token with the name.
    assert looks_like_leak("how often does memory decay run", node) is False


def test_leak_detection_ignores_short_shared_words() -> None:
    node = CorpusNode(uuid="u", name="Set the id", text="Set the id value")
    # "set"/"the"/"id" are short/common; not distinctive, so not a leak.
    assert looks_like_leak("where do we set the id", node) is False


def test_leak_detection_flags_camelcase_identifier() -> None:
    node = CorpusNode(uuid="u", name="YawnResistanceDto", text="a DTO for resistances")
    assert looks_like_leak("what fields does YawnResistanceDto have", node) is True
    # a genuine paraphrase about the same topic, no identifier echo, is fine
    assert looks_like_leak("what fields were added for resistances", node) is False


def test_leak_detection_ignores_shared_topic_word() -> None:
    # The regression that motivated the identifier-shape heuristic: "decay" is a topic
    # word a real user would type, NOT a leaked identifier, so it must not be flagged.
    node = CorpusNode(uuid="u", name="lifecycle_decay_interval_s", text="the decay interval setting")
    assert looks_like_leak("how often does memory decay run", node) is False


# --- build_eval_set (stubbed corpus + generator) ---------------------------


class _StubCorpus:
    def __init__(self, nodes, clusters=None) -> None:
        self._nodes = nodes
        self._clusters = clusters or {}

    def sample_candidates(self):
        return self._nodes

    def duplicate_cluster(self, node, threshold):
        # Return siblings (excluding self; build_eval_set adds self back).
        return self._clusters.get(node.uuid, [])


class _StubGen:
    def __init__(self, mapping=None, default="a paraphrased question") -> None:
        self._mapping = mapping or {}
        self._default = default

    def generate(self, prompt: str) -> str:
        for key, val in self._mapping.items():
            if key in prompt:
                return val
        return self._default


def test_build_skips_empty_text_nodes() -> None:
    nodes = [
        CorpusNode(uuid="u1", name="n1", text="real text"),
        CorpusNode(uuid="u2", name="n2", text="   "),  # whitespace only -> skipped
    ]
    res = build_eval_set(_StubCorpus(nodes), _StubGen(), n=10, seed=1)
    assert {q.source_uuid for q in res.queries} == {"u1"}


def test_build_skips_empty_generated_query() -> None:
    nodes = _nodes(1)
    res = build_eval_set(_StubCorpus(nodes), _StubGen(default="   "), n=10, seed=1)
    assert res.queries == []
    assert res.skipped_empty == 1


def test_build_credits_duplicate_cluster_and_includes_self() -> None:
    nodes = [CorpusNode(uuid="u1", name="n1", text="a fact about proxies")]
    corpus = _StubCorpus(nodes, clusters={"u1": ["dup1", "dup2"]})
    res = build_eval_set(corpus, _StubGen(), n=10, seed=1)
    q = res.queries[0]
    assert q.gold_cluster_ids[0] == "u1"  # self first
    assert set(q.gold_cluster_ids) == {"u1", "dup1", "dup2"}
    assert res.cluster_sizes == [3]


def test_build_flags_leaky_queries_separately() -> None:
    node = CorpusNode(uuid="u1", name="PUBLIC_GA_ID", text="the Google Analytics tracking id")
    # Generator cheats and echoes the distinctive name token.
    corpus = _StubCorpus([node])
    gen = _StubGen(default="what is PUBLIC_GA_ID")
    res = build_eval_set(corpus, gen, n=10, seed=1)
    assert res.queries == []
    assert len(res.leaked) == 1
    assert res.leaked[0].source_uuid == "u1"


def test_build_records_source_text_hash() -> None:
    nodes = [CorpusNode(uuid="u1", name="n1", text="stable text")]
    res = build_eval_set(_StubCorpus(nodes), _StubGen(), n=10, seed=1)
    assert len(res.queries[0].source_text_sha256) == 64  # sha256 hex
