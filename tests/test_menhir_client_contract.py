"""Contract tests for HttpMenhirClient -- the client the real LME ingest actually uses.

Why this file exists: HttpMenhirClient had NO test coverage, so a `diff` reference was added
to `ingest()`'s body without adding it to `ingest()`'s signature. Every test passed (they all
drive StubMenhirClient, which DID get the parameter) while every real ingest died with
`NameError: name 'diff' is not defined`. It surfaced only when a live LME build was attempted.

Two guards, both cheap:
  1. signature parity between the stub and the real client -- a param added to one must be
     added to the other, which is what actually drifted.
  2. exercise the real ingest()/ingest_raw() bodies against a fake transport, so an unresolved
     name in the payload-building path fails here instead of a day into a graph build.
"""

from __future__ import annotations

import inspect

import pytest

from archolith_bench.harness.menhir_client import HttpMenhirClient, StubMenhirClient


class _FakeResponse:
    status_code = 200

    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {"episode_id": "episode-1", "status": "QUEUED"}

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict:
        return self._data


class _FakeHttpx:
    """Records calls so the test can assert on the built payload."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.response_data: dict | None = None

    def post(self, url, *, params=None, json=None, headers=None, **kw):
        self.posts.append({"url": url, "params": params, "json": json})
        return _FakeResponse(self.response_data)

    def delete(self, url, **kw):  # pragma: no cover - unused here
        return _FakeResponse()

    def close(self) -> None:  # pragma: no cover - trivial
        return None


@pytest.fixture()
def http_client() -> tuple[HttpMenhirClient, _FakeHttpx]:
    client = HttpMenhirClient("http://localhost:9999")
    fake = _FakeHttpx()
    client._client = fake  # type: ignore[attr-defined]
    return client, fake


def test_ingest_signature_parity_stub_vs_http():
    """A kwarg on one client must exist on the other. This is the drift that shipped."""
    stub = set(inspect.signature(StubMenhirClient.ingest).parameters) - {"self"}
    http = set(inspect.signature(HttpMenhirClient.ingest).parameters) - {"self"}
    assert stub == http, (
        f"ingest() kwargs drifted: stub-only={sorted(stub - http)}, http-only={sorted(http - stub)}"
    )


def test_ingest_body_resolves_every_name(http_client):
    """The regression: ingest() referenced `diff` without declaring it -> NameError at runtime.

    Passing every optional kwarg walks each `if X is not None` branch in the payload builder,
    so any name the body references but the signature omits raises here.
    """
    client, fake = http_client
    result = client.ingest(
        "ns-1",
        "user",
        "hello world",
        occurred_at="2024-01-01T00:00:00Z",
        session_id="s-1",
        source="user",
        diff="--- a\n+++ b\n",
        wait=False,
        flagged=True,
        bootstrap_scope="general",
    )
    assert len(fake.posts) == 1
    payload = fake.posts[0]["json"]
    assert payload["namespace"] == "ns-1"
    assert payload["episode"] == "user: hello world"
    assert payload["source"] == "user"
    assert payload["diff"] == "--- a\n+++ b\n"
    assert payload["occurred_at"] == "2024-01-01T00:00:00Z"
    assert result == {"episode_id": "episode-1", "status": "QUEUED"}


def test_ingest_omits_unset_optionals(http_client):
    """Unset optionals must stay out of the payload rather than posting explicit nulls."""
    client, fake = http_client
    client.ingest("ns-1", "user", "hello")
    payload = fake.posts[0]["json"]
    for key in ("diff", "source", "occurred_at", "session_id"):
        assert key not in payload, f"unset {key} must not be sent"


def test_ingest_skips_empty_content(http_client):
    """Empty content is a no-op -- LME haystacks contain blank turns."""
    client, fake = http_client
    client.ingest("ns-1", "user", "")
    assert fake.posts == []


def test_record_turn_evidence_forwards_source_time(http_client):
    client, fake = http_client

    client.record_turn_evidence(
        "ns-1",
        "I have added 25 postcards.",
        session_id="s-1",
        occurred_at="2023-11-30T20:25:00Z",
        turn_key="turn-25",
    )

    payload = fake.posts[0]["json"]
    assert fake.posts[0]["url"].endswith("/api/turn-evidence")
    assert payload["session_id"] == "s-1"
    assert payload["occurred_at"] == "2023-11-30T20:25:00Z"
    assert payload["turn_key"] == "turn-25"


def test_ingest_raw_body_resolves_every_name(http_client):
    """ingest_raw() posts an exact episode body for deterministic probes."""
    client, fake = http_client
    client.ingest_raw("ns-1", "exact body", source="user", wait=False)
    payload = fake.posts[0]["json"]
    assert payload == {"episode": "exact body", "namespace": "ns-1", "source": "user"}


def test_recall_preserves_scalar_authority_and_provenance_for_llm(http_client):
    """A stale semantic value must not outrank a correct current scalar View in flat text."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "legacy-125",
                "name": "Starbucks Rewards app",
                "content": "User needs 125 stars to reach the Gold level.",
                "memory_type": "SEMANTIC",
                "is_scalar_authority": False,
                "is_superseded_view": False,
            },
            {
                "uuid": "view-120",
                "name": "user's stars needed (gold_level): 120",
                "content": "current stars needed (gold_level) = 120.",
                "memory_type": "SCALAR_STATE",
                "is_scalar_authority": True,
                "is_superseded_view": False,
            },
        ],
        "authority_layer": [
            {
                "kind": "current",
                "status": "leads",
                "subject": "user",
                "attribute": "stars_needed",
                "scope": "gold_level",
                "value": "120",
                "valid_at": "2023-07-30T03:56:00Z",
                "view_uuid": "view-120",
                "has_foundation": True,
                "contributors": [
                    {
                        "operation": "absolute",
                        "stated_span": "I need 120 stars to reach the gold level",
                        "valid_at": "2023-07-30T03:56:00Z",
                    }
                ],
            }
        ],
    }

    recalled = client.recall("lme-0f05491a", "How many stars do I need?", limit=10)

    assert recalled[0].startswith("[AUTHORITATIVE CURRENT MEMORY]")
    assert "current fact: user — stars needed (gold level) = 120" in recalled[0]
    assert "I need 120 stars to reach the gold level" in recalled[0]
    assert "2023-07-30T03:56:00Z" in recalled[0]
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "125 stars" in recalled[1]
    assert sum("current stars needed" in snippet for snippet in recalled) == 0


def test_recall_humanizes_authoritative_duration_without_losing_normalized_value(http_client):
    """Duration authority must not expose a bare normalized number that loses the source clock value."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "old-personal-best",
                "name": "personal best time",
                "content": "User set a personal best time of 27:12.",
                "memory_type": "SEMANTIC",
            }
        ],
        "authority_layer": [
            {
                "kind": "current",
                "status": "leads",
                "subject": "user",
                "attribute": "personal_best_time",
                "value_kind": "duration",
                "unit": "seconds",
                "value": "1550",
                "valid_at": "2023-05-27T10:20:00Z",
                "view_uuid": "current-personal-best",
                "has_foundation": True,
                "contributors": [
                    {
                        "operation": "absolute",
                        "stated_span": "hoping to beat my personal best time of 25:50",
                        "valid_at": "2023-05-27T10:20:00Z",
                    }
                ],
            }
        ],
    }

    recalled = client.recall("lme-6a1eabeb", "What was my personal best time?", limit=10)

    assert "current fact: user — personal best time = 1550 seconds (25 minutes 50 seconds; 25:50)" in recalled[0]
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "27:12" in recalled[1]


def test_recall_without_authority_layer_preserves_legacy_serialization(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "name": "Biscuit",
                "content": "The user's dog is named Biscuit.",
                "memory_type": "SEMANTIC",
            }
        ]
    }

    assert client.recall("ns-1", "dog name") == ["Biscuit: The user's dog is named Biscuit."]


def test_recall_without_scalar_authority_preserves_source_time_for_chronology(http_client):
    """The French-press fallback must expose world time instead of relying on result order."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "ratio-5",
                "name": "French press ratio",
                "content": "Use 5 oz of water.",
                "memory_type": "SEMANTIC",
                "temporal_facts": [{
                    "fact": "French press ratio uses 5 oz of water",
                    "valid_at": "2023-06-30T11:33:00Z",
                    "created_at": "2026-07-30T05:01:00Z",
                    "temporal_role": "current_belief",
                }],
            },
            {
                "uuid": "ratio-6",
                "name": "French press ratio",
                "content": "Use 6 oz of water.",
                "memory_type": "SEMANTIC",
                "temporal_facts": [{
                    "fact": "French press ratio uses 6 oz of water",
                    "valid_at": "2023-02-11T17:37:00Z",
                    "invalid_at": "2023-06-30T11:33:00Z",
                    "created_at": "2026-07-30T05:00:00Z",
                    "expired_at": "2026-07-30T05:01:00Z",
                    "temporal_role": "superseded_belief",
                }],
            },
        ],
        "authority_layer": [],
    }

    recalled = client.recall(
        "lme-6071bd76", "Did I switch to more or less water?", limit=10
    )

    assert "2023-06-30T11:33:00Z | French press ratio uses 5 oz of water" in recalled[0]
    assert (
        "2023-02-11T17:37:00Z through 2023-06-30T11:33:00Z"
        " | French press ratio uses 6 oz of water"
    ) in recalled[1]
    assert "belief: current belief" in recalled[0]
    assert "belief: superseded belief" in recalled[1]
    assert all("2026-07-30" not in memory for memory in recalled)
    assert fake.posts[0]["json"]["include_invalidated"] is True


def test_recall_marks_missing_source_time_unknown(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [{
            "name": "French press ratio",
            "content": "Use 5 oz of water.",
            "temporal_facts": [],
        }]
    }

    assert client.recall("lme-6071bd76", "When?") == [
        "French press ratio: Use 5 oz of water.\nsource time: unknown"
    ]


def test_recall_ignores_timestamp_only_temporal_bookkeeping(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [{
            "name": "French press ratio",
            "content": "Use 5 oz of water.",
            "temporal_facts": [
                {
                    "created_at": "2026-07-30T05:01:00Z",
                    "temporal_role": "current_belief",
                },
                {
                    "fact": "French press ratio uses 5 oz of water",
                    "valid_at": "2023-06-30T11:33:00Z",
                    "temporal_role": "current_belief",
                },
            ],
        }]
    }

    recalled = client.recall("lme-6071bd76", "When?")

    assert recalled == [
        "French press ratio: Use 5 oz of water.\n"
        "source-time evidence (when the fact was true):\n"
        "- 2023-06-30T11:33:00Z | French press ratio uses 5 oz of water"
        " | belief: current belief"
    ]
    assert "supporting fact text unavailable" not in recalled[0]


def test_recall_renders_said_time_only_when_menhir_sends_it(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [{
            "name": "Fitbit",
            "content": "User wears a Fitbit.",
            "temporal_facts": [
                {
                    "fact": "User has worn a Fitbit for 3 months",
                    "valid_at": "2023-03-01T00:00:00Z",
                    "said_at": "2023-06-01T10:00:00Z",
                    "temporal_role": "current_belief",
                },
                {
                    "fact": "User switched to an Apple Watch",
                    "valid_at": "2023-05-20T00:00:00Z",
                    "temporal_role": "current_belief",
                },
            ],
        }]
    }

    recalled = client.recall("lme-08e075c7", "What watch?")

    assert recalled == [
        "Fitbit: User wears a Fitbit.\n"
        "source-time evidence (when the fact was true):\n"
        "- 2023-03-01T00:00:00Z | said 2023-06-01T10:00:00Z"
        " | User has worn a Fitbit for 3 months | belief: current belief\n"
        "- 2023-05-20T00:00:00Z | User switched to an Apple Watch | belief: current belief"
    ]


def test_recall_labels_string_results_when_authority_is_present(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": ["User needs 125 stars."],
        "authority_layer": [
            {
                "kind": "current",
                "status": "leads",
                "attribute": "stars_needed",
                "value": "120",
                "has_foundation": True,
            }
        ],
    }

    recalled = client.recall("ns-1", "stars")

    assert recalled[0].startswith("[AUTHORITATIVE CURRENT MEMORY]")
    assert recalled[1] == "[RELATED MEMORY | non-authoritative] User needs 125 stars."


def test_recall_event_authority_unique_lead_renders_full_history(http_client):
    """A status='leads' event verdict must render the selected object and its evidence."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "stale-espresso",
                "name": "Espresso machine",
                "content": "User was thinking about buying an espresso machine.",
                "memory_type": "SEMANTIC",
            }
        ],
        "event_authority_layer": [
            {
                "status": "leads",
                "gate": "pass",
                "kind": "purchase",
                "predicate": "purchased",
                "subject_uuid": "user-7",
                "object_display": "espresso machine",
                "object_key": "entity:appliance:espresso_machine",
                "valid_at": "2023-07-30T03:56:00Z",
                "time_basis": "world",
                "domain": "purchases",
                "stated_span": "I bought an espresso machine yesterday",
                "assertion_key": "assertion:espresso:2023",
                "episode_uuid": "ep-55",
                "turn_evidence_uuid": "turn-88",
                "has_foundation": True,
            }
        ],
    }

    recalled = client.recall("lme-e1", "Did I buy an espresso machine?", limit=10)

    assert recalled[0].startswith("[AUTHORITATIVE EVENT HISTORY]")
    assert "event: espresso machine" in recalled[0]
    assert "predicate: purchased" in recalled[0]
    assert "object key: entity:appliance:espresso_machine" in recalled[0]
    assert "valid at: 2023-07-30T03:56:00Z" in recalled[0]
    assert "time basis: world" in recalled[0]
    assert "domain: purchases" in recalled[0]
    assert '"I bought an espresso machine yesterday"' in recalled[0]
    assert "evidence identities: assertion:espresso:2023, ep-55, turn-88" in recalled[0]
    assert "gate: pass" in recalled[0]
    assert "preference:" in recalled[0]
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "thinking about buying" in recalled[1]


def test_recall_event_blocking_anchor_gate_returns_advisory_without_items(http_client):
    """A blocking selection-failure gate must render an advisory and suppress ordinary items."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "tempting-1",
                "name": "Tempting memory",
                "content": "User owns a red bike.",
                "memory_type": "SEMANTIC",
            },
            {
                "uuid": "tempting-2",
                "name": "Another tempting memory",
                "content": "User is planning to buy a bike.",
                "memory_type": "SEMANTIC",
            },
        ],
        "event_authority_layer": [
            {
                "status": "leads",
                "gate": "pass",
                "predicate": "acquired",
                "object_display": "blue bike",
            },
            {
                "status": "advisory",
                "gate": "anchor",
                "reason": "no evidence-anchored object could be selected",
            }
        ],
    }

    recalled = client.recall("lme-e2", "Which bike did I buy?", limit=10)

    assert len(recalled) == 1
    assert recalled[0].startswith("[EVENT HISTORY VERDICT | advisory]")
    assert "gate: anchor" in recalled[0]
    assert "no evidence-anchored object could be selected" in recalled[0]
    assert all("bike" not in m.lower() or "gate" in m.lower() for m in recalled)


def test_recall_event_blocking_gates_all_labeled_variants(http_client):
    """Every blocking selection-failure gate must suppress ordinary items the same way."""
    client, fake = http_client
    for gate in ("ambiguity", "time", "scope", "no_candidate"):
        fake.response_data = {
            "results": [{"name": "tempt", "content": "A tempting recollection."}],
            "event_authority_layer": [{"status": "advisory", "gate": gate}],
        }
        recalled = client.recall("lme-e3", "which?", limit=10)
        assert len(recalled) == 1, f"{gate} should suppress ordinary items"
        assert recalled[0].startswith("[EVENT HISTORY VERDICT | advisory]")
        assert f"gate: {gate}" in recalled[0]
        assert "tempting recollection" not in recalled[0]


def test_recall_event_blocking_gate_requires_advisory_status(http_client):
    """A contradictory leads record must not activate advisory-only suppression."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "name": "Related note",
                "content": "A normal recollection.",
                "memory_type": "SEMANTIC",
            }
        ],
        "event_authority_layer": [
            {
                "status": "leads",
                "gate": "anchor",
                "predicate": "acquired",
                "object_display": "blue bike",
            }
        ],
    }

    recalled = client.recall("lme-e3c", "which?", limit=10)

    assert recalled[0].startswith("[AUTHORITATIVE EVENT HISTORY]")
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "normal recollection" in recalled[1]


def test_recall_event_nonblocking_advisory_does_not_suppress_items(http_client):
    """A non-blocking advisory (status=advisory, gate=pass) labels but keeps ordinary items."""
    client, fake = http_client
    fake.response_data = {
        "results": [{"name": "note", "content": "A normal recollection.", "memory_type": "SEMANTIC"}],
        "event_authority_layer": [
            {"status": "advisory", "gate": "pass", "reason": "event noted but not authoritative"}
        ],
    }

    recalled = client.recall("lme-e3b", "note", limit=10)

    assert recalled[0].startswith("[EVENT HISTORY VERDICT | advisory]")
    assert "gate: pass" in recalled[0]
    assert "event noted but not authoritative" in recalled[0]
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")


def test_recall_event_authority_with_scalar_layer_both_render_first(http_client):
    """Scalar and event authority may coexist; both leads render before ordinary items."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "rel-mem",
                "name": "Old note",
                "content": "User keeps an old notebook entry.",
                "memory_type": "SEMANTIC",
            }
        ],
        "authority_layer": [
            {
                "kind": "current",
                "status": "leads",
                "attribute": "gold_level",
                "value": "120",
                "has_foundation": True,
                "view_uuid": "view-120",
            }
        ],
        "event_authority_layer": [
            {
                "status": "leads",
                "gate": "pass",
                "predicate": "reached",
                "object_display": "gold level",
                "valid_at": "2023-07-30T03:56:00Z",
                "stated_span": "I finally hit the gold level",
            }
        ],
    }

    recalled = client.recall("lme-e4", "gold level", limit=10)

    assert recalled[0].startswith("[AUTHORITATIVE CURRENT MEMORY]")
    assert "gold level = 120" in recalled[0]
    assert recalled[1].startswith("[AUTHORITATIVE EVENT HISTORY]")
    assert "event: gold level" in recalled[1]
    assert recalled[2].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "old notebook entry" in recalled[2]


def test_recall_event_blocking_suppresses_scalar_authority_and_items(http_client):
    """A blocking anchor advisory must hide scalar current-state authority and ordinary items."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {
                "uuid": "tempting",
                "name": "Tempting memory",
                "content": "User bought a blue bike.",
                "memory_type": "SEMANTIC",
            }
        ],
        "authority_layer": [
            {
                "kind": "current",
                "status": "leads",
                "attribute": "bike_color",
                "value": "blue",
                "has_foundation": True,
                "view_uuid": "view-bike",
            }
        ],
        "event_authority_layer": [
            {
                "status": "advisory",
                "gate": "anchor",
                "reason": "no evidence-anchored object could be selected",
            }
        ],
    }

    recalled = client.recall("lme-e6", "What color was the bike before?", limit=10)

    assert len(recalled) == 1
    assert recalled[0].startswith("[EVENT HISTORY VERDICT | advisory]")
    assert "gate: anchor" in recalled[0]
    assert all("AUTHORITATIVE CURRENT MEMORY" not in m for m in recalled)
    assert all("bike_color" not in m for m in recalled)
    assert all("blue bike" not in m for m in recalled)


def test_recall_event_authority_tolerates_malformed_and_empty_layers(http_client):
    """A malformed event layer must never crash recall or invent items."""
    client, fake = http_client
    # malformed: string layers, non-string object_display, missing evidence identity fields
    fake.response_data = {
        "results": [{"name": "Ok", "content": "A normal recollection."}],
        "authority_layer": "not-a-list",
        "event_authority_layer": {
            "status": "leads",
            "gate": "pass",
            "object_display": 42,  # not a string -> fall back to predicate
            "predicate": "purchased",
            "assertion_key": "assertion:7",
            "turn_evidence_uuid": None,
        },
    }

    recalled = client.recall("lme-e5", "ok", limit=10)

    assert recalled[0].startswith("[AUTHORITATIVE EVENT HISTORY]")
    assert "event: purchased" in recalled[0]
    assert "evidence identities: assertion:7" in recalled[0]
    assert recalled[1].startswith("[RELATED memory MEMORY | non-authoritative]")

    # empty layers: legacy behavior with no authority and no event layer present
    fake.response_data = {"results": [{"name": "Biscuit", "content": "The dog is Biscuit."}]}
    assert client.recall("lme-e5", "dog") == ["Biscuit: The dog is Biscuit."]

    # event layer present but empty list -> still legacy-identical
    fake.response_data = {
        "results": [{"name": "Biscuit", "content": "The dog is Biscuit."}],
        "event_authority_layer": [],
    }
    assert client.recall("lme-e5", "dog") == ["Biscuit: The dog is Biscuit."]


_STARS_AUTHORITY = {
    "kind": "current",
    "status": "leads",
    "subject": "user",
    "attribute": "stars_needed",
    "value": "120",
    "valid_at": "2023-07-30T03:56:00Z",
    "view_uuid": "view-120",
    "has_foundation": True,
    "contributors": [],
}


def _facet_fact(uuid: str, fact: str) -> dict:
    return {
        "uuid": uuid,
        "name": fact,
        "content": fact,
        "memory_type": "SEMANTIC",
        "retrieval_score_kind": "verb_facet",
        "temporal_facts": [{"fact": fact, "valid_at": "2023-05-01T00:00:00Z"}],
    }


def test_recall_verb_facet_facts_follow_verdicts_outside_the_limit(http_client):
    """Verb-facet facts go right after the verdicts and do not take one of the ``limit`` slots."""
    client, fake = http_client
    fake.response_data = {
        "results": [
            {"uuid": "m1", "name": "A", "content": "Ranked memory one.", "memory_type": "SEMANTIC"},
            {"uuid": "m2", "name": "B", "content": "Ranked memory two.", "memory_type": "SEMANTIC"},
            _facet_fact("e1", "The user needs to return the boots to Zara."),
        ],
        "authority_layer": [_STARS_AUTHORITY],
    }

    recalled = client.recall("lme-1", "How many items do I need to pick up or return?", limit=2)

    # Without the facet fact the cut keeps [verdict, m1]; the fact is inserted, nothing else moves.
    assert len(recalled) == 3
    assert recalled[0].startswith("[AUTHORITATIVE CURRENT MEMORY]")
    assert recalled[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "return the boots to Zara" in recalled[1]
    assert "2023-05-01T00:00:00Z" in recalled[1]
    assert "Ranked memory one." in recalled[2]
    assert not any("Ranked memory two." in s for s in recalled)


def test_recall_verb_facet_facts_lead_when_there_is_no_verdict(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [
            {"name": "A", "content": "Ranked memory one."},
            _facet_fact("e1", "The user still needs to pick up the navy blue blazer."),
            {"name": "B", "content": "Ranked memory two."},
        ]
    }

    recalled = client.recall("ns-1", "What do I need to pick up?", limit=1)

    assert len(recalled) == 2
    assert recalled[0].startswith("The user still needs to pick up the navy blue blazer.")
    assert recalled[1] == "A: Ranked memory one."


def test_recall_without_verb_facet_facts_keeps_the_limit_cut(http_client):
    client, fake = http_client
    fake.response_data = {
        "results": [{"name": n, "content": f"Memory {n}."} for n in ("A", "B", "C")],
        "authority_layer": [_STARS_AUTHORITY],
    }

    recalled = client.recall("lme-1", "stars", limit=2)

    assert len(recalled) == 2
    assert recalled[0].startswith("[AUTHORITATIVE CURRENT MEMORY]")
    assert "Memory A." in recalled[1]
