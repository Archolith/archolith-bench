"""Harness smoke presets: resolution, precedence, and safety."""

from __future__ import annotations

import argparse

import pytest

from archolith_bench.harness import ADAPTERS
from archolith_bench.harness.presets import (
    PRESETS,
    apply_preset,
    format_preset_list,
    get_preset,
)


def _parser() -> argparse.ArgumentParser:
    """Mirror the harness parser's preset-relevant options and defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("benchmark_id", nargs="?", default=None)
    p.add_argument("--arms", default="direct,proxy_only,proxy_plus_filter")
    p.add_argument("--subset", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--confirm-menhir-reset", action="store_true")
    p.add_argument("--menhir-url", default=None)
    return p


class TestPresetRegistry:
    def test_every_preset_targets_a_real_adapter(self):
        """A preset naming a benchmark that does not exist is dead on arrival."""
        for name, preset in PRESETS.items():
            assert preset.benchmark_id in ADAPTERS, (
                f"preset {name} targets unknown benchmark {preset.benchmark_id}"
            )

    def test_the_five_planned_presets_exist(self):
        for name in (
            "longbench-v2-smoke",
            "bigcodebench-hard-smoke",
            "swe-bench-smoke",
            "longmemeval-proxy-smoke",
            "longmemeval-menhir-smoke",
        ):
            assert name in PRESETS

    def test_unknown_preset_lists_the_known_set(self):
        with pytest.raises(KeyError) as e:
            get_preset("nope")
        assert "longbench-v2-smoke" in str(e.value)

    def test_presets_are_small(self):
        """A smoke preset that runs hundreds of items is not a smoke preset."""
        for name, preset in PRESETS.items():
            limit = preset.overrides.get("limit")
            assert limit is not None, f"{name} has no limit"
            assert limit <= 25, f"{name} limit {limit} is too large for a smoke run"


class TestPresetSafety:
    def test_no_preset_enables_a_destructive_flag(self):
        """Presets set scale, never authorisation to mutate a graph."""
        for name, preset in PRESETS.items():
            assert "confirm_menhir_reset" not in preset.overrides, name
            assert "dry_run_menhir_reset" not in preset.overrides, name

    def test_no_preset_supplies_a_menhir_target(self):
        for name, preset in PRESETS.items():
            assert "menhir_url" not in preset.overrides, name
            assert "neo4j_uri" not in preset.overrides, name

    def test_memory_preset_declares_what_it_still_needs(self):
        preset = get_preset("longmemeval-menhir-smoke")
        assert "--menhir-url" in preset.requires
        assert "--confirm-menhir-reset" in preset.requires

    def test_applying_the_memory_preset_leaves_reset_off(self):
        p = _parser()
        args = p.parse_args([])
        apply_preset(get_preset("longmemeval-menhir-smoke"), args, p)
        assert args.confirm_menhir_reset is False
        assert args.menhir_url is None


class TestPresetPrecedence:
    def test_preset_fills_unset_values(self):
        p = _parser()
        args = p.parse_args([])
        applied = apply_preset(get_preset("swe-bench-smoke"), args, p)
        assert args.benchmark_id == "swe-bench"
        assert args.limit == 5
        assert args.subset == "lite"
        assert "limit" in applied

    def test_explicit_limit_beats_the_preset(self):
        p = _parser()
        args = p.parse_args(["--limit", "50"])
        applied = apply_preset(get_preset("swe-bench-smoke"), args, p)
        assert args.limit == 50
        assert "limit" not in applied

    def test_explicit_arms_beat_the_preset(self):
        p = _parser()
        args = p.parse_args(["--arms", "direct"])
        apply_preset(get_preset("longbench-v2-smoke"), args, p)
        assert args.arms == "direct"

    def test_explicit_benchmark_id_beats_the_preset(self):
        p = _parser()
        args = p.parse_args(["longbench-v2"])
        apply_preset(get_preset("swe-bench-smoke"), args, p)
        assert args.benchmark_id == "longbench-v2"

    def test_applied_list_reports_only_what_changed(self):
        p = _parser()
        args = p.parse_args(["--limit", "7", "--arms", "direct"])
        applied = apply_preset(get_preset("longbench-v2-smoke"), args, p)
        assert "limit" not in applied
        assert "arms" not in applied
        assert "benchmark_id" in applied


class TestPresetListing:
    def test_listing_names_every_preset(self):
        out = format_preset_list()
        for name in PRESETS:
            assert name in out

    def test_listing_warns_presets_are_not_evidence(self):
        out = format_preset_list()
        assert "not promotable evidence" in out

    def test_listing_shows_required_flags(self):
        out = format_preset_list()
        assert "--confirm-menhir-reset" in out
