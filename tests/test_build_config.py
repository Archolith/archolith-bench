"""Build YAML schema, resolution, build id and CLI plumbing (no network, no bash execution)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from archolith_bench.build import __main__ as build_cli
from archolith_bench.build import config as bc

ROOT = Path(__file__).resolve().parents[1]
BUILDS = ROOT / "builds"
LME_DIR = ROOT / "scripts" / "longmemeval"
CONFIG_SH = (LME_DIR / "config.sh").read_text(encoding="utf-8")
BUILD_SH = (LME_DIR / "build_graph.sh").read_text(encoding="utf-8")

_DEFAULT_RE = re.compile(r'^(LME_[A-Z0-9_]+)="\$\{\1:-(.*?)\}"', re.MULTILINE)
SH_DEFAULTS = dict(_DEFAULT_RE.findall(CONFIG_SH))

# Keys whose config.sh default is not a plain literal (or is not in config.sh at all), with the
# value default.yaml must carry and why.
NON_LITERAL = {
    # config.sh: ${LME_ANCHORED_TIME_MODEL} (follows the model). null leaves it unset so it follows.
    "LME_ANCHORED_TIME_ATTACH_MODEL": None,
    # config.sh: ${MENHIR_CANONICAL_SELF_BINDING_MODE:-off}; "off" is the no-env result.
    "LME_CANONICAL_SELF_BINDING_MODE": "off",
    # config.sh: "not-needed" placeholder. null leaves it unset; the openai path never reads it.
    "LME_EXTRACT_API_KEY": None,
    # Read only in build_graph.sh as ${VAR:-}, never defaulted in config.sh: absent = null.
    "LME_FIXTURE_PATH": None,
    "LME_FIXTURE_COUNT": None,
    "LME_FIXTURE_SHA256": None,
}


@pytest.fixture
def default_yaml() -> Path:
    return BUILDS / "default.yaml"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def test_default_loads_with_every_schema_key(default_yaml):
    cfg = bc.resolve_yaml(default_yaml)
    for f in bc.SCHEMA:
        assert f.key in cfg[f.section], f.path
    assert sum(len(v) for v in cfg.values()) == len(bc.SCHEMA)


def test_every_var_is_distinct_and_appears_in_scripts():
    variables = [f.var for f in bc.SCHEMA if f.var]
    assert len(variables) == len(set(variables))
    assert [f.path for f in bc.SCHEMA if f.var is None] == ["dataset.limit"]
    for var in variables:
        assert var in BUILD_SH or var in CONFIG_SH, var


def test_derived_and_run_control_vars_not_in_schema():
    for var in ("LME_NEO4J_CONTAINER", "LME_NEO4J_HTTP_URL", "LME_NEO4J_PW", *bc.RUN_CONTROL_VARS):
        assert var not in bc.MAPPED_VARS


def test_default_yaml_matches_config_sh_defaults(default_yaml):
    cfg = bc.resolve_yaml(default_yaml)
    for f in bc.SCHEMA:
        if f.var is None:
            continue
        value = cfg[f.section][f.key]
        if f.var in NON_LITERAL:
            assert value == NON_LITERAL[f.var], f.path
            continue
        assert f.var in SH_DEFAULTS, f"{f.var} has no literal default in config.sh"
        sh = SH_DEFAULTS[f.var]
        assert "${" not in sh, f"{f.var} default is not literal; add it to NON_LITERAL"
        if sh == "":
            assert value is None, f.path
        else:
            assert bc.env_to_string(f, value) == sh, f.path


def test_limit_default_matches_build_graph_positional():
    assert 'LIMIT="${1:-30}"' in BUILD_SH
    assert bc.resolve_yaml(BUILDS / "default.yaml")["dataset"]["limit"] == 30


def test_extends_and_override(tmp_path):
    cfg = bc.resolve_yaml(BUILDS / "per-type-1-acceptance.yaml")
    assert cfg["dataset"]["limit"] == 6
    assert cfg["ingest"]["segmentation"] == "adaptive"
    (tmp_path / "default.yaml").write_text((BUILDS / "default.yaml").read_text(encoding="utf-8"), newline="\n")
    child = _write(tmp_path, "c.yaml", "extends: default.yaml\ningest:\n  concurrency: 4\n")
    assert bc.resolve_yaml(child)["ingest"]["concurrency"] == 4


def test_unknown_key_rejected_with_path(tmp_path):
    (tmp_path / "default.yaml").write_text((BUILDS / "default.yaml").read_text(encoding="utf-8"), newline="\n")
    child = _write(tmp_path, "c.yaml", "extends: default.yaml\ningest:\n  bogus: 1\n")
    with pytest.raises(ValueError, match=r"ingest\.bogus"):
        bc.resolve_yaml(child)
    child = _write(tmp_path, "d.yaml", "extends: default.yaml\nnope:\n  a: 1\n")
    with pytest.raises(ValueError, match="nope"):
        bc.resolve_yaml(child)


def test_missing_key_rejected(tmp_path):
    child = _write(tmp_path, "c.yaml", "dataset:\n  limit: 3\n")
    with pytest.raises(ValueError, match=r"missing key: dataset\.dataset"):
        bc.resolve_yaml(child)


def test_type_validation(tmp_path):
    (tmp_path / "default.yaml").write_text((BUILDS / "default.yaml").read_text(encoding="utf-8"), newline="\n")
    for body, path in (
        ("ingest:\n  concurrency: '4'\n", "ingest.concurrency"),
        ("ingest:\n  concurrency: true\n", "ingest.concurrency"),
        ("ingest:\n  store_only_context: 1\n", "ingest.store_only_context"),
        ("ingest:\n  segmentation: null\n", "ingest.segmentation"),
        ("ingest:\n  writer_max_usd: abc\n", "ingest.writer_max_usd"),
    ):
        child = _write(tmp_path, "c.yaml", "extends: default.yaml\n" + body)
        with pytest.raises(ValueError, match=re.escape(path)):
            bc.resolve_yaml(child)


def test_env_string_coercion(default_yaml):
    cfg = bc.resolve_yaml(default_yaml)
    cfg["ingest"]["writer_max_usd"] = 2.5
    sets, unset = bc.env_assignments(cfg, {}, {})
    assert sets["LME_STORE_ONLY_CONTEXT"] == "1"
    assert sets["LME_ASSISTANT_STORE_ONLY"] == "0"
    assert sets["LME_INGEST_CONCURRENCY"] == "30"
    assert sets["LME_BOLT"] == "7689"
    assert sets["LME_WRITER_MAX_USD"] == "2.5"
    assert sets["LME_SCALAR_THRESHOLD"] == "2/3"
    assert "LME_FIXTURE_PATH" in unset and "LME_EXTRACT_API_KEY" in unset


def test_secret_read_from_api_key_env_and_excluded(tmp_path, default_yaml):
    cfg = bc.resolve_yaml(default_yaml)
    cfg["extraction"]["api_key_env"] = "MY_KEY_VAR"
    environ = {"MY_KEY_VAR": "sk-secret-value"}
    sets, _ = bc.env_assignments(cfg, environ, {})
    assert sets["LME_EXTRACT_API_KEY"] == "sk-secret-value"
    rec = bc.resolved_record(default_yaml, cfg, {})
    assert "sk-secret-value" not in yaml.safe_dump(rec)
    other = {s: dict(v) for s, v in cfg.items()}
    other["extraction"]["api_key_env"] = "OTHER_VAR"
    assert bc.build_id("x", cfg, {}) == bc.build_id("x", other, {})
    with pytest.raises(KeyError):
        bc.env_assignments(cfg, {}, {})


def test_env_override_applied_warned_recorded_and_changes_id(default_yaml, caplog):
    cfg = bc.resolve_yaml(default_yaml)
    new, overrides, warnings = bc.apply_env_overrides(
        cfg, {"LME_INGEST_CONCURRENCY": "5", "LME_STORE_ONLY_CONTEXT": "0", "LME_EXTRACT_API_KEY": "sk-zzz"}
    )
    assert new["ingest"]["concurrency"] == 5
    assert new["ingest"]["store_only_context"] is False
    assert overrides == {
        "LME_EXTRACT_API_KEY": "<redacted>",
        "LME_INGEST_CONCURRENCY": 5,
        "LME_STORE_ONLY_CONTEXT": False,
    }
    assert len(warnings) == 3
    assert all("sk-zzz" not in w for w in warnings)
    assert any("LME_INGEST_CONCURRENCY" in w for w in warnings)
    assert bc.build_id("d", new, overrides) != bc.build_id("d", cfg, {})
    sets, _ = bc.env_assignments(new, {"LME_EXTRACT_API_KEY": "sk-zzz"}, overrides)
    assert sets["LME_EXTRACT_API_KEY"] == "sk-zzz"
    assert "sk-zzz" not in yaml.safe_dump(bc.resolved_record(default_yaml, new, overrides))


def test_child_env_unsets_mapped_and_passes_unmapped():
    environ = {"LME_INGEST_CONCURRENCY": "5", "LME_RESUME": "1", "LME_ANSWER_MODEL": "x", "PATH": "p"}
    env = bc.child_env(environ, {"LME_INGEST_CONCURRENCY": "30"}, ["LME_FIXTURE_PATH"])
    assert env["LME_INGEST_CONCURRENCY"] == "30"
    assert env["LME_RESUME"] == "1"
    assert env["LME_ANSWER_MODEL"] == "x"
    assert env["PATH"] == "p"


def test_build_id_stable_across_key_order_and_format(tmp_path, default_yaml):
    cfg = bc.resolve_yaml(default_yaml)
    reordered = {sec: dict(reversed(list(v.items()))) for sec, v in reversed(list(cfg.items()))}
    bid = bc.build_id("default", cfg, {})
    assert bid == bc.build_id("default", reordered, {})
    assert re.fullmatch(r"default-[0-9a-f]{10}", bid)
    assert bc.build_id("default", bc.resolve_yaml(BUILDS / "per-type-1-acceptance.yaml"), {}) != bid


class _Run:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, env=None, cwd=None):
        self.calls.append((cmd, env, cwd))

        class R:
            returncode = 7

        return R()


def _run_main(monkeypatch, argv, environ):
    runner = _Run()
    monkeypatch.setattr(build_cli.subprocess, "run", runner)
    monkeypatch.setattr(build_cli.shutil, "which", lambda name: "/usr/bin/bash")
    rc = build_cli.main(argv, environ)
    return rc, runner


def test_run_control_flags_set_env_without_changing_build_id(monkeypatch, tmp_path):
    base = ["--build", str(BUILDS / "default.yaml"), "--results-dir", str(tmp_path / "r")]
    rc, run1 = _run_main(monkeypatch, base, {"PATH": "p"})
    assert rc == 7
    rid1 = yaml.safe_load((tmp_path / "r" / "build.resolved.yaml").read_text(encoding="utf-8"))["build_id"]
    cmd, env, _ = run1.calls[0]
    assert cmd[-1] == "30" and cmd[-2].endswith("scripts/longmemeval/build_graph.sh")
    assert env["LME_RESULTS_DIR"] == (tmp_path / "r").resolve().as_posix()
    assert "LME_RESUME" not in env

    flags = base + ["--resume", "--reset-file", "ns.txt", "--stop-after-items", "2"]
    _, run2 = _run_main(monkeypatch, flags, {"PATH": "p"})
    _, env2, _ = run2.calls[0]
    assert env2["LME_RESUME"] == "1"
    assert env2["LME_RESET_FILE"] == "ns.txt"
    assert env2["LME_INGEST_STOP_AFTER_ITEMS"] == "2"
    rid2 = yaml.safe_load((tmp_path / "r" / "build.resolved.yaml").read_text(encoding="utf-8"))["build_id"]
    assert rid1 == rid2

    _, run3 = _run_main(monkeypatch, base + ["--fresh"], {"PATH": "p"})
    assert run3.calls[0][1]["LME_REQUIRE_FRESH"] == "1"


def test_non_dry_run_requires_results_dir(monkeypatch, capsys):
    rc, run = _run_main(monkeypatch, ["--build", str(BUILDS / "default.yaml")], {})
    assert rc == 2 and not run.calls
    assert "--results-dir" in capsys.readouterr().err


def test_dry_run_masks_secret_and_runs_nothing(monkeypatch, capsys, tmp_path):
    cfgfile = _write(tmp_path, "k.yaml", "extends: default.yaml\nextraction:\n  api_key_env: MY_KEY_VAR\n")
    (tmp_path / "default.yaml").write_text((BUILDS / "default.yaml").read_text(encoding="utf-8"), newline="\n")
    rc, run = _run_main(monkeypatch, ["--build", str(cfgfile), "--dry-run"], {"MY_KEY_VAR": "sk-secret-value"})
    out = capsys.readouterr().out
    assert rc == 0 and not run.calls
    assert "sk-secret-value" not in out
    assert "LME_EXTRACT_API_KEY=***" in out
    assert "build_id: k-" in out
    assert "LME_STORE_ONLY_CONTEXT=1" in out
