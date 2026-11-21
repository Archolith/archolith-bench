"""build CLI settings resolved in Python agree with what build_graph.sh's config.sh produced."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from archolith_bench.build import __main__ as build_cli
from archolith_bench.build import settings
from archolith_bench.build.config import apply_env_overrides, child_env, env_assignments, resolve_yaml
from archolith_bench.build.pipeline import native, read_settings

REPO = Path(__file__).resolve().parents[1]
LME = REPO / "scripts" / "longmemeval"
PIPELINE_SRC = (REPO / "archolith_bench" / "build" / "pipeline.py").read_text(encoding="utf-8")
CONFIG_SRC = (LME / "config.sh").read_text(encoding="utf-8")


def _bash() -> str:
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    return bash


def _base_env(**values: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LME_", "MENHIR_", "BENCH_", "BENCHMARK_", "LONGMEMEVAL_"))
           and k not in ("ARCH_DIR", "SCRIPT_DIR")}
    env.update(values)
    return env


def _cli_env(build: Path, environ: dict[str, str]) -> dict[str, str]:
    """The env the CLI hands config.sh/settings.resolve: YAML assignments over the caller's env."""
    config, overrides, _ = apply_env_overrides(resolve_yaml(build), environ)
    sets, unset = env_assignments(config, environ, overrides)
    return child_env(environ, sets, unset)


def _config_sh(env: dict[str, str], names: list[str]) -> dict[str, str]:
    """Source the real config.sh as build_graph.sh does, then print each name (unset -> absent)."""
    dump = "".join(f'[ "${{{n}+x}}" ] && printf "%s\\0%s\\0" {n} "${{{n}}}"\n' for n in names)
    command = f'set -eu\ncd "{REPO.as_posix()}"\nsource scripts/longmemeval/config.sh\n{dump}true\n'
    out = subprocess.run([_bash(), "-c", command], env=env, capture_output=True, check=True).stdout
    parts = out.decode("utf-8").split("\0")[:-1]
    return {parts[i]: native(parts[i + 1]) for i in range(0, len(parts), 2)}


def _exported(env: dict[str, str]) -> dict[str, str]:
    command = (f'set -eu\ncd "{REPO.as_posix()}"\nsource scripts/longmemeval/config.sh\n'
               'env -0')
    out = subprocess.run([_bash(), "-c", command], env=env, capture_output=True, check=True).stdout
    pairs = (item.split("=", 1) for item in out.decode("utf-8").split("\0") if "=" in item)
    return {k: v for k, v in pairs}


CASES = [
    ("default.yaml", {}),
    ("default.yaml", {"LME_NEO4J_NAME": "menhir-lme-smoke", "LME_RESULTS_DIR": "C:/tmp/r",
                      "MENHIR_MAIN": "C:/src/menhir", "LME_RESUME": "1", "LME_RESET_FILE": "ns.txt"}),
    ("default.yaml", {"MENHIR_CANONICAL_SELF_BINDING_MODE": "observe", "BENCH_PY": "C:/py/python.exe",
                      "LME_MANIFEST_PATH": "C:/tmp/m.json", "LONGMEMEVAL_VARIANT": "s"}),
    ("per-type-1-acceptance.yaml", {}),
]


@pytest.mark.parametrize("build, extra", CASES)
def test_settings_match_config_sh(build: str, extra: dict[str, str]):
    env = _cli_env(REPO / "builds" / build, _base_env(**extra))
    resolved, _ = settings.resolve(env)
    names = sorted(n for n in resolved if re.match(r"^(LME_|MENHIR_|BENCH_|BENCHMARK_|LONGMEMEVAL_)", n))
    shell = _config_sh(env, names)
    assert {n: resolved[n] for n in names} == {n: shell[n] for n in names if n in shell}
    assert set(names) == set(shell)


def test_exported_env_matches_config_sh():
    env = _cli_env(REPO / "builds" / "default.yaml", _base_env())
    _, exported = settings.resolve(env)
    shell = _exported(env)
    for var in settings.EXPORTED:
        assert exported[var] == shell[var]
    # config.sh adds no other name to the environment children inherit.
    assert set(shell) - set(env) - {"_", "SHLVL", "PWD", "OLDPWD"} <= set(settings.EXPORTED)


def _assigned_by_config_sh() -> set[str]:
    return set(re.findall(r"^(?:export )?([A-Z][A-Z0-9_]*)=", CONFIG_SRC, re.M))


def test_every_config_sh_name_pipeline_reads_is_resolved():
    env = _cli_env(REPO / "builds" / "default.yaml", _base_env())
    resolved, _ = settings.resolve(env)
    read = set(re.findall(r"\b((?:LME|MENHIR|BENCH)_[A-Z0-9_]+|LONGMEMEVAL_VARIANT)\b", PIPELINE_SRC))
    missing = (read & _assigned_by_config_sh()) - set(resolved)
    assert not missing


def test_stdin_pairs_round_trip_through_pipeline_reader():
    env = _cli_env(REPO / "builds" / "default.yaml", _base_env())
    resolved, exported = settings.resolve(env)

    class Stream:
        def read(self):
            return settings.stdin_pairs(resolved)

    got = read_settings(Stream(), exported)
    for name in resolved:
        if re.match(r"^(LME_|MENHIR_|BENCH_)|^LONGMEMEVAL_VARIANT$", name):
            assert got[name] == resolved[name]
    assert "ARCH_DIR" not in got


def test_cli_starts_pipeline_with_resolved_settings(monkeypatch, tmp_path):
    seen = {}

    def run(cmd, input=None, env=None, cwd=None):
        seen.update(cmd=cmd, input=input, env=env, cwd=cwd)
        return type("R", (), {"returncode": 3})()

    monkeypatch.setattr(build_cli.subprocess, "run", run)
    argv = ["--build", str(REPO / "builds" / "default.yaml"), "--results-dir", str(tmp_path)]
    assert build_cli.main(argv, environ=_base_env()) == 3
    assert seen["cmd"][1].endswith("archolith_bench/build/pipeline.py") and seen["cmd"][2] == "30"
    pairs = seen["input"].split(b"\0")
    kv = dict(zip(pairs[0:-1:2], pairs[1::2]))
    assert kv[b"LME_RESULTS_DIR"] == tmp_path.resolve().as_posix().encode()
    assert kv[b"LME_MANIFEST_PATH"] == f"{tmp_path.resolve().as_posix()}/manifest.json".encode()
    assert seen["env"]["MENHIR_OPENAI_SERVICE_TIER"] == "flex"
    assert "LME_MANIFEST_PATH" not in seen["env"]  # a shell variable, never exported
