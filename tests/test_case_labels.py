"""The case labeller (ml#3): which scenario, if any, produced each case of a capture."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from helpers import write_jsonl
from siblings import needs, sibling

from synthaea_ml.cases import label as lab
from synthaea_ml.data.process_tree import descendants_in, process_edges

S = 1_000_000_000
RUNNER = 100


def _event(pid: int, ppid: int, ts: int, gen: int, parent_gen: int) -> dict:
    meta = {
        "pid": pid,
        "ppid": ppid,
        "timestamp_ns": ts,
        "comm": "x",
        "process_generation": gen,
        "parent_process_generation": parent_gen,
    }
    return {"type": "exec", "meta": meta}


def _run(**overrides: Any) -> dict:
    doc: dict[str, Any] = {
        "root_pid": RUNNER,
        "start_ns": 10 * S,
        "end_ns": 100 * S,
        "ran": ["dropper-chain", "odd-parents/webapp-spawns-shell"],
        "failed": [],
        "scenarios": [
            {
                "name": "dropper-chain",
                "pid": 110,
                "start_ns": 11 * S,
                "end_ns": 20 * S,
                "exit_code": 0,
            },
            {
                "name": "odd-parents/webapp-spawns-shell",
                "pid": 120,
                "start_ns": 30 * S,
                "end_ns": 40 * S,
                "exit_code": 0,
            },
        ],
    }
    doc.update(overrides)
    return doc


# The runner (100) runs two scenarios under their `timeout`s (110, 120) and its own `sleep`
# (130); cron (200) runs beside it; 111 is a beacon dropper-chain left running past its end.
EVENTS = [
    _event(RUNNER, 1, 9 * S, 1, 0),
    _event(110, RUNNER, 11 * S, 2, 1),
    _event(111, 110, 12 * S, 3, 2),
    _event(112, 111, 50 * S, 4, 3),  # the beacon's child, after the scenario's window
    _event(120, RUNNER, 30 * S, 5, 1),
    _event(121, 120, 31 * S, 6, 5),
    _event(130, RUNNER, 21 * S, 7, 1),
    _event(200, 1, 15 * S, 8, 0),
]


def _case(case_id: str, ppid: int, parent_gen: int | None, *techniques: str) -> dict:
    return {
        "case_id": case_id,
        "ppid": ppid,
        "comm": "x",
        "parent_generation": parent_gen,
        "severity": "high",
        "score": None,
        "techniques": list(techniques),
        "sources": [],
        "detections": [],
    }


@pytest.fixture
def scenario_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir()
    (directory / "dropper-chain.yaml").write_text(
        "name: dropper-chain\n"
        "expected_detections:\n"
        '  - technique: "T1105/T1059/T1071"\n'
        "    rule: rule_spawn_connect_filewrite\n"
        "    min_count: 1\n"
        "    tolerance: 0\n",
        encoding="utf-8",
    )
    return directory


def _label(
    tmp_path: Path, scenario_dir: Path, cases: list[dict], events_rows: list[dict] | None = None
) -> dict[str, lab.CaseLabel]:
    events = write_jsonl(tmp_path / "events.jsonl", EVENTS if events_rows is None else events_rows)
    cases_path = write_jsonl(tmp_path / "cases.jsonl", cases)
    labels = lab.label_cases(lab.parse_run(_run()), events, cases_path, scenario_dir)
    return {label.case_id: label for label in labels}


def test_a_case_under_a_scenario_that_reached_its_sidecar_technique_is_a_reaching_positive(
    tmp_path: Path, scenario_dir: Path
) -> None:
    got = _label(tmp_path, scenario_dir, [_case("c", 111, 3, "T1105", "T1105/T1059/T1071")])["c"]
    assert got.label == lab.SCENARIO
    assert got.scenario == "dropper-chain"
    assert got.expected_techniques == ("T1105/T1059/T1071",)
    assert got.reached_expected is True


def test_a_scenario_case_that_reached_only_part_of_a_combined_technique_did_not_reach_it(
    tmp_path: Path, scenario_dir: Path
) -> None:
    """Exact string match, as `compute_passed`: `T1105` alone is not the sidecar's combined
    `T1105/T1059/T1071`."""
    got = _label(tmp_path, scenario_dir, [_case("c", 110, 2, "T1105")])["c"]
    assert got.label == lab.SCENARIO
    assert got.reached_expected is False


def test_a_child_the_scenario_left_running_after_its_window_is_still_the_scenarios(
    tmp_path: Path, scenario_dir: Path
) -> None:
    got = _label(tmp_path, scenario_dir, [_case("c", 112, 4)])["c"]
    assert got.label == lab.SCENARIO
    assert got.scenario == "dropper-chain"


def test_a_scenario_without_a_sidecar_carries_no_expectation_rather_than_an_empty_one(
    tmp_path: Path, scenario_dir: Path
) -> None:
    got = _label(tmp_path, scenario_dir, [_case("c", 121, 6, "T1059")])["c"]
    assert got.label == lab.SCENARIO
    assert got.scenario == "odd-parents/webapp-spawns-shell"
    assert got.expected_techniques is None
    assert got.reached_expected is None


def test_the_runners_own_processes_are_harness_and_the_hosts_are_background(
    tmp_path: Path, scenario_dir: Path
) -> None:
    got = _label(
        tmp_path,
        scenario_dir,
        [_case("sleep", 130, 7), _case("runner", RUNNER, 1), _case("cron", 200, 8)],
    )
    assert got["sleep"].label == lab.HARNESS
    assert got["runner"].label == lab.HARNESS
    assert got["cron"].label == lab.BACKGROUND
    assert all(label.scenario is None for label in got.values())


def test_a_recycled_pid_of_a_scenario_process_is_not_the_scenarios(
    tmp_path: Path, scenario_dir: Path
) -> None:
    """Same pid as the dropper-chain's child, another generation: another process."""
    recycled = _event(111, 1, 60 * S, 99, 0)
    got = _label(
        tmp_path, scenario_dir, [_case("c", 111, 99, "T1105/T1059/T1071")], [*EVENTS, recycled]
    )["c"]
    assert got.label == lab.BACKGROUND
    assert got.reached_expected is None


def test_a_case_whose_parent_never_appears_in_the_capture_is_unattributed_not_background(
    tmp_path: Path, scenario_dir: Path
) -> None:
    """First Debian campaign (2026-11-16): `bind-shell` runs `nc` from a subshell that only
    forks, so the subshell emits no event and `nc`'s case hung under a parent the capture
    never saw. It was labelled `background`, a scenario's attack counted as host activity."""
    nc = _event(141, 140, 13 * S, 10, 9)  # the subshell 140 (generation 9) is silent
    got = _label(tmp_path, scenario_dir, [_case("nc", 140, 9, "T1059/T1071")], [*EVENTS, nc])
    assert got["nc"].label == lab.UNATTRIBUTED
    assert got["nc"].scenario is None


def test_a_scenario_root_exec_d_long_before_its_recorded_start_is_not_its_root(
    tmp_path: Path,
) -> None:
    """A scenario's root is forked after its start is taken, so it gets a one-second lead,
    not the runner's thirty: an older process holding the pid is not the scenario."""
    events = write_jsonl(
        tmp_path / "events.jsonl", [_event(110, 1, 5 * S, 2, 0), _event(111, 110, 12 * S, 3, 2)]
    )
    tree = descendants_in(
        process_edges(events), 110, (11 * S, 20 * S), root_lead_ns=lab.SCENARIO_ROOT_LEAD_NS
    )
    assert tree == set()


def test_the_summary_counts_cases_and_reaching_cases_per_scenario(
    tmp_path: Path, scenario_dir: Path
) -> None:
    labels = _label(
        tmp_path,
        scenario_dir,
        [
            _case("a", 111, 3, "T1105/T1059/T1071"),
            _case("b", 110, 2, "T1105"),
            _case("c", 200, 8),
        ],
    )
    summary = lab.summarize(list(labels.values()))
    assert summary["by_label"] == {lab.SCENARIO: 2, lab.BACKGROUND: 1}
    assert summary["by_scenario"] == {"dropper-chain": {"cases": 2, "reached_expected": 1}}


# ── run.json and cases.jsonl the labeller refuses ───────────────────────────────────────


def test_a_run_json_from_before_per_scenario_windows_is_refused_with_the_fix() -> None:
    doc = _run()
    del doc["scenarios"]
    with pytest.raises(ValueError, match="re-run malicious-run.sh"):
        lab.parse_run(doc)


@pytest.mark.parametrize(
    "broken",
    [
        _run(root_pid=True),
        _run(start_ns="10"),
        _run(scenarios=[{"pid": 1, "start_ns": 11 * S, "end_ns": 12 * S, "exit_code": 0}]),
        _run(
            scenarios=[{"name": "x", "pid": 1, "start_ns": 5 * S, "end_ns": 12 * S, "exit_code": 0}]
        ),
        _run(
            scenarios=[
                {"name": "x", "pid": 1, "start_ns": 11 * S, "end_ns": 200 * S, "exit_code": 0}
            ]
        ),
        _run(scenarios=["x"]),
    ],
    ids=["bool pid", "string ts", "no name", "starts early", "ends late", "not an object"],
)
def test_a_run_json_the_labeller_cannot_trust_is_refused(broken: dict) -> None:
    with pytest.raises(ValueError):
        lab.parse_run(broken)


@pytest.mark.parametrize(
    "doc",
    [
        None,
        0,
        1.5,
        "x",
        [],
        [1, 2],
        {},
        {"scenarios": None},
        {"root_pid": {}},
        {"root_pid": 1, "start_ns": 2, "end_ns": 1, "scenarios": [{}]},
    ],
)
def test_parse_run_raises_only_value_error_on_any_json_value(doc: Any) -> None:
    with pytest.raises(ValueError):
        lab.parse_run(doc)


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"ppid": 1, "techniques": []}',
        '{"case_id": "c", "ppid": "1", "techniques": []}',
        '{"case_id": "c", "ppid": 1, "parent_generation": "2", "techniques": []}',
        '{"case_id": "c", "ppid": 1, "techniques": "T1059"}',
        '{"case_id": "c", "ppid": 1, "techniques": [1]}',
    ],
)
def test_a_cases_line_that_is_not_a_case_is_refused_with_its_line_number(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case("ok", 1, None)) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases\.jsonl:2"):
        list(lab.iter_cases(path))


# ── Command line ────────────────────────────────────────────────────────────────────────


def test_the_command_line_writes_one_label_per_case(tmp_path: Path, scenario_dir: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_run()), encoding="utf-8")
    write_jsonl(tmp_path / "events.jsonl", EVENTS)
    write_jsonl(tmp_path / "cases.jsonl", [_case("a", 111, 3), _case("b", 200, 8)])
    status = lab.main(
        [
            "--run-json",
            str(tmp_path / "run.json"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--cases",
            str(tmp_path / "cases.jsonl"),
            "--scenario-dir",
            str(scenario_dir),
            "--output",
            str(tmp_path / "labels.jsonl"),
        ]
    )
    assert status == 0
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert [(r["case_id"], r["label"]) for r in rows] == [
        ("a", lab.SCENARIO),
        ("b", lab.BACKGROUND),
    ]


def test_the_command_line_fails_on_a_bad_run_json_without_a_traceback(
    tmp_path: Path, scenario_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    old_run = _run()
    del old_run["scenarios"]
    (tmp_path / "run.json").write_text(json.dumps(old_run), encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        status = lab.main(
            [
                "--run-json",
                str(tmp_path / "run.json"),
                "--events",
                str(tmp_path / "events.jsonl"),
                "--cases",
                str(tmp_path / "cases.jsonl"),
                "--scenario-dir",
                str(scenario_dir),
                "--output",
                str(tmp_path / "labels.jsonl"),
            ]
        )
    assert status == 1
    assert "re-run malicious-run.sh" in caplog.text
    assert not (tmp_path / "labels.jsonl").exists()


# ── The real sidecars ───────────────────────────────────────────────────────────────────


@needs("scenarios", "scenarios", "dropper-chain.yaml")
def test_every_scenario_malicious_run_sh_runs_has_its_techniques_read_from_its_sidecar() -> None:
    """The names `malicious-run.sh` records are the sidecars' file stems: every non
    odd-parents scenario it runs must resolve to expected techniques."""
    script = sibling("scenarios", "ml-capture", "malicious-run.sh").read_text(encoding="utf-8")
    line = next(row for row in script.splitlines() if row.startswith("SCENARIOS=("))
    names = line[len("SCENARIOS=(") : line.index(")")].split()
    expected = lab.expected_techniques(sibling("scenarios", "scenarios"), names)
    assert set(expected) == set(names)
    assert all(expected[name] for name in names)
