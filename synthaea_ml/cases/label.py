"""Labels the cases of a scenario capture by the scenario that produced them (ml#3).

The T2 scorer learns from cases, not traces: `agent replay` turns a capture into one case per
entity, the fused verdict of every engine for a `(ppid, comm)`. A capture of the lab scenarios
also holds the host's own activity, so a case is a positive only if it comes from a
scenario's process tree. This module decides that from three inputs of the same run:

- `run.json`, from `scenarios/ml-capture/malicious-run.sh`: the runner's pid and window, and
  each scenario's own root pid and window;
- the capture (`events.jsonl`) the cases were replayed from, for the process ancestry;
- `cases.jsonl` from `agent replay`.

Each case gets one of four labels:

- `scenario`: its entity's parent process descends from one scenario's root. It carries that
  scenario's name and, when the scenario has a sidecar, whether the case reached any of the
  sidecar's expected techniques. Techniques match as exact strings, but the rule is any-of,
  unlike the scenario replay's `compute_passed` (all of the expected `(technique, rule)` pairs,
  with counts): a case is one entity's verdict, and a scenario's techniques are often spread
  over several of its cases, so all-of per case would call most of them misses;
- `harness`: the parent descends from the runner but from no scenario (the runner's own
  `sleep`, `date`, `python3`);
- `unattributed`: the parent never appears in the capture, so its ancestry is unknown. A
  subshell that only forks emits no event: on the first Debian campaign (2026-11-16) the `nc`
  of `bind-shell` hung under such a parent. A window-based guess would also catch a cron job
  whose daemon parent was silent, so these are left out rather than guessed;
- `background`: anything else (the agent, SSH, cron).

The ancestry `label` is the training target; `reached_expected` is a coverage diagnostic only.
It is computed from the case's own `techniques`, so a trainer that used it as a target over
technique-derived features would learn the label from its inputs.

What it does not do: decide what a negative is for training. `background` cases are host
activity during an attack run, not a benign corpus (ml#2), and `scenario` cases that reached
no expected technique are still the scenario's: whether to train on them is the trainer's call.

Usage:
    python -m synthaea_ml.cases.label --run-json out/run.json \\
        --events events-malicious.jsonl --cases cases.jsonl --output labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from synthaea_ml import siblings
from synthaea_ml.cli_logging import configure_cli_logging
from synthaea_ml.data.process_tree import ProcessKey, descendants_in, process_edges
from synthaea_ml.evaluation.scenario_replay import load_expected_detections

logger = logging.getLogger(__name__)


class Label(StrEnum):
    """Where a case comes from; serialises as its plain string value."""

    SCENARIO = "scenario"
    HARNESS = "harness"
    UNATTRIBUTED = "unattributed"
    BACKGROUND = "background"


SCENARIO = Label.SCENARIO
HARNESS = Label.HARNESS
UNATTRIBUTED = Label.UNATTRIBUTED
BACKGROUND = Label.BACKGROUND

# Scenarios that exist only for the capture (odd parents for T1) have no sidecar by design.
ODD_PARENTS_PREFIX = "odd-parents/"

# A scenario's root (the `timeout` it runs under) is forked after its recorded start, so it
# needs no lead beyond clock granularity between `date` and the sensor's timestamp.
SCENARIO_ROOT_LEAD_NS = 1_000_000_000


@dataclass(frozen=True)
class ScenarioRun:
    """One scenario of a run, as `run.json` records it."""

    name: str
    pid: int
    start_ns: int
    end_ns: int
    exit_code: int


@dataclass(frozen=True)
class RunRecord:
    """A `run.json`: the runner's pid and window, and its scenarios in run order."""

    root_pid: int
    start_ns: int
    end_ns: int
    scenarios: tuple[ScenarioRun, ...]


@dataclass(frozen=True)
class CaseLabel:
    """One case's label. `expected_techniques` and `reached_expected` are `None` when the
    case is not a scenario's or its scenario has no sidecar."""

    case_id: str
    label: Label
    scenario: str | None
    techniques: tuple[str, ...]
    expected_techniques: tuple[str, ...] | None
    reached_expected: bool | None


def _int_field(record: Mapping[str, Any], key: str, where: str) -> int:
    value = record.get(key)
    # bool is an int subclass; a `true` pid is a broken file, not pid 1.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where}: `{key}` must be an integer, got {value!r}")
    return value


def parse_run(doc: Any) -> RunRecord:
    """The `RunRecord` in a parsed `run.json`.

    Raises:
        ValueError: If a field is missing or mistyped, a scenario's window is outside the
            run's, or two scenarios share a name (each script runs once per run, and a label
            names its scenario): a run.json the labeller cannot trust labels nothing.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"run.json must be an object, got {type(doc).__name__}")
    start_ns = _int_field(doc, "start_ns", "run.json")
    end_ns = _int_field(doc, "end_ns", "run.json")
    entries = doc.get("scenarios")
    if not isinstance(entries, list):
        raise ValueError(
            "run.json has no `scenarios` list: it predates per-scenario windows "
            "(scenarios#29); re-run malicious-run.sh"
        )
    scenarios: list[ScenarioRun] = []
    for index, entry in enumerate(entries):
        where = f"run.json scenarios[{index}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"{where}: must be an object with a string `name`")
        scenario = ScenarioRun(
            name=entry["name"],
            pid=_int_field(entry, "pid", where),
            start_ns=_int_field(entry, "start_ns", where),
            end_ns=_int_field(entry, "end_ns", where),
            exit_code=_int_field(entry, "exit_code", where),
        )
        if not start_ns <= scenario.start_ns <= scenario.end_ns <= end_ns:
            raise ValueError(f"{where}: window is not inside the run's")
        if any(earlier.name == scenario.name for earlier in scenarios):
            raise ValueError(f"{where}: scenario {scenario.name!r} is listed twice")
        scenarios.append(scenario)
    return RunRecord(
        root_pid=_int_field(doc, "root_pid", "run.json"),
        start_ns=start_ns,
        end_ns=end_ns,
        scenarios=tuple(scenarios),
    )


def load_run(path: Path) -> RunRecord:
    """The `RunRecord` in the `run.json` at `path`.

    Raises:
        ValueError: If the file is not JSON or not a valid run record.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path}: not JSON: {err}") from err
    return parse_run(doc)


@dataclass(frozen=True)
class RunTrees:
    """The process trees of a run: each scenario's, the runner's whole tree, and every
    process the capture saw at all."""

    by_scenario: dict[str, set[ProcessKey]]
    runner: set[ProcessKey]
    seen: frozenset[ProcessKey]


def run_trees(events: Path, run: RunRecord) -> RunTrees:
    """Each scenario's process tree and the runner's, from the capture at `events`.

    A scenario's descendants are bounded by the run's end, not the scenario's: a child it left
    running (a backgrounded beacon) is still its own. A capture without process generations
    keys a process by its pid alone, so a recycled pid can join a tree; the 1 s root lead is
    then the only guard, and a process in two trees is logged (its cases go to the first
    scenario in run order).
    """
    if not run.scenarios:
        logger.warning("run.json lists no scenario: no case can be labelled `%s`", SCENARIO)
    edges = process_edges(events)
    by_scenario = {
        scenario.name: descendants_in(
            edges,
            scenario.pid,
            (scenario.start_ns, scenario.end_ns),
            root_lead_ns=SCENARIO_ROOT_LEAD_NS,
            descendant_end_ns=run.end_ns,
        )
        for scenario in run.scenarios
    }
    _log_shared_processes(by_scenario)
    runner = descendants_in(edges, run.root_pid, (run.start_ns, run.end_ns))
    return RunTrees(by_scenario=by_scenario, runner=runner, seen=frozenset(edges))


def _log_shared_processes(by_scenario: Mapping[str, set[ProcessKey]]) -> None:
    names = list(by_scenario)
    for index, first in enumerate(names):
        for later in names[index + 1 :]:
            shared = by_scenario[first] & by_scenario[later]
            if shared:
                logger.warning(
                    "scenarios %s and %s share %d process(es); their cases go to %s",
                    first,
                    later,
                    len(shared),
                    first,
                )


def expected_techniques(scenario_dir: Path, names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """The sidecar's expected techniques of each scenario in `names`.

    `odd-parents/*` scenarios, which exist only for the capture, have no sidecar: they are left
    out, so their cases carry no expected techniques rather than an empty list that reads as
    "expected nothing". Every other scenario must have one.

    Raises:
        FileNotFoundError: If a scenario outside `odd-parents/` has no sidecar: a wrong
            `--scenario-dir` or a missing scenarios checkout, which would otherwise leave every
            case without its expectation and still exit 0.
        ValueError: If a sidecar is malformed, or its `name:` is not its file stem (the label
            names a scenario by the stem `malicious-run.sh` records).
    """
    expected: dict[str, tuple[str, ...]] = {}
    for name in names:
        if name.startswith(ODD_PARENTS_PREFIX):
            continue
        sidecar = scenario_dir / f"{name}.yaml"
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"{sidecar}: no sidecar for scenario {name!r} (wrong --scenario-dir, or the "
                "scenarios checkout is missing)"
            )
        try:
            sidecar_name, detections = load_expected_detections(sidecar)
        except (KeyError, TypeError, yaml.YAMLError) as err:
            raise ValueError(f"{sidecar}: malformed sidecar: {err!r}") from err
        if sidecar_name != name:
            raise ValueError(f"{sidecar}: `name: {sidecar_name}` is not its file stem {name!r}")
        expected[name] = tuple(d.technique for d in detections)
    return expected


def iter_cases(path: Path) -> Iterator[dict[str, Any]]:
    """The case objects in the `cases.jsonl` at `path`.

    Raises:
        ValueError: If a line is not a JSON object with a string `case_id`, an integer `ppid`
            and a list `techniques`: `agent replay` fails rather than write a torn case, so a
            bad line is a wrong file.
    """
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            case = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{where}: not JSON: {err}") from err
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ValueError(f"{where}: not a case object")
        _int_field(case, "ppid", where)
        generation = case.get("parent_generation")
        if generation is not None:
            _int_field(case, "parent_generation", where)
        techniques = case.get("techniques")
        if not isinstance(techniques, list) or not all(isinstance(t, str) for t in techniques):
            raise ValueError(f"{where}: `techniques` must be a list of strings")
        yield case


def label_case(
    case: Mapping[str, Any],
    trees: RunTrees,
    expected: Mapping[str, tuple[str, ...]],
) -> CaseLabel:
    """The label of one case of the run whose trees are `trees`."""
    parent: ProcessKey = (case["ppid"], case.get("parent_generation"))
    techniques = tuple(case["techniques"])
    for name, tree in trees.by_scenario.items():
        if parent in tree:
            wanted = expected.get(name)
            return CaseLabel(
                case_id=case["case_id"],
                label=SCENARIO,
                scenario=name,
                techniques=techniques,
                expected_techniques=wanted,
                reached_expected=None if wanted is None else bool(set(wanted) & set(techniques)),
            )
    label: Label
    if parent in trees.runner:
        label = HARNESS
    elif parent not in trees.seen:
        label = UNATTRIBUTED
    else:
        label = BACKGROUND
    return CaseLabel(
        case_id=case["case_id"],
        label=label,
        scenario=None,
        techniques=techniques,
        expected_techniques=None,
        reached_expected=None,
    )


def label_cases(run: RunRecord, events: Path, cases: Path, scenario_dir: Path) -> list[CaseLabel]:
    """Every case in `cases`, labelled against `run` and the capture at `events`."""
    trees = run_trees(events, run)
    expected = expected_techniques(scenario_dir, [s.name for s in run.scenarios])
    return [label_case(case, trees, expected) for case in iter_cases(cases)]


def summarize(labels: Sequence[CaseLabel]) -> dict[str, Any]:
    """Counts by label, and per scenario how many cases it produced and how many reached an
    expected technique: diversity per scenario, not the total, is what limits the dataset."""
    per_scenario: dict[str, dict[str, int]] = {}
    for label in labels:
        if label.scenario is None:
            continue
        counts = per_scenario.setdefault(label.scenario, {"cases": 0, "reached_expected": 0})
        counts["cases"] += 1
        counts["reached_expected"] += int(bool(label.reached_expected))
    return {
        "by_label": dict(Counter(label.label for label in labels)),
        "by_scenario": per_scenario,
    }


@dataclass(frozen=True)
class LabelArgs:
    """The parsed command line of a labelling run."""

    run_json: Path
    events: Path
    cases: Path
    scenario_dir: Path
    output: Path


def parse_args(argv: Sequence[str] | None = None) -> LabelArgs:
    """The run's options from `argv` (the process's arguments when `None`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=siblings.scenario_sidecars(),
        help="the scenario sidecars (default: the scenarios sibling checkout)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return LabelArgs(
        run_json=args.run_json,
        events=args.events,
        cases=args.cases,
        scenario_dir=args.scenario_dir,
        output=args.output,
    )


def label_run(args: LabelArgs) -> dict[str, Any]:
    """Label the cases, write one label per line to `args.output`, and return the summary."""
    labels = label_cases(load_run(args.run_json), args.events, args.cases, args.scenario_dir)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for label in labels:
            out.write(json.dumps(asdict(label)) + "\n")
    summary = summarize(labels)
    logger.info("labelled %d cases: %s", len(labels), json.dumps(summary, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Label a run's cases from the command line `argv`; returns the exit status."""
    args = parse_args(argv)
    configure_cli_logging()
    try:
        label_run(args)
    except (ValueError, OSError) as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
