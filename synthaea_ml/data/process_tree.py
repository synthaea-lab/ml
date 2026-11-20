"""Process trees read back from a wire capture (`events.jsonl`).

A capture of lab scenarios also holds the host's own processes (the agent, SSH, cron). What
tells a scenario's processes apart is ancestry: every event carries its process identity and
its parent's in `meta`, so the tree under a known root pid can be rebuilt from the capture
alone. `t1_lineage_gain` uses it for the scenario runner's whole tree, the case labeller
(ml#3) for each scenario's own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from synthaea_ml.units import NS_PER_SECOND

logger = logging.getLogger(__name__)

#: A process identity: `(pid, process_generation)`. The generation tells a recycled pid's
#: lives apart; it is `None` when the sensor did not stamp one.
ProcessKey = tuple[int, int | None]

#: `ProcessKey -> (ppid, parent_generation, first_ts_ns)`.
ProcessEdges = dict[ProcessKey, tuple[int, int | None, int]]

# A runner records its start time once it is running, so its own exec precedes the window:
# the root may have been first seen this long before `start_ns`.
ROOT_LEAD_NS = 30 * NS_PER_SECOND


def process_edges(path: Path) -> ProcessEdges:
    """`(pid, generation) -> (ppid, parent_generation, first_ts_ns)` from every wire event.

    Every event carries its identity and its parent's in `meta`, so a process that forked
    and never exec'd (a subshell) still links its children to the runner. A line that is not
    a JSON object with a `meta` holding `pid` and `ppid` is skipped: a capture cut while the
    agent was writing ends with a torn line. A line before the last that is not a JSON object,
    and an event with no `timestamp_ns` (kept at 0, so outside every window), are counted and
    logged: either means the capture is not what the trees assume.
    """
    edges: ProcessEdges = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    not_json = 0
    without_timestamp = 0
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if not isinstance(event, dict):
            if line.strip() and number < len(lines):
                not_json += 1
            continue
        meta = event.get("meta")
        if not isinstance(meta, dict) or "pid" not in meta or "ppid" not in meta:
            continue
        key = (meta["pid"], meta.get("process_generation"))
        if "timestamp_ns" not in meta:
            without_timestamp += 1
        ts = meta.get("timestamp_ns", 0)
        if key not in edges or ts < edges[key][2]:
            edges[key] = (meta["ppid"], meta.get("parent_process_generation"), ts)
    if not_json:
        logger.warning(
            "%s: skipped %d line(s) before the last that are not JSON objects", path, not_json
        )
    if without_timestamp:
        logger.warning(
            "%s: %d event(s) have no meta timestamp_ns; their processes fall outside every window",
            path,
            without_timestamp,
        )
    return edges


def descendants_in(
    edges: ProcessEdges,
    root_pid: int,
    window: tuple[int, int],
    root_lead_ns: int = ROOT_LEAD_NS,
    descendant_end_ns: int | None = None,
) -> set[ProcessKey]:
    """The `(pid, generation)` of `root_pid`'s process tree in `edges`, the root included.

    With `window = (start_ns, end_ns)`, the root is the incarnation of `root_pid` first seen in
    `[start_ns - root_lead_ns, end_ns]` (a runner's exec precedes the start time it records;
    an earlier life of a recycled pid is another process); a descendant is a process first seen
    in `[start_ns, descendant_end_ns]` whose parent incarnation is already in the set.
    `descendant_end_ns` defaults to `end_ns`; a later bound keeps a child the root left running
    (a backgrounded beacon) in its tree.
    """
    start_ns, end_ns = window
    last_ns = end_ns if descendant_end_ns is None else descendant_end_ns
    members: set[ProcessKey] = {
        key
        for key, (_, _, ts) in edges.items()
        if key[0] == root_pid and start_ns - root_lead_ns <= ts <= end_ns
    }
    changed = True
    while changed:
        changed = False
        for key, (ppid, parent_generation, ts) in edges.items():
            if key in members or not start_ns <= ts <= last_ns:
                continue
            if (ppid, parent_generation) in members:
                members.add(key)
                changed = True
    return members


def descendants_of(
    path: Path,
    root_pid: int,
    start_ns: int,
    end_ns: int,
    root_lead_ns: int = ROOT_LEAD_NS,
) -> set[ProcessKey]:
    """`descendants_in` over the capture at `path`, with descendants bounded by `end_ns`."""
    return descendants_in(process_edges(path), root_pid, (start_ns, end_ns), root_lead_ns)
