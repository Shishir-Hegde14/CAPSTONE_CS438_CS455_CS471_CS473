#!/usr/bin/env python3
"""Check whether a paired replay run actually started with self-generated successes.

The OS protocol seeds run 3 with the most recent *correct, completed* sessions
from run 1. If run 1 produced fewer successes than REPLAY_COUNT, the replay
condition starts with fewer examples (or none) and any baseline/replay contrast
is not a valid self-experience replay test.
"""
from __future__ import annotations

import argparse
import json
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", required=True, help="run1_warmup.jsonl")
    ap.add_argument("--replay", required=True, help="run3_replay.jsonl")
    ap.add_argument("--replay-count", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    warmup = load_jsonl(args.warmup)
    replay = load_jsonl(args.replay)
    warmup_correct_records = sum(1 for r in warmup if r.get("correct"))
    replay_correct_records = sum(1 for r in replay if r.get("correct"))
    # The runner's replay seeder requires BOTH correct AND status == completed.
    warmup_successes = sum(
        1 for r in warmup if r.get("correct") and r.get("status") == "completed"
    )
    initial_examples = (
        len(replay[0].get("replay_task_indices", [])) if replay else 0
    )
    max_examples = max(
        (len(r.get("replay_task_indices", [])) for r in replay), default=0
    )
    replay_successes = sum(
        1 for r in replay if r.get("correct") and r.get("status") == "completed"
    )
    expected_initial = min(args.replay_count, warmup_successes)
    valid = args.replay_count == 0 or (
        warmup_successes >= args.replay_count and initial_examples >= expected_initial
    )
    caveat = (
        "Warm-up produced fewer than replay_count successful sessions; run 3 did "
        "not start with a full set of self-generated examples. The paired replay "
        "contrast is degenerate and must not be interpreted as a lifelong-learning effect."
        if not valid
        else "Run 3 started with the expected number of successful warm-up experiences."
    )
    data = {
        "warmup_file": args.warmup,
        "replay_file": args.replay,
        "requested_replay_count": args.replay_count,
        "warmup_sessions": len(warmup),
        "warmup_correct_records": warmup_correct_records,
        "warmup_completed_successes": warmup_successes,
        "warmup_successes": warmup_successes,  # backwards-compatible alias
        "initial_replay_examples": initial_examples,
        "expected_initial_examples": expected_initial,
        "max_replay_examples_seen": max_examples,
        "replay_sessions": len(replay),
        "replay_correct_records": replay_correct_records,
        "replay_completed_successes": replay_successes,
        "replay_successes": replay_successes,  # backwards-compatible alias
        "valid_self_experience_replay": valid,
        "caveat": caveat,
        "counting_note": "Raw summary correct counts include every record with correct=true; replay seeding and this check use only correct AND status=completed records.",
        "paper_equivalent": False,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"replay validity: {args.output} valid={valid} warmup_successes={warmup_successes}")


if __name__ == "__main__":
    main()
