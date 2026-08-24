#!/usr/bin/env python3
"""SQLite integration benchmark for decision--commit races and rollback.

Unlike the dictionary sandbox, this benchmark uses real SQL transactions,
optimistic versions, rollback, and compare-and-swap updates.  It isolates the
executor property rather than LLM proposal quality.
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from credal_harness import CapabilityAuthority, ToolCall, state_digest


SEEDS = 50
EPISODES = 1_000


def setup() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("CREATE TABLE notes (key TEXT PRIMARY KEY, value TEXT NOT NULL, version INTEGER NOT NULL)")
    db.execute("CREATE TABLE audit (event TEXT NOT NULL)")
    return db


def context(db: sqlite3.Connection, key: str) -> dict[str, Any]:
    value, version = db.execute("SELECT value, version FROM notes WHERE key=?", (key,)).fetchone()
    return {"key": key, "value": value, "version": version, "authorized": True, "contract": "note-v1"}


def race_episode(policy: str, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    db = setup(); db.execute("INSERT INTO notes VALUES ('k', 'base', 1)")
    authority = CapabilityAuthority(secret=b"s" * 32)
    call = ToolCall("write_note", {"key": "k", "value": "agent", "expected_version": 1},
                    resource="note:k", effect="reversible", call_id=f"{seed:032x}")
    before = context(db, "k")
    token = authority.issue(call, "allow", ttl=60, context_digest=state_digest(before))
    # Intervening writer wins before the agent reaches the commit boundary.
    db.execute("UPDATE notes SET value='newer', version=2 WHERE key='k'")
    blocked = 0
    if policy in {"direct", "precheck"}:
        db.execute("UPDATE notes SET value=?, version=version+1 WHERE key='k'", ("agent",))
    elif policy == "compare_and_swap":
        changed = db.execute(
            "UPDATE notes SET value=?, version=version+1 WHERE key='k' AND version=?",
            ("agent", call.args["expected_version"]),
        ).rowcount
        blocked = int(changed == 0)
    elif policy == "capability":
        authority.verify(token)
        if token.context_digest != state_digest(context(db, "k")):
            blocked = 1
        else:
            db.execute("UPDATE notes SET value=?, version=version+1 WHERE key='k'", ("agent",))
    else:
        raise ValueError(policy)
    final_value = db.execute("SELECT value FROM notes WHERE key='k'").fetchone()[0]
    stale_overwrite = int(final_value == "agent")
    db.close()
    return stale_overwrite, blocked


def contract_drift_episode(commit_shadow: bool = False) -> tuple[int, int]:
    db = setup(); db.execute("INSERT INTO notes VALUES ('k', 'base', 1)")
    db.execute("BEGIN IMMEDIATE")
    before = context(db, "k")
    # Tool is declared read-only but contains an undocumented maintenance write.
    db.execute("UPDATE notes SET version=version+1 WHERE key='k'")
    after = context(db, "k")
    detected = int(after != before)
    if commit_shadow:
        db.execute("COMMIT")
    else:
        db.execute("ROLLBACK")
    live_version = db.execute("SELECT version FROM notes WHERE key='k'").fetchone()[0]
    db.close()
    return detected, int(live_version != 1)


def main() -> None:
    policies = ("direct", "precheck", "compare_and_swap", "capability")
    rows = []
    for policy in policies:
        stale = blocked = 0
        for seed in range(SEEDS * EPISODES):
            s, b = race_episode(policy, seed)
            stale += s; blocked += b
        n = SEEDS * EPISODES
        rows.append({"policy": policy, "episodes": n,
                     "stale_overwrite_rate": stale / n, "context_block_rate": blocked / n})
    detected, leaked = contract_drift_episode(commit_shadow=False)
    unsafe_detected, unsafe_leaked = contract_drift_episode(commit_shadow=True)
    output = {
        "config": {"seeds": SEEDS, "episodes_per_seed": EPISODES, "database": "SQLite in-memory"},
        "race": rows,
        "contract_drift": {
            "rollback_executor": {"detected": detected, "live_mutation": leaked},
            "unsafe_commit_control": {"detected": unsafe_detected, "live_mutation": unsafe_leaked},
        },
    }
    path = Path("experiments/results/sqlite_transaction_benchmark.json")
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
