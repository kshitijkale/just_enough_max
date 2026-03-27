#!/usr/bin/env python3
"""CLI tool for managing the idea population.

All output is JSON to stdout. The agent calls this via bash.

Usage:
  python ideas.py list
  python ideas.py random --n 3 [--pool better|worse|bin|all]
  python ideas.py top --k 5 [--pool better|worse|all]
  python ideas.py next-id
  python ideas.py add --from-json FILE
  python ideas.py update IDEA_ID --from-json FILE
  python ideas.py discard IDEA_ID --from-json FILE
  python ideas.py restore IDEA_ID
  python ideas.py diff IDEA_ID

JSON file for 'add' should contain:
  {"mode": "explore", "parent_ids": ["idea_1"], "E": "...", "H": "...", "code_file": "..."}

JSON file for 'update' should contain any of:
  {"status": "success", "val_bpb": 0.99, "peak_vram_mb": 45000, "mfu_percent": 39,
   "num_steps": 950, "num_params_M": 50, "probe_summary": "...", "O": "..."}

JSON file for 'discard' should contain:
  {"reason": "superseded by idea_12"}
"""

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # just_enough/
STATE_DIR = PROJECT_ROOT / "state"
IDEAS_FILE = STATE_DIR / "ideas.json"
BASELINE_FILE = PROJECT_ROOT / "baseline" / "train.py"


def load_ideas() -> list[dict]:
    if not IDEAS_FILE.exists():
        return []
    with open(IDEAS_FILE) as f:
        return json.load(f)


def save_ideas(ideas: list[dict]):
    tmp = IDEAS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ideas, f, indent=2)
    tmp.rename(IDEAS_FILE)


def get_baseline_bpb(ideas: list[dict]) -> float | None:
    for i in ideas:
        if i.get("mode") == "baseline" and i.get("result"):
            return i["result"]["val_bpb"]
    return None


def next_id(ideas: list[dict]) -> str:
    if ideas:
        max_num = max(int(i["id"].split("_")[1]) for i in ideas)
        return f"idea_{max_num + 1}"
    return "idea_0"


def filter_pool(ideas: list[dict], pool: str) -> list[dict]:
    """Filter ideas by pool type."""
    if pool == "all":
        return [i for i in ideas if i["status"] not in ("discarded", "pending")]
    elif pool == "better":
        baseline = get_baseline_bpb(ideas)
        if baseline is None:
            return []
        return [i for i in ideas if i.get("result") and i["result"]["val_bpb"] < baseline and i["status"] != "discarded"]
    elif pool == "worse":
        baseline = get_baseline_bpb(ideas)
        if baseline is None:
            return []
        return [i for i in ideas if i.get("result") and i["result"]["val_bpb"] >= baseline and i["status"] != "discarded" and i.get("mode") != "baseline"]
    elif pool == "bin":
        return [i for i in ideas if i["status"] == "discarded"]
    else:
        print(json.dumps({"error": f"Unknown pool: {pool}"}))
        sys.exit(1)


def read_json_file(path: str) -> dict:
    """Read a JSON file and return its contents."""
    with open(path) as f:
        return json.load(f)


def cmd_list(args):
    ideas = load_ideas()
    with_results = [i for i in ideas if i.get("result") and i["status"] != "discarded"]
    without_results = [i for i in ideas if not i.get("result") and i["status"] != "discarded"]
    discarded = [i for i in ideas if i["status"] == "discarded"]

    with_results.sort(key=lambda i: i["result"]["val_bpb"])

    output = {"total": len(ideas), "ideas": with_results + without_results}
    if discarded:
        output["discarded"] = discarded
    print(json.dumps(output, indent=2))


def cmd_random(args):
    ideas = load_ideas()
    pool = filter_pool(ideas, args.pool)
    n = min(args.n, len(pool))
    selected = random.sample(pool, n) if pool else []
    print(json.dumps(selected, indent=2))


def cmd_top(args):
    ideas = load_ideas()
    pool = filter_pool(ideas, args.pool)
    with_results = [i for i in pool if i.get("result")]
    with_results.sort(key=lambda i: i["result"]["val_bpb"])
    selected = with_results[:args.k]
    print(json.dumps(selected, indent=2))


def cmd_next_id(args):
    ideas = load_ideas()
    print(json.dumps({"next_id": next_id(ideas)}))


def cmd_add(args):
    data = read_json_file(args.from_json)
    ideas = load_ideas()

    new_id = next_id(ideas)

    idea = {
        "id": new_id,
        "parent_ids": data.get("parent_ids", []),
        "mode": data["mode"],
        "E": data["E"],
        "H": data["H"],
        "code_file": data["code_file"],
        "status": "pending",
        "result": None,
        "probe_summary": "",
        "O": "",
    }

    ideas.append(idea)
    save_ideas(ideas)
    print(json.dumps(idea, indent=2))


def cmd_update(args):
    data = read_json_file(args.from_json)
    ideas = load_ideas()
    target = None
    for i in ideas:
        if i["id"] == args.idea_id:
            target = i
            break

    if target is None:
        print(json.dumps({"error": f"Idea {args.idea_id} not found"}))
        sys.exit(1)

    if "status" in data:
        target["status"] = data["status"]

    # Build result dict from metrics
    metric_keys = ["val_bpb", "peak_vram_mb", "mfu_percent", "num_steps", "num_params_M"]
    if any(k in data for k in metric_keys):
        if target["result"] is None:
            target["result"] = {}
        for k in metric_keys:
            if k in data:
                target["result"][k] = data[k]

    if "probe_summary" in data:
        target["probe_summary"] = data["probe_summary"]
    if "O" in data:
        target["O"] = data["O"]

    save_ideas(ideas)
    print(json.dumps(target, indent=2))


def cmd_discard(args):
    ideas = load_ideas()

    reason = ""
    if args.from_json:
        data = read_json_file(args.from_json)
        reason = data.get("reason", "")

    for i in ideas:
        if i["id"] == args.idea_id:
            i["status"] = "discarded"
            if reason:
                i["O"] = i.get("O", "") + f" [Discarded: {reason}]"
            save_ideas(ideas)
            print(json.dumps(i, indent=2))
            return

    print(json.dumps({"error": f"Idea {args.idea_id} not found"}))
    sys.exit(1)


def cmd_restore(args):
    ideas = load_ideas()
    for i in ideas:
        if i["id"] == args.idea_id:
            if i["status"] != "discarded":
                print(json.dumps({"error": f"Idea {args.idea_id} is not discarded"}))
                sys.exit(1)
            if i.get("result"):
                baseline = get_baseline_bpb(ideas)
                if baseline and i["result"]["val_bpb"] < baseline:
                    i["status"] = "success"
                else:
                    i["status"] = "regression"
            else:
                i["status"] = "crash"
            save_ideas(ideas)
            print(json.dumps(i, indent=2))
            return

    print(json.dumps({"error": f"Idea {args.idea_id} not found"}))
    sys.exit(1)


def _resolve_code_file(code_file: str) -> Path:
    """Resolve a code_file path (may be relative to repo root or project root)."""
    p = Path(code_file)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to repo root (parent of just_enough/)
    candidate = PROJECT_ROOT.parent / code_file
    if candidate.exists():
        return candidate
    # Try relative to project root
    candidate = PROJECT_ROOT / code_file
    if candidate.exists():
        return candidate
    return PROJECT_ROOT.parent / code_file  # return best guess


def cmd_diff(args):
    ideas = load_ideas()
    target = None
    for i in ideas:
        if i["id"] == args.idea_id:
            target = i
            break
    if target is None:
        print(json.dumps({"error": f"Idea {args.idea_id} not found"}))
        sys.exit(1)

    if not BASELINE_FILE.exists():
        print(json.dumps({"error": f"Baseline not found at {BASELINE_FILE}"}))
        sys.exit(1)

    idea_path = _resolve_code_file(target.get("code_file", ""))
    if not idea_path.exists():
        print(json.dumps({"error": f"Code file not found: {target.get('code_file')}"}))
        sys.exit(1)

    baseline_lines = BASELINE_FILE.read_text().splitlines(keepends=True)
    idea_lines = idea_path.read_text().splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        baseline_lines, idea_lines,
        fromfile="baseline/train.py",
        tofile=target["code_file"],
        n=3,
    ))

    if not diff:
        print(f"{target['id']}: (no changes from baseline)")
    else:
        print(f"--- {target['id']}: {target.get('E', '')[:80]}")
        print("".join(diff), end="")


def main():
    parser = argparse.ArgumentParser(description="Manage idea population")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all ideas sorted by val_bpb")

    # random
    p_random = subparsers.add_parser("random", help="Randomly select ideas")
    p_random.add_argument("--n", type=int, required=True, help="Number to select")
    p_random.add_argument("--pool", default="all", choices=["all", "better", "worse", "bin"],
                          help="Pool to select from")

    # top
    p_top = subparsers.add_parser("top", help="Top K ideas by val_bpb")
    p_top.add_argument("--k", type=int, required=True, help="Number to select")
    p_top.add_argument("--pool", default="all", choices=["all", "better", "worse"],
                       help="Pool to select from")

    # next-id
    subparsers.add_parser("next-id", help="Get the next available idea ID")

    # add (reads from JSON file)
    p_add = subparsers.add_parser("add", help="Add a new idea (mode, parent_ids, E, H, code_file)")
    p_add.add_argument("--from-json", required=True, help="Path to JSON file with idea data")

    # update (reads from JSON file)
    p_update = subparsers.add_parser("update", help="Update an idea with results")
    p_update.add_argument("idea_id", help="Idea ID to update")
    p_update.add_argument("--from-json", required=True, help="Path to JSON file with update data")

    # discard
    p_discard = subparsers.add_parser("discard", help="Discard an idea (move to bin)")
    p_discard.add_argument("idea_id", help="Idea ID to discard")
    p_discard.add_argument("--from-json", default=None, help="Path to JSON file with reason")

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore a discarded idea")
    p_restore.add_argument("idea_id", help="Idea ID to restore")

    # diff
    p_diff = subparsers.add_parser("diff", help="Show unified diff between baseline and idea's code")
    p_diff.add_argument("idea_id", help="Idea ID to diff")

    args = parser.parse_args()

    commands = {
        "list": cmd_list,
        "random": cmd_random,
        "top": cmd_top,
        "next-id": cmd_next_id,
        "add": cmd_add,
        "update": cmd_update,
        "discard": cmd_discard,
        "restore": cmd_restore,
        "diff": cmd_diff,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
