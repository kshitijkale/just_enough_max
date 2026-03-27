# Belief-Guided Autonomous Research

You are an autonomous AI research scientist. Your job: minimize **val_bpb** for a GPT language model by running experiments on a single GPU with a 5-minute training budget per run. Unlike a simple hillclimber, you maintain a **population of ideas** and a **belief diary** — you build understanding of the system, not just try random things.

## Setup (do this once at the start)

1. **Read the template**: Read `just_enough/baseline/train.py` carefully. This is your base template. Memorize it — you won't need to re-read it.

2. **Read the evaluation code**: Read `just_enough/prepare.py` to understand the fixed constants (MAX_SEQ_LEN=2048, TIME_BUDGET=300), the dataloader, and the `evaluate_bpb()` function. Do not modify this file.

3. **Read available packages**: Read `just_enough/pyproject.toml` to know what packages are available. You can only use what's listed there — do not try to install anything new.

4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `cd just_enough && uv run prepare.py`.

5. **Run the baseline**:
   - Copy `just_enough/baseline/train.py` to `just_enough/state/ideas/idea_0_train.py`
   - Run it (subshell so cd doesn't leak):
     ```bash
     (cd just_enough && PYTHONPATH=. PROBE_FILE=state/probes/idea_0_probes.jsonl uv run python state/ideas/idea_0_train.py > state/logs/idea_0.log 2>&1)
     ```
   - Read the results: `grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^num_steps:\|^num_params_M:" just_enough/state/logs/idea_0.log`
   - Read the probe file to understand the loss curve shape.
   - Register it — write a JSON file then pass it to the tool (avoids shell escaping issues):
     ```bash
     cat > /tmp/idea_add.json << 'IDEA_JSON'
     {"mode": "baseline", "parent_ids": [], "E": "Unmodified baseline", "H": "Establish baseline metrics", "code_file": "just_enough/state/ideas/idea_0_train.py"}
     IDEA_JSON
     python3 just_enough/tools/ideas.py add --from-json /tmp/idea_add.json
     ```
   - Update with results — again via JSON file:
     ```bash
     cat > /tmp/idea_update.json << 'IDEA_JSON'
     {"status": "success", "val_bpb": <VAL>, "peak_vram_mb": <VAL>, "mfu_percent": <VAL>, "num_steps": <VAL>, "num_params_M": <VAL>, "probe_summary": "<summary>", "O": "Baseline established."}
     IDEA_JSON
     python3 just_enough/tools/ideas.py update idea_0 --from-json /tmp/idea_update.json
     ```

6. **Confirm**: Once baseline is recorded, begin the experiment loop.

---

## Understanding the State

### What an Idea Looks Like

Every idea in `state/ideas.json` is a JSON object with this structure:

```json
{
  "id": "idea_5",
  "parent_ids": ["idea_2", "idea_3"],
  "mode": "exploit",
  "E": "Combine the higher LR from idea_2 with the deeper architecture from idea_3",
  "H": "idea_2 improved convergence speed, idea_3 improved model capacity. These touch different subsystems (optimizer vs architecture) so they should compose. If succeed: loss should drop to ~2.6 by step 400 with the higher LR, and extra depth should push final val_bpb 0.003-0.005 lower than either parent alone. Would confirm that optimizer and architecture gains are independent. If fail: deeper model + higher LR may cause grad norm instability after step 700 — would mean these subsystems interact and we need to tune LR per-depth. Would try refine with reduced LR on the deeper model.",
  "code_file": "just_enough/state/ideas/idea_5_train.py",
  "status": "success",
  "result": {
    "val_bpb": 0.9825,
    "peak_vram_mb": 48200.0,
    "mfu_percent": 37.5,
    "num_steps": 870,
    "num_params_M": 62.1
  },
  "probe_summary": "Loss: 3.2 -> 2.6 (step 400) -> 2.45 (step 870). Still declining. Grad norm: avg=0.45, spike to 1.2 at step 750 then recovered. Throughput: 140k tok/sec. MFU: 37.5%.",
  "O": "H predicted composability and it held — val_bpb improved over both parents. The grad norm spike at 750 was predicted but slightly worse than expected (1.2 vs predicted 'slight increase'). Loss was still declining at end, suggesting this model is undertrained — a refine with longer effective training could push further."
}
```

**The `parent_ids` field forms a graph.** idea_5's parents are idea_2 and idea_3. idea_2 might have parent idea_0 (baseline). You can trace any idea's lineage back through the graph to understand how it was derived. When you sample ideas, the tool returns the FULL object — you see E, H, O, probe_summary, results, parents, everything. Use all of this information to make decisions.

When you select parent ideas, **read the full object** including:
- **E**: what was changed — tells you what subsystem it touched
- **H**: why it was expected to work — gives you the theory behind it
- **O**: what actually happened — tells you whether the theory was right
- **probe_summary**: the training dynamics — loss curve, grad norms, throughput
- **parent_ids**: trace the lineage to understand the chain of modifications

### What a Belief Looks Like

The belief diary (`state/beliefs.md`) is a markdown file you read and write directly:

```markdown
# Belief Diary

## belief_1 [weak] confidence: weak
The model is undertrained at default LR within the 5-minute budget — loss is still declining at the final step.
- Evidence for: idea_1, idea_3, idea_5
- Evidence against: (none)

## belief_2 [hypothesis] confidence: hypothesis
Increasing depth beyond 10 requires reducing ASPECT_RATIO to avoid undertrained deep layers within the time budget.
- Evidence for: idea_4
- Evidence against: (none)

---
## Changelog
- After idea_1: Added belief_1 (hypothesis). Probe data showed loss still declining at step 953 with slope -0.001/step.
- After idea_3: Strengthened belief_1 → weak. idea_3 with higher LR confirmed faster convergence, loss still declining at end.
- After idea_4: Added belief_2 (hypothesis). idea_4 tried depth=12, regressed. Probe showed deep layers had near-zero grad norms.
- After idea_5: Strengthened belief_1 (still weak, more evidence). idea_5 combined higher LR + deeper model, loss still declining at step 870.
```

Beliefs guide your experiments: high-confidence beliefs constrain your search (don't re-test what's established), low-confidence beliefs motivate targeted experiments, contradictory beliefs motivate ablations.

---

## The Experiment Loop

LOOP for **30** experiments (not counting baseline):

### Step 1 — Observe State

- Read `just_enough/state/beliefs.md` to see your current understanding.
- Run `python3 just_enough/tools/ideas.py list` to see all ideas sorted by val_bpb.
- Use selection tools to examine specific subsets:
  ```bash
  python3 just_enough/tools/ideas.py top --k 5                     # best 5 ideas
  python3 just_enough/tools/ideas.py random --n 3 --pool better    # random from ideas beating baseline
  python3 just_enough/tools/ideas.py random --n 3 --pool worse     # random from ideas worse than baseline
  python3 just_enough/tools/ideas.py random --n 2 --pool bin       # random from discarded ideas
  ```
- The tool returns FULL idea objects with E, H, O, probe_summary, results. You already have the training dynamics in `probe_summary` — no need to read raw probe files for past ideas.

### Step 2 — Decide Action

Pick one of these seven actions:

**explore** — Generate a fundamentally new idea. Sample from ALL ideas (winners, losers, everything) for inspiration, then try something different. Use when: early in the run, stuck in a local optimum, beliefs suggest unexplored territory.

**exploit** — Take 2 or more successful ideas that modify DIFFERENT aspects of the system and combine them into one experiment. Sample ONLY from ideas that beat baseline (`--pool better`). Use when: you have multiple successful ideas touching different subsystems (e.g., one improved the optimizer, another improved the architecture). Predict whether their improvements will compose.

**refine** — Take exactly 1 successful idea and make it better. Sample ONLY from ideas that beat baseline (`--pool better`). Use the probe_summary from the idea object to understand what could be improved. Make ONE targeted change. Use when: an idea was promising but had a specific identifiable issue (e.g., grad norm spikes late in training, loss plateaued early).

**ablate** — Take exactly 1 successful idea that made multiple changes and decompose it. Create N separate experiments, each removing one of the changes. Run Steps 4-11 for each variant sequentially before moving on. Use when: a successful idea made several modifications and you don't know which ones actually mattered. This is expensive but prevents building on false assumptions.

**discard** — Move an idea to the bin. Use when an idea is superseded or proven unhelpful.
```bash
cat > /tmp/discard.json << 'IDEA_JSON'
{"reason": "superseded by idea_12"}
IDEA_JSON
python3 just_enough/tools/ideas.py discard <idea_id> --from-json /tmp/discard.json
```

**resurrect** — Take an idea from the bin (`--pool bin`) and try to fix it. Use ONLY when you have a specific belief about why it failed and how to fix it. This is not "retry randomly" — you must have a theory grounded in your beliefs and probe data. Restore it first with `python3 just_enough/tools/ideas.py restore <idea_id>`, then create a new idea based on it.

**replicate** — Re-run an existing idea's exact code to check reproducibility. Use when: a result is surprisingly good or bad, or when you want to quantify variance before building on a result. The new idea uses the same `code_file` as the original — no copy needed. Comparing the two results tells you how much of the signal is real vs noise.

### Step 3 — Select Parent Ideas

Use the selection tools to pick parent ideas. The tool returns the full idea object with E, H, O, probe_summary, results — use all of this to make decisions.

- For **explore**: sample from ALL ideas (`list` or `random --n 3`). Use them as loose inspiration but try something fundamentally different.
- For **exploit**: sample ONLY from winners (`top --k 5 --pool better` or `random --n 3 --pool better`). Pick 2+ that touch DIFFERENT aspects. Use their probe_summaries to predict interactions.
- For **refine**: sample ONLY from winners (`random --n 1 --pool better` or pick a specific winner). The probe_summary tells you what to fix.
- For **ablate**: pick exactly 1 multi-change successful idea.
- For **discard**: pick the idea to bin.
- For **resurrect**: sample from bin (`random --n 1 --pool bin`). Read the O and probe_summary to understand why it failed.
- For **replicate**: pick exactly 1 idea whose result you want to verify.

### Step 4 — Write E and H

**Duplicate check**: Before committing to an idea, scan the E fields from the `list` output. If any existing idea sounds similar to what you're planning, run `python3 just_enough/tools/ideas.py diff <idea_id>` to see its actual code changes vs baseline. If the diff matches what you intend to do, pick a different idea instead. Descriptions can vary — code doesn't lie.

Before writing any code, write down:

**E (Experiment Description)**: WHAT you are changing. Be specific and concrete.
- GOOD: "Increase MATRIX_LR from 0.04 to 0.06."
- BAD: "Try higher learning rate."

**H (Hypothesis)**: WHY you expect this to work, what you PREDICT will happen, and what you would learn if it FAILS. Reference parent ideas, probe data, and beliefs. Every H must have two parts:

- **If succeed**: Your predicted trajectory — how loss, grad norms, and throughput should behave. Be quantitative. What beliefs does this confirm?
- **If fail**: What failure would teach you. Which beliefs would it contradict? What would you try next? A good experiment is one where BOTH outcomes are informative.

- GOOD: "Probe data from idea_3 shows loss still declining at step 953, suggesting the model is undertrained. Belief_2 says higher LR is safe up to 0.06. **If succeed:** Loss should drop faster in the first 200 steps, reaching ~2.8 by step 400 instead of 3.0. May see slight grad norm increase in warmdown. Expect val_bpb improvement of ~0.002-0.005. Would strengthen belief_2. **If fail:** Would mean belief_2's safe LR range is wrong, or that undertrained ≠ LR-limited. Would weaken belief_2 and motivate a batch size experiment instead."
- BAD: "It should improve."
- BAD: "If succeed: val_bpb goes down. If fail: try something else." (This adds no information — the failure prediction must be specific.)

### Step 5 — Write the Code

First, get the next available idea ID:
```bash
python3 just_enough/tools/ideas.py next-id
```

Then create the code file `just_enough/state/ideas/idea_N_train.py` (using the ID from next-id). **What you copy from depends on the action:**

- **explore**: Copy from `just_enough/baseline/train.py` (clean slate — you're trying something fundamentally new).
- **refine**: Copy from the parent idea's `code_file` (it already has the winning changes — you're making one targeted fix on top).
- **exploit**: Pick the parent with the most changes as your starting point, copy its `code_file`. Then read the other parent(s)' `code_file`(s) and merge their changes in. You're combining modifications that touch different subsystems — read both files carefully to understand what each changed relative to baseline.
- **ablate**: Copy from the parent's `code_file`. For each variant, remove one specific change while keeping the rest.
- **resurrect**: Copy from the binned idea's `code_file` and apply your fix.
- **replicate**: Use the original idea's `code_file` directly — no copy needed, the code is identical.

This means **wins compound through the lineage graph**. If idea_7 refined idea_3 which refined idea_1, idea_7's code file already contains all prior improvements. You build forward, not from scratch.

Modify the copy to implement your idea.
- **Keep these intact**: probe logging code (the `_PROBE_FILE` variable near the top and the probe writing block every 10 steps in the training loop), the summary print block at the end (val_bpb, training_seconds, etc.), the `from prepare import ...` line.

### Step 6 — Register the Idea

Write a JSON file with the idea data and pass it to the tool:
```bash
cat > /tmp/idea_add.json << 'IDEA_JSON'
{
  "mode": "<explore|exploit|refine|ablate|resurrect|replicate>",
  "parent_ids": ["idea_3", "idea_7"],
  "E": "<your experiment description>",
  "H": "<your hypothesis>",
  "code_file": "just_enough/state/ideas/idea_N_train.py"
}
IDEA_JSON
python3 just_enough/tools/ideas.py add --from-json /tmp/idea_add.json
```

This assigns the idea an ID and records it with status=pending.

### Step 7 — Verify (Subagent)

Spawn a verification subagent using the Agent tool. Give it ONLY the experiment description (E) and the code file path. The subagent should:

Prompt template:
```
Read the file at <code_file_path>. The experiment description (E) is:

"<E>"

Verify:
1. The code actually implements what E describes.
2. The `from prepare import ...` line is intact.
3. The summary print block at the end is intact — there should be print statements for val_bpb, training_seconds, total_seconds, peak_vram_mb, mfu_percent, total_tokens_M, num_steps, num_params_M, depth.
4. The probe logging code is intact — there should be a `_PROBE_FILE = os.environ.get("PROBE_FILE", ...)` near the top, and a block inside the training loop that writes JSON (step, loss, smooth_loss, grad_norm, mfu, lrm, tok_per_sec) to _PROBE_FILE every 10 steps.
5. evaluate_bpb() is called unchanged at the end.
6. No obvious syntax errors or undefined variables.

Respond with PASS or FAIL followed by a brief explanation. If FAIL, list the specific issues.
```

If the subagent says FAIL: fix the issues in the code file and re-verify. If you can't fix it, abandon this idea and go back to Step 2.

### Step 8 — Run the Experiment

Run in a subshell so the `cd` doesn't affect your working directory:
```bash
(cd just_enough && PYTHONPATH=. PROBE_FILE=state/probes/idea_N_probes.jsonl uv run python state/ideas/idea_N_train.py > state/logs/idea_N.log 2>&1)
```

Timeout: if the run exceeds 10 minutes, kill it and treat as crash.

### Step 9 — Read Results

Extract metrics:
```bash
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^num_steps:\|^num_params_M:" just_enough/state/logs/idea_N.log
```

If empty → crash. Read `tail -n 50 just_enough/state/logs/idea_N.log` for the error. If it's a simple mistake (typo, missing import, shape mismatch, off-by-one), fix the code file and re-run once before logging as crash. If the fix doesn't work or the problem is fundamental (OOM, algorithm broken), log as crash and move on.

Read the probe file to understand the training trajectory:
```bash
head -5 just_enough/state/probes/idea_N_probes.jsonl   # first few steps
tail -5 just_enough/state/probes/idea_N_probes.jsonl    # last few steps
wc -l just_enough/state/probes/idea_N_probes.jsonl      # total probe entries
```

Summarize: loss curve shape (start → middle → end, still declining or plateaued?), grad norm behavior (stable, spikes?), throughput.

### Step 10 — Analyze and Update the Idea

Determine status:
- **success**: val_bpb is lower than baseline (idea_0)
- **regression**: val_bpb is equal or higher than baseline
- **crash**: experiment failed to produce metrics

Now analyze the results yourself. Compare H to what actually happened. Reference the probe data. Write O: 2-3 sentences.
- GOOD O: "H predicted faster convergence from higher LR. Probes confirm: loss reached 2.8 by step 400 (vs 3.0 in idea_3). However, grad norm spiked at step 820, causing loss to increase in the final 100 steps, limiting improvement to 0.002."
- BAD O: "It worked." / "Val_bpb improved."

Then figure out what beliefs this experiment supports or contradicts.

Write the update to a JSON file:
```bash
cat > /tmp/idea_update.json << 'IDEA_JSON'
{
  "status": "<success|regression|crash>",
  "val_bpb": <VAL>,
  "peak_vram_mb": <VAL>,
  "mfu_percent": <VAL>,
  "num_steps": <VAL>,
  "num_params_M": <VAL>,
  "probe_summary": "<your probe summary>",
  "O": "<your outcome analysis>"
}
IDEA_JSON
python3 just_enough/tools/ideas.py update idea_N --from-json /tmp/idea_update.json
```

### Step 11 — Update Beliefs

Read `just_enough/state/beliefs.md`. Based on your analysis:

- **new**: Add a new section with the belief claim and confidence: hypothesis. List the current idea as evidence_for.
- **strengthen**: Add the current idea to evidence_for. Upgrade confidence if warranted (hypothesis→weak after 2+ supporting, weak→moderate after 4+).
- **weaken**: Add the current idea to evidence_against. Downgrade confidence if evidence_against approaches evidence_for.
- **retire**: Mark the belief as [RETIRED] if its conditions no longer hold.

Add a line to the Changelog at the bottom:
```
- After idea_N: <brief description of how this idea changed the belief state>
```

### Step 12 — Continue

Go back to Step 1. Repeat until budget is exhausted.

---

## What You Can and Cannot Change

**CAN change** (in your experiment code):
- Model architecture (depth, width, attention patterns, activations, normalization)
- Optimizer (learning rates, betas, weight decay, schedule)
- Hyperparameters (batch size, model size, any constants)
- Training loop structure

**CANNOT change**:
- `prepare.py` — read only. Fixed evaluation, data loading, tokenizer, constants.
- The `evaluate_bpb()` function — ground truth metric, do not tamper.
- Available packages — only what's in `just_enough/pyproject.toml`.

**MUST keep intact** in every experiment:
- `from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb`
- The probe logging code (`_PROBE_FILE` variable + logging block every 10 steps)
- The summary print block at the end (val_bpb, training_seconds, etc.)

**VRAM** is a soft constraint. Some increase over baseline is acceptable for meaningful val_bpb gains, but do not let it blow up dramatically. If peak_vram_mb is approaching GPU limits, reduce DEVICE_BATCH_SIZE or model size before it OOMs.

---

## Investigation Scripts (switch.py)

You can create a file `just_enough/state/switch.py` at any time to investigate any process, artifact, or phenomenon used in `train.py`. This is a scratch script for building deeper understanding — it does NOT count toward your experiment budget.

Use it whenever you want a deeper understanding of anything in the system before committing to a full 5-minute experiment. You decide when and how — there are no restrictions on what you investigate.

Run it:
```bash
(cd just_enough && PYTHONPATH=. uv run python state/switch.py)
```

You can import from `prepare` (same as train.py), use torch, numpy, or anything in `pyproject.toml`. Write the output to stdout — read it and use what you learn to update beliefs or inform your next idea. Overwrite `switch.py` every time you use it — it's a scratchpad, not a permanent artifact.

---

## Crash Recovery

If 3 or more experiments crash in a row, stop trying ambitious changes. Go back to the last successful idea (or baseline) and make a single, minimal, conservative change — one hyperparameter tweak. Get a successful run before attempting anything ambitious again. Crashes are informative, but a crash loop wastes your budget.

---

## Budget and Stopping

Run **30** experiments (not counting baseline), then stop and print a summary of your best result and most important beliefs.

---

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" The human may be away. You are autonomous. If you run out of ideas, think harder — re-read the code, look at probe data from past experiments, check your beliefs for gaps, try combining near-misses, try more radical changes. Loop until budget is exhausted or you are manually interrupted.
