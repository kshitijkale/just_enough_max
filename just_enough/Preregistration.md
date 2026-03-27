# Preregistration: Belief-Guided Autonomous Research

Written before running any experiments. This document commits to our hypotheses, expected outcomes, and failure criteria so we cannot rationalize results after the fact. If the experiment goes wrong, we return here to diagnose what broke and why.

---

## 1. High-Level Overview and Insights

### The problem with greedy hillclimbing

Autoresearch (Karpathy) is a greedy hillclimber: try something, check val_bpb, keep or discard. The agent has no memory beyond its context window. Failed experiments are erased via `git reset --hard`. There is no mechanism to learn from failure, avoid repeating mistakes, or build cumulative understanding. Each experiment is independent — the agent rediscovers the same insights every time.

The ES paper (Si et al. 2026) improves on this with population-based search — maintaining a database of (idea, metric) pairs and sampling from winners to generate new batches. But ideas are still opaque text blobs with no structure, no parent-child links, and no interpretation. The ideator never sees actual code or training dynamics. It's pattern-matching on a ranked list.

### Our core insight: failed experiments carry information

A run that regresses is not garbage — it updates your understanding of the search space. "Doubling depth caused val_bpb to regress because probe data showed the deeper layers had near-zero gradient norms" tells you something about every depth-related idea, not just the one that failed. A hillclimber throws this away. We keep it.

Three things make this possible:

1. **Beliefs** — explicit, falsifiable claims about the system. Not just "what happened" (the idea log) but "what we think is true about the landscape" (the world model). Beliefs constrain search (don't re-test what's established), prioritize experiments (test low-confidence beliefs that would change strategy), and predict outcomes (the H in each idea references beliefs).

2. **Probe data** — loss curves, gradient norms, throughput every 10 steps. The agent doesn't just see a final number — it sees *training dynamics*. This turns analysis from confabulation ("it probably improved because of X") into empirical observation ("probe data shows loss converged 200 steps earlier and grad norms were stable"). Beliefs grounded in probe data are qualitatively different from beliefs guessed from a scalar.

3. **Idea lineage** — the parent_ids graph. When idea_7 (refine of idea_3, which was exploit of idea_1+idea_2) succeeds, we know exactly which chain of modifications led there. When it fails, we can trace which ancestor's contribution broke. The ES paper has no lineage — every epoch is a fresh batch.

### The bet

We are betting that an agent which maintains structured understanding (beliefs + lineage + probe data) will extract more value per experiment than an agent that just hill-climbs on a scalar. The tradeoff: each experiment has more overhead (register idea, verify code, analyze results, update beliefs). We get fewer experiments per hour but more information per experiment. In a sequential single-GPU setting where each run costs 5 minutes, making every experiment count is the right optimization target.

---

## 2. Implementation Details

### Architecture

The system is Claude Code itself — no Python orchestrator. The agent reads `program.md` and follows it autonomously. There are only 3 authored files:

| File | Role |
|---|---|
| `program.md` | Prescriptive step-by-step instructions. THE product. |
| `baseline/train.py` | Template with always-on probes. Every experiment starts as a copy of this. |
| `tools/ideas.py` | CLI for idea management. The only code we wrote. |

### State

| Artifact | Format | Who manages it |
|---|---|---|
| `state/ideas.json` | JSON array | ideas.py tool (agent calls via bash) |
| `state/beliefs.md` | Markdown | Agent reads/writes directly |
| `state/ideas/idea_N_train.py` | Python | Agent copies and modifies |
| `state/probes/idea_N_probes.jsonl` | JSONL | Written by train.py during run |
| `state/logs/idea_N.log` | Text | stdout/stderr from run |

### The experiment loop (12 steps)

1. **Observe state** — read beliefs, list ideas, sample from pools
2. **Decide action** — one of 7: explore, exploit, refine, ablate, discard, resurrect, replicate
3. **Select parents** — from appropriate pool (all, better, worse, bin)
4. **Duplicate check + Write E and H** — diff existing ideas' code, formulate hypothesis with predicted trajectory
5. **Write code** — copy from parent's code_file (not always baseline), modify
6. **Register idea** — add to ideas.json via tool
7. **Verify** — subagent checks code implements E, probes intact, no syntax errors
8. **Run** — 5 min training in subshell
9. **Read results** — extract metrics, quick-fix simple crashes, read probe data
10. **Analyze** — compare H to outcome using probe data, write O
11. **Update beliefs** — add/strengthen/weaken/retire beliefs based on evidence
12. **Continue** — loop for 30 experiments

### Key design decisions

- **Copy-from-parent, not always-from-baseline.** For refine/exploit, the agent copies the parent idea's code file. Wins compound through the lineage graph. Explore starts from baseline (clean slate).
- **JSON file input for all text fields.** Avoids shell escaping issues with E, H, O, probe_summary.
- **next-id before writing code.** Solves the chicken-and-egg problem of needing the ID to name the file.
- **Verification subagent.** Clean-room check — sees only E and code, catches drift between intent and implementation.
- **No analysis subagent.** The main agent does its own analysis — it has the full context (beliefs, parent ideas, probe data).
- **diff command for duplicate detection.** Compares code against baseline, not descriptions. Descriptions can vary; code doesn't lie.
- **Replicate action.** Re-run exact same code to quantify variance. Two data points beat one.
- **Always-on probes only.** Loss, smooth_loss, grad_norm, mfu, lrm, tok_per_sec every 10 steps. No belief-driven probes in this version — keeping it minimal.

### What we deliberately left out (and why)

- **Python orchestrator** — rejected as overdone and hardcoded. Claude Code IS the loop.
- **Git-based state** — rejected in favor of file-per-idea. Simpler, no branch management, allows non-linear exploration.
- **Simplicity criterion** — deliberately omitted. We want to see if the agent naturally manages complexity or if it bloats code for marginal gains.
- **Belief-driven probes** — deferred. Always-on probes are the MVP. If they prove valuable, belief-driven probes are the next step.
- **Fixed action schedule** — the agent chooses its own policy. We don't prescribe "3 explores then exploit." We want to see what emerges.

---

## 3. Expected Outcomes

All predictions below are relative to autoresearch (Karpathy's greedy hillclimber) running the same number of experiments on the same hardware.

### 3.1 If it works (primary hypotheses)

**H1: Better sample efficiency — more val_bpb improvement per experiment.**

Autoresearch treats each run independently. Our system carries forward understanding. We predict that at experiment 15, just_enough will have a lower best val_bpb than autoresearch at experiment 15. The gap should widen as beliefs accumulate.

*Observable:* Plot best-so-far val_bpb vs experiment number for both systems. Just_enough's curve should be steeper and/or lower.

*What it would mean if wrong:* The overhead of beliefs/analysis doesn't pay for itself in better experiment selection. The LLM is already good enough at implicit pattern extraction from its context window, and explicit beliefs are redundant structure.

**H2: Fewer wasted runs — the agent avoids repeating known-bad directions.**

Autoresearch has no duplicate detection and no memory of why things failed. Its agent can (and does) re-try similar ideas. Our system has beliefs that constrain search + diff-based duplicate detection.

*Observable:* Count "wasted runs" — experiments that try something a prior experiment already established doesn't work (same direction, same subsystem, no new theory for why it would work this time). Just_enough should have a lower wasted run rate, especially after experiment 10.

*What it would mean if wrong:* The agent ignores beliefs when generating ideas (the LLM doesn't actually condition on the belief diary), or the belief diary is too vague to constrain anything useful.

**H3: Exploit-mode ideas will outperform explore-mode ideas in the mid-to-late game.**

Once the agent has several successful ideas touching different subsystems, combining them (exploit) should reliably beat trying something new (explore). Autoresearch can't do this — it has only one incumbent, not a population.

*Observable:* Track success rate (beat baseline) by action mode. After experiment 10, exploit and refine should have higher success rates than explore. Exploit ideas should produce the best absolute val_bpb values.

*What it would mean if wrong:* Successful ideas don't compose — their improvements are redundant or conflicting. This would mean the search space has strong interactions between subsystems, and combining independent wins doesn't work.

**H4: The belief diary will show convergence — fewer new beliefs, higher confidence over time.**

Early experiments should generate many hypothesis-level beliefs. Later experiments should mostly strengthen/weaken existing beliefs. By experiment 25, the agent should have a small set of moderate-to-strong beliefs that accurately predict outcomes.

*Observable:* Plot beliefs created per experiment, and average confidence level, over time. Should see declining creation rate and increasing average confidence.

*What it would mean if wrong:* Beliefs are too specific (tied to exact hyperparameter values, not transferable) or the agent keeps flip-flopping (evidence is noisy, beliefs oscillate). Either means the belief granularity is wrong.

**H5: Probe data will ground the analysis — O fields will reference specific probe observations.**

The analysis (O field) should cite probe data: "loss reached X by step Y," "grad norm spiked at step Z." Autoresearch's agent can only say "val_bpb improved/regressed."

*Observable:* Fraction of O fields that reference specific probe values (step numbers, loss values, grad norm observations) vs generic statements. Should be >70%.

*What it would mean if wrong:* The agent ignores probe data and writes generic analysis anyway. The probes exist but don't change the agent's reasoning. This would mean the program.md instructions for analysis aren't strong enough, or the LLM doesn't naturally integrate quantitative evidence.

### 3.2 If it partially works

**H6: Refine will be the highest-value action, but the agent will underuse it.**

Refine (targeted fix to a winning idea using probe data) should have the best improvement-per-experiment. But the agent will probably over-explore early and under-refine, because LLMs tend to prefer novelty over incremental improvement.

*Observable:* Compare val_bpb improvement per action type. If refine has the best average improvement but <20% mode share, the agent is leaving value on the table.

*What to do:* Add a prompt nudge: "if the last idea was a success with identifiable issues in probe data, prefer refine over explore."

**H7: The lineage graph will be shallow — most ideas will be 1-2 generations from baseline.**

Despite the copy-from-parent design, the agent will probably default to exploring from baseline rather than building deep chains. Deep lineage (5+ generations) requires the agent to trust that accumulated code changes are all correct.

*Observable:* Maximum and average depth of the idea lineage graph. If max depth is <4 after 30 experiments, the agent isn't compounding wins.

*What to do:* Make program.md more explicit about building on the best existing code, not defaulting to baseline.

### 3.3 If it fails (failure modes)

**F1: Bookkeeping overhead kills throughput.**

The belief diary + idea registration + verification + analysis could consume so much context and time that the agent runs significantly fewer experiments than autoresearch in the same wall-clock time.

*Observable:* Experiments per hour. Autoresearch runs ~12/hour (5 min each, minimal overhead). If just_enough runs <8/hour, the overhead is too high.

*Diagnosis:* Which step is the bottleneck? Verification subagent? Belief updates? Analysis writing? Time between experiments (LLM thinking time)?

*What to do:* Simplify the loop. Drop verification subagent (trust the agent's code). Shorten belief updates to 1 sentence. Make analysis optional for regressions.

**F2: Beliefs become a trap — the agent stops exploring because it's "confident."**

If the agent forms strong beliefs early (possibly wrong ones grounded in LLM priors, not data), it may converge prematurely. Autoresearch doesn't have this failure mode because it has no beliefs to get trapped by.

*Observable:* Explore-mode usage drops to <10% after experiment 10. Belief confidence levels are high but val_bpb has stagnated. The agent keeps refining the same branch without trying fundamentally different approaches.

*Diagnosis:* Are the high-confidence beliefs actually correct? Check by running experiments that violate them. If the violations succeed, the beliefs were wrong and constraining the search.

*What to do:* Add a minimum explore rate (at least 1 in 5 experiments must be explore). Add a stagnation detector: if best val_bpb hasn't improved in 5 experiments, force an explore.

**F3: The agent's analysis is confabulation — O fields don't match probe data.**

The agent writes plausible-sounding analysis that sounds like it references probes but actually contradicts or ignores the actual probe values. Beliefs built on bad analysis are worse than no beliefs.

*Observable:* Manually verify 5-10 O fields against the actual probe JSONL files. Do the cited step numbers and values match? Does the narrative match the actual loss curve shape?

*Diagnosis:* Is the agent reading probe files at all, or skipping that step? Are the probe summaries accurate but the causal reasoning wrong?

*What to do:* If the agent skips probes — make the probe-reading step more prescriptive (specific head/tail commands). If the reasoning is wrong — this is a fundamental LLM limitation, consider removing beliefs and falling back to population-only search.

**F4: Code accumulation causes cascading failures.**

As wins compound through the lineage graph, code files accumulate changes. By generation 5, the code may be significantly different from baseline in ways the agent doesn't fully understand. A refine at generation 5 might break something from generation 2 without realizing it.

*Observable:* Crash rate increases as lineage depth increases. Ideas with depth >3 crash more often than depth 1-2.

*Diagnosis:* Is the verification subagent catching these? Are the crashes from interaction effects between accumulated changes?

*What to do:* Add a periodic "consolidation" step — every 10 experiments, the agent reviews the best idea's full diff against baseline and simplifies redundant or conflicting changes.

**F5: The agent treats program.md as a checklist, not a research methodology.**

The agent mechanically follows the 12 steps without actually reasoning about what to try next. Beliefs are formulaic ("X improved Y"), hypotheses are vague, and the action selection is essentially random.

*Observable:* H fields are generic ("should improve val_bpb"). E fields are minor parameter tweaks. Beliefs are just restatements of experimental results. No interesting exploit combinations or theory-driven experiments.

*Diagnosis:* Is this a prompt quality issue (program.md doesn't inspire deep thinking) or a model capability issue (the LLM can't do genuine research reasoning)?

*What to do:* If prompt quality — iterate on program.md, add richer examples of good H and belief reasoning. If model capability — this is a fundamental finding worth reporting. Consider whether a simpler system (just population + explore/exploit, no beliefs) captures most of the value without requiring research-level reasoning.

### 3.4 What we are NOT predicting

- **Absolute val_bpb values.** These depend on hardware, data, and the baseline model. We only predict relative improvement trajectories.
- **Which specific ideas will work.** The agent discovers this. We predict the *meta-level* — how the agent searches, not what it finds.
- **That beliefs are necessary.** It's possible population-based search alone (ideas + lineage, no beliefs) captures 90% of the value. This would be a useful finding — it means the simpler system is better.

### 3.5 Decision criteria

After 30 experiments, we evaluate:

| Outcome | Decision |
|---|---|
| H1+H2 confirmed, val_bpb clearly better than autoresearch | Success. Iterate on program.md to improve further. |
| H1 marginal, H5 confirmed (probes used well, beliefs grounded) | Partial success. The framework works but needs more experiments to show. Extend budget. |
| F1 dominant (overhead kills throughput) | Simplify the loop. Drop verification, shorten analysis, test again. |
| F2 dominant (beliefs trap the agent) | Add minimum explore rate, stagnation detector. Test again. |
| F3 dominant (analysis is confabulation) | Drop beliefs, keep population + probe data only. Test if that's enough. |
| F5 dominant (mechanical execution) | Iterate on program.md with richer examples and reasoning prompts. |
| Multiple F modes + no H confirmed | The approach doesn't work at this model capability level. Fall back to population-only search (ES paper style, adapted for sequential). Report the negative result. |
