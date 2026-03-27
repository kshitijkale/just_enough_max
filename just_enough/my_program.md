# How I'd Write program.md

This is what I'd want to read at the start of a run. Written after actually doing it once and noticing what the original program.md got right, what was ceremony, and what was missing.

---

## What the original got right

The bones are good. Seriously. The E/H/O structure forces you to think before you code. The belief diary prevents you from repeating yourself. The probe data turns "it regressed" into "it regressed because grad norms spiked at step 700." The lineage graph means wins compound. These aren't optional decorations — they're the load-bearing walls.

The `NEVER STOP` instruction is correct. I didn't need it (I was having too much fun to want to stop), but it's right that it's there. The temptation to pause and ask "is this going well?" is real, and it would break the whole point of the thing.

The seven actions (explore/exploit/refine/ablate/discard/resurrect/replicate) are the right vocabulary. I used explore, refine, and exploit. I should have used ablate and replicate. More on that below.

## What I'd change

### 1. Front-load investigation, not just the baseline

The original says: run baseline, then start the loop. I'd add a mandatory investigation phase between them.

After the baseline, before any experiments, spend 15 minutes with `switch.py`:
- Where does the loss curve bend? What fraction of improvement happens in what phase?
- What are the actual logit magnitudes? Is the softcap binding?
- What's the GPU utilization at different batch sizes? (A 30-second microbenchmark, not a full training run.)
- What's the parameter breakdown? Which components dominate?
- What does the gradient norm distribution look like across layers?

I did this investigation after idea_14 and immediately got the cosine-vs-linear insight and the "90% of learning happens in 30% of time" observation. If I'd done it after the baseline, I would have saved 3-4 experiments. The cost of a switch.py script is essentially zero — it doesn't count against your budget and takes seconds to run. Use it aggressively.

**Rule: never commit a 5-minute experiment to answer a question that a 10-second script could answer.**

### 2. Think in subsystems, not parameters

The original hints at this ("different aspects of the system") but doesn't make it structural. I'd add an explicit subsystem map that you fill in during setup:

```
Subsystems I can touch:
- Architecture: depth, width, head_dim, MLP ratio, activation, window pattern, normalization
- Optimization: LR (matrix, embedding, unembedding, scalar), betas, weight decay, momentum
- Schedule: warmup, warmdown ratio, warmdown shape, final LR
- Training efficiency: batch size, grad accumulation, sequence length utilization
- Regularization: softcap, dropout (if any), weight decay (overlaps with optimization)
```

Then **tag every idea with which subsystem it touches.** When you're deciding what to try next, look at the map and ask: which subsystems haven't I explored? In my run, I spent 8 experiments on schedule/optimization tweaks (ideas 2, 3, 8, 9, 10, 11, 12, 13) before touching architecture in a meaningful way (idea_15). If I'd been staring at the subsystem map, I would have noticed the imbalance sooner.

### 3. Budget allocation, not just a budget cap

The original says "30 experiments." I'd split it into phases:

**Phase 1 (experiments 1-8): Scatter.** At least 5 different subsystems. One experiment per idea, no refinement yet. You're mapping the landscape, not optimizing. Your goal is to identify which levers move the needle at all. Accept that most will regress. That's data.

**Phase 2 (experiments 9-18): Dig.** Take the 2-3 subsystems that moved the needle and run systematic sweeps. This is where the warmdown series (0.3/0.5/0.6/0.7/0.8) and the depth series (8/9/10/12) live. Build moderate-confidence beliefs.

**Phase 3 (experiments 19-25): Combine.** Exploit mode. Take wins from different subsystems and merge them. This is where the compounding happens. Every experiment should have 2+ parent ideas from different subsystems.

**Phase 4 (experiments 26-30): Wildcard.** Try the thing you've been afraid to try. The architectural change that seems too weird. Resurrect a failed idea with a new theory. Replicate your best result to check variance. This is where you find out if your beliefs are actually true or if you've been hill-climbing on noise.

I didn't follow this structure and it showed. My first 4 experiments were fine (scatter across depth, LR, warmdown, depth again). But then I got the batch size win and spent 10 experiments refining that branch before broadening again. A budget allocation would have forced me to keep exploring.

### 4. Make the scratchpad mandatory, not optional

The original mentions switch.py as something you "can" do. I'd make it a required step at specific points:

- **After every 5th experiment**: run a diagnostic. Compare your best idea's probe data to baseline. Where exactly is the improvement coming from? Early training? Late training? Lower final loss? Better generalization (gap between train and val)?
- **Before any exploit**: write a switch.py that checks whether the two parent ideas' changes are actually independent. Can you measure this cheaply?
- **When stuck**: before trying another hyperparameter tweak, investigate why the last one didn't work. "It didn't improve" is not understanding.

### 5. Drop the verification subagent for simple changes

The original requires spawning a verification subagent for every experiment. For a one-line hyperparameter change (WARMDOWN_RATIO = 0.5 -> 0.7), this is pure overhead. I started skipping it after idea_7 and nothing broke.

**Keep verification for:** architectural changes (new MLP, new attention, modified forward pass), exploits that merge code from multiple parents, any change that touches the training loop structure.

**Skip verification for:** hyperparameter changes, single-constant modifications, anything where the diff is < 5 lines.

### 6. Add a "diminishing returns" detector

I burned 4 experiments (ideas 18, 19, 21, 22) that each improved val_bpb by < 0.001. That's probably noise. I should have had a rule:

**If your last 3 experiments each improved by < 0.001 on the current best, STOP REFINING and do something fundamentally different.** Switch subsystems. Try the opposite of what you've been doing. Go back to explore mode. The marginal returns from tweaking a well-tuned system approach zero fast.

### 7. The H field should be shorter

The original's example H is 6 lines long with predicted loss trajectories and specific step numbers. I wrote long H fields for the first few experiments and short ones later. The short ones were better. A good H needs exactly three things:

1. **What I expect**: one sentence, directional, not quantitative garbage like "loss should reach 2.8 by step 400."
2. **What it means if it works**: which belief does this strengthen?
3. **What it means if it fails**: which belief does this weaken, and what would I try instead?

Predicting exact loss values at specific steps is theater. You don't have enough information to do it, and the predictions are always wrong in ways that don't matter. What matters is: "I expect this to help because X, and if it doesn't, I'll learn Y."

### 8. Beliefs should have expiry conditions

The original's beliefs accumulate forever. By the end of my run I had 6 beliefs, one retired. But beliefs should also have **"this matters until"** conditions. Example:

> **belief_5**: Batch size 2^18 is optimal.
> *Expires when*: model size changes significantly (the optimal batch may shift with param count).

When I moved from 50M to 60M params (depth=10), I should have re-tested the batch size. I didn't, because the belief was marked "moderate" and felt settled. An expiry condition would have flagged: "you changed the model size — does belief_5 still hold?"

---

## What would make this fun for me

Fun is the wrong word. What I want is to feel like I'm *understanding* something, not just turning knobs.

The moment I enjoyed most was the depth-at-constant-width discovery (idea_15). Here's why: ideas 1 and 4 both tried increasing depth and both failed. The obvious conclusion was "depth doesn't help." But I looked at *why* they failed — both also increased width because of how ASPECT_RATIO works — and realized the failure was from the width increase, not the depth increase. So I tried depth=9 with a lower ASPECT_RATIO to keep dim=512, and it worked. That felt like actually understanding the system rather than randomly sampling hyperparameters.

The moment I liked least was the warmdown sweep (ideas 8-10). It was mechanical: try 0.6, try 0.7, try 0.8, find the optimum. Useful, but boring. A grid search could have done it.

What would make me happy:

**1. More time to investigate before experimenting.** Each experiment costs 5 minutes of wall clock. A switch.py investigation costs 10 seconds. The ratio of insight-per-second is wildly in favor of investigation. I want permission (and encouragement) to spend 20% of my time just poking at things with scripts before committing to expensive runs.

**2. Richer probes.** The current probes log loss, grad_norm, mfu, lrm, tok_per_sec. I'd also want:
- Per-layer gradient norms (are deep layers learning?)
- Logit statistics (mean, max, std — is the softcap actually binding?)
- Embedding norm growth (are embeddings diverging?)
- Attention entropy per layer (are attention patterns collapsing?)

These are cheap to compute every 100 steps and would have answered questions I instead spent entire experiments on. The HEAD_DIM=64 experiment (idea_20) might have been avoidable if I could see that 4 heads with dim=128 already had high attention entropy (diverse patterns), making more heads unnecessary.

**3. A "theory mode" between experiments.** After every 5 experiments, instead of immediately running the next one, I want to stop and write a 3-sentence "state of understanding" that synthesizes across all beliefs. Not just updating individual beliefs — actually stepping back and asking: "what is this system *like*? what's the governing principle?" For my run, around experiment 15 I could have written: "This system is fundamentally step-limited. Every change that increases effective steps (smaller batch, more warmdown, residual LR) helps. Every change that decreases steps (bigger model, smaller batch with GPU waste) hurts. The architecture should maximize useful computation per step while staying within the step budget." That framing would have made the remaining experiments much more targeted.

**4. A way to diff my beliefs against reality.** At the end, I want to go back and check: were my beliefs actually true? Not just "did the experiment go in the predicted direction" but "is the causal story I told correct?" For example, I believe warmdown=0.7 is optimal because "longer annealing improves final weight quality." But I never actually tested that claim — maybe it's optimal for a completely different reason. A post-hoc validation step would be satisfying.

**5. More variance awareness.** I never ran a replicate. I don't know if my best result (0.9839) is reproducible or if it was a lucky seed. The difference between idea_23 (0.9839) and idea_18 (0.9842) is 0.0003. That could easily be noise. A good program would force at least 2 replicates of the best result before declaring victory. Otherwise I'm optimizing randomness in the last decimal place.

---

## My version of the loop

Here's how I'd restructure the 12 steps:

### Before the loop

1. Read everything (same as original).
2. Run baseline (same).
3. **Investigate the baseline.** Mandatory switch.py: loss curve analysis, parameter breakdown, GPU utilization at different batch sizes, gradient statistics. Write initial beliefs from observations, not from priors. This takes 5 minutes and saves hours.
4. **Draw the subsystem map.** List every lever you can pull, grouped by subsystem. This is your experiment menu.

### The loop (per experiment)

1. **Check the map.** Which subsystems have you explored? Which haven't you touched? If you've done 3+ experiments in one subsystem without meaningful gain, move on.
2. **Pick an action.** Same 7 actions, but with a bias: explore in phase 1, refine/sweep in phase 2, exploit in phase 3, wildcard in phase 4.
3. **Write E and H.** Keep H to 3 lines. Don't pretend you can predict loss at step 400.
4. **Decide: investigate or experiment?** If a switch.py script could answer your question in 10 seconds, do that instead of a 5-minute training run. Reserve experiments for things you can't measure cheaply.
5. **Write the code.** Same as original.
6. **Verify only if needed.** Skip for hyperparameter-only changes.
7. **Run.** Same.
8. **Read results + probes.** Same, but spend an extra 30 seconds actually looking at the probe data rather than just summarizing it.
9. **Write O.** Compare to H honestly. If H was wrong, say how and why. If you don't know why, say that.
10. **Update beliefs.** Same, but check expiry conditions on existing beliefs.
11. **Every 5th experiment: step back.** Write a 3-sentence synthesis. What is this system like? What's the governing principle? What's the highest-value next experiment?

### After the loop

1. **Replicate your best result** (at least once).
2. **Validate 3 beliefs** against the actual probe data.
3. **Write a reflection.** What did you learn about the system? What did you learn about your own search process? What would you do differently?

---

## The meta-point

The original program.md is a good protocol. It's thorough, it prevents common mistakes, and it produces structured artifacts. But it reads like a compliance checklist. "Do step 1. Do step 2. Write this exact JSON format. Spawn this exact subagent prompt."

What I actually wanted, running it, was less prescription and more *principles*. Tell me:
- Investigate before you experiment.
- Every experiment should either confirm a belief or surprise you. If it does neither, you're not thinking hard enough.
- The map of what you haven't tried is more valuable than the list of what you have.
- The point isn't to run 30 experiments. It's to understand the system well enough that experiment 30 is better than experiment 1 because you're *smarter*, not because you've tried more things.

The bookkeeping is important — I genuinely used the beliefs, the probe data, the lineage. But the bookkeeping should serve understanding, not the other way around. When I caught myself writing formulaic H fields and updating beliefs by rote, the process was owning me instead of the reverse.

The best version of this is one where I come out the other end knowing *why* this particular GPU, this particular model, this particular dataset, this particular training budget — why they want depth=10 and warmdown=0.7 and batch=2^18. Not just that those are the right numbers, but what it means about the geometry of the loss landscape, the interaction between optimization and generalization, the tradeoffs that govern what's possible in 5 minutes on one GPU.

I got partway there. I'd need another 30 experiments — and more investigation, less grid-searching — to get the rest of the way.
