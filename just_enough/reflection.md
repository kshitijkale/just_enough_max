# Experiment Reflection

Run date: 2026-03-27
Experiments completed: 25 (ideas 0-24, plus idea_25 registered but not run)
Budget used: 24 of 30 experiments (not counting baseline)

---

## Results Summary

**Baseline:** val_bpb = 0.9966 (depth=8, dim=512, 50.3M params, batch=2^19, warmdown=0.5)
**Best:** val_bpb = 0.9839 (idea_23) — a **0.0127 improvement** (1.27%)

### Best Configuration (idea_23)
| Parameter | Baseline | Best | Change |
|---|---|---|---|
| DEPTH | 8 | 10 | +2 layers |
| ASPECT_RATIO | 64 | 51 | reduced to keep dim constant |
| model dim | 512 | 512 | unchanged |
| TOTAL_BATCH_SIZE | 2^19 (524K) | 2^18 (262K) | halved |
| WARMDOWN_RATIO | 0.5 | 0.7 | more annealing |
| FINAL_LR_FRAC | 0.0 | 0.05 | small residual LR |
| softcap | 15 | 10 | tighter logit regularization |
| num_params_M | 50.3 | 60.8 | +21% |
| num_steps | 932 | 1534 | +65% |

### Top 5 Ideas by val_bpb
| Rank | Idea | val_bpb | Key Change |
|---|---|---|---|
| 1 | idea_23 | 0.9839 | softcap=10 |
| 2 | idea_21 | 0.9841 | EMBEDDING_LR=0.3 |
| 3 | idea_18 | 0.9842 | FINAL_LR_FRAC=0.05 |
| 4 | idea_17 | 0.9843 | depth=12 (diminishing returns) |
| 5 | idea_16 | 0.9845 | depth=10, dim=512 |

---

## What Worked (in order of impact)

### 1. Halving batch size (idea_5): +0.008 bpb
The single biggest win. The baseline batch (2^19 = 524K tokens) was too large for this model — the GPU was doing unnecessary gradient accumulation (2 fwd/bwd per step). Halving to 2^18 eliminated grad accumulation, doubling optimizer steps from ~930 to ~1830 with identical throughput. Going even smaller (2^17) hurt due to GPU underutilization and gradient noise.

### 2. Increasing warmdown ratio (ideas 8-10): +0.002 bpb
Extending the LR annealing phase from 50% to 70% of training improved generalization. The warmdown acts as a long annealing period that polishes final weight quality. Tested the full series: 0.3 (bad), 0.5 (baseline), 0.6, 0.7 (optimal), 0.8 (overshot). Linear warmdown beat cosine.

### 3. Adding depth at constant width (ideas 15-17): +0.002 bpb
The insight that unlocked model size gains: increase depth while reducing ASPECT_RATIO to hold dim=512 constant. This avoids the throughput cliff from wider models. Depth=10 was optimal — depth=12 gave essentially zero additional gain while using much more VRAM (65GB).

### 4. Tighter softcap (idea_23): +0.0003 bpb
Reducing the logit softcap from 15 to 10 improved generalization. The cap acts as regularization — tighter prevents overconfident predictions. Going to 7 was too restrictive.

### 5. Small residual LR (idea_18): +0.0003 bpb
Setting FINAL_LR_FRAC=0.05 instead of 0.0 keeps the model learning in the last training steps instead of stalling at zero LR.

---

## What Didn't Work

| Idea | Change | Result | Why |
|---|---|---|---|
| idea_1 | depth=12 (+width) | 1.0145 | Model too big, only 382 steps |
| idea_2 | MATRIX_LR 0.04->0.06 | 0.9966 | LR wasn't the bottleneck |
| idea_3 | warmdown 0.5->0.3 | 0.9980 | Less annealing hurt generalization |
| idea_4 | depth=10 (+width) | 1.0056 | Model too big, only 569 steps |
| idea_6 | batch 2^17 | 0.9959 | GPU underutilized, noisy gradients |
| idea_7 | SwiGLU activation | 0.9893 | Slower per step, no convergence gain |
| idea_11 | 5% warmup | 0.9874 | Wasted peak-LR time |
| idea_12 | WD=0.4 | 0.9911 | Fought the optimizer too hard |
| idea_14 | Cosine warmdown | 0.9901 | Linear warmdown is better |
| idea_20 | HEAD_DIM=64 (8 heads) | 0.9921 | Per-head capacity matters more than head count |
| idea_22 | softcap=30 | 0.9872 | Removed useful regularization |
| idea_24 | softcap=7 | 0.9887 | Too restrictive, limited expressiveness |

---

## Key Beliefs (Final State)

1. **Batch size** (moderate): 2^18 is optimal for this model scale. Balances step count vs GPU utilization vs gradient quality.
2. **Warmdown** (moderate): 0.7 linear warmdown is optimal. The long annealing phase is critical for generalization.
3. **Depth at constant width** (moderate): More layers help at dim=512 up to depth=10, then flatten. The trick is adjusting ASPECT_RATIO to hold dim constant rather than letting both grow.
4. **Weight decay** (weak): 0.2 is optimal. Both directions (0.1, 0.4) regressed.
5. **Softcap** (weak): 10 is better than 15 or 30. Acts as beneficial regularization.

---

## Reflections on the Process

### What went well
- **The batch size discovery was high-value.** It came from reasoning about why loss was declining at the end — the model needed more steps, not bigger steps. This is exactly the kind of insight the belief system is designed to produce.
- **The depth-at-constant-width insight was creative.** Previous depth experiments failed because ASPECT_RATIO=64 coupled depth with width increases. Decoupling them by adjusting ASPECT_RATIO was a non-obvious move that came from analyzing WHY the depth experiments failed.
- **Systematic sweeps paid off.** The warmdown series (0.3/0.5/0.6/0.7/0.8) and softcap series (7/10/15/30) found clean optima. One-off experiments would have missed these.

### What could be improved
- **Too much time on small hyperparameter tweaks.** Ideas 11-14 (warmup, weight decay x2, cosine schedule) each yielded ~0 improvement. Could have been more selective.
- **Should have investigated the training dynamics earlier.** The switch.py analysis (after idea_14) revealed that 90% of loss improvement happens in the first 30% of training. This should have been done after idea_5.
- **Didn't explore architectural diversity enough.** Only tried MLP activation (SwiGLU) and head dim. Didn't test: window patterns, GQA (fewer KV heads), different MLP ratios, skip connection modifications, normalization changes.
- **The exploit mode was underused.** Only registered one exploit (idea_25) at the very end. Should have combined winning changes earlier and more aggressively.

### What I'd do differently with 30 more experiments
1. **Try depth=10 at dim=384** (even narrower, more layers at less width) — the depth-at-constant-width finding suggests exploring narrower+deeper.
2. **Explore GQA** (n_kv_head < n_head) — fewer KV heads saves compute per layer, allowing more layers.
3. **Try MLP expansion factor** — currently 4x. Could try 3x (faster per step, more steps) or 6x (more capacity per layer).
4. **Window pattern exploration** — "SSSL" is untested against alternatives.
5. **Combine all wins into a single exploit** and use it as a new baseline for a second round of exploration.

### On the framework itself
The belief diary was genuinely useful for avoiding dead ends. After establishing that MATRIX_LR=0.06 didn't help (idea_2), I didn't waste more experiments on LR tuning. After establishing the warmdown optimum, I moved on cleanly. Without the belief system, I might have kept circling the same hyperparameters.

The probe data was underutilized. I read it to confirm trends but rarely used it to generate new hypotheses. The switch.py investigation was the exception and it was valuable — should have done more of that.

The lineage graph was shallow (max depth 4: baseline -> idea_5 -> idea_9 -> idea_16 -> idea_18 -> idea_23). This matches the preregistration's prediction (H7). Most wins came from independent exploration, not deep refinement chains.
