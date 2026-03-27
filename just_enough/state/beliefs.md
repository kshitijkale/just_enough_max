# Belief Diary

## belief_1 [weak] confidence: weak
Depth can be increased IF width is held constant. Depth+width increases (ASPECT_RATIO=64) regress (idea_1, idea_4). Depth-only increase (ASPECT_RATIO=56, depth=9) at same dim=512 improves (idea_15). The key is controlling total param count, not just depth.
- Evidence for: idea_1 (depth=12+wider regressed), idea_4 (depth=10+wider regressed)
- Evidence against: idea_15 (depth=9 same width improved)

## belief_6 [moderate] confidence: moderate
Adding depth at constant width (dim=512) improves val_bpb up to depth=10, then flattens. Depth series at dim=512: 8=0.9867, 9=0.9859, 10=0.9845, 12=0.9843. Optimal depth is 10 for this setup.
- Evidence for: idea_15 (depth=9), idea_16 (depth=10 best), idea_17 (depth=12 marginal gain)
- Evidence against: (none)

## belief_2 [weak] confidence: weak
The baseline model is undertrained — more optimizer steps significantly improve val_bpb. The bottleneck is step count, not LR magnitude. Halving batch size gave 2x steps and 0.008 bpb improvement.
- Evidence for: idea_0 (loss still declining), idea_5 (halving batch -> 2x steps -> 0.008 improvement)
- Evidence against: idea_2 (higher LR alone didn't help — it's about step count, not step size)

## belief_5 [weak] confidence: weak
TOTAL_BATCH_SIZE 2^18 (262K) is optimal for this 50M param model. 2^19 is too large (baseline, fewer steps), 2^17 is too small (MFU drops, gradient noise increases). The sweet spot balances step count against GPU utilization and gradient quality.
- Evidence for: idea_5 (2^18 best val_bpb=0.9885), idea_6 (2^17 regressed to 0.9959 — too noisy)
- Evidence against: (none)

## belief_3 [RETIRED] confidence: retired
MATRIX_LR is already near-optimal at 0.04 for this model size. 1.5x increase produced nearly identical results. The undertraining signal (declining loss at end) is likely caused by the aggressive warmdown schedule (50% of budget) rather than insufficient LR.
- Evidence for: idea_2 (LR didn't help)
- Evidence against: idea_3 (reducing warmdown hurt — warmdown is beneficial, not wasteful)
- RETIRED: The warmdown hypothesis was wrong. 50% warmdown helps generalization.

## belief_4 [moderate] confidence: moderate
Optimal warmdown ratio is 0.7 for this setup. Series: 0.3=0.9980, 0.5=0.9885, 0.6=0.9880, 0.7=0.9867 (best), 0.8=0.9875 (overshot). 30% at peak LR + 70% annealing is the sweet spot.
- Evidence for: idea_3 (0.3 bad), idea_5 (0.5 ok), idea_8 (0.6 better), idea_9 (0.7 best), idea_10 (0.8 worse)
- Evidence against: (none)

---
## Changelog
- After idea_1: Added belief_1 (hypothesis). Depth=12 regressed due to undertraining (382 steps). Added belief_2 (hypothesis). Both idea_0 and idea_1 showed loss still declining at end.
- After idea_2: Added evidence against belief_2 (higher LR didn't help). Added belief_3 (hypothesis). LR near-optimal, warmdown schedule may be the real bottleneck.
- After idea_3: Retired belief_3 (warmdown reduction hurt). Added belief_4 (hypothesis). 50% warmdown is beneficial for generalization.
- After idea_4: Strengthened belief_1 -> weak. Even depth=10 (86M params) regressed with only 569 steps.
- After idea_5: Strengthened belief_2 -> weak. Halving batch gave 2x steps and 0.008 bpb improvement. Added belief_5 (weak). Batch size was too large.
- After idea_6: Updated belief_5 — 2^18 is optimal, 2^17 is too small (MFU drops, gradient noise). Established batch size sweet spot.
- After idea_7: ReLU^2 outperforms SwiGLU at this scale. SwiGLU added overhead (fewer steps) without better convergence.
- After idea_8: Strengthened belief_4 -> weak. Warmdown=0.6 improved on 0.5 (0.9880 vs 0.9885). New best val_bpb=0.9880.
- After idea_9: Warmdown=0.7 further improved to 0.9867. New best.
- After idea_10: Warmdown=0.8 overshot (0.9875). Belief_4 -> moderate. Optimal warmdown is 0.7.
- After idea_11: Warmup=0.05 didn't help (0.9874 vs 0.9867). No warmup is optimal.
- After idea_12: WD=0.4 regressed (0.9911). Higher WD fights optimizer.
- After idea_13: WD=0.1 also worse (0.9872). WD=0.2 confirmed optimal.
- After idea_14: Cosine warmdown worse than linear (0.9901 vs 0.9867).
- After idea_15: Depth=9 at constant width (dim=512) new best 0.9859! Updated belief_1, added belief_6.
- After idea_16: Depth=10 new best 0.9845. Strengthened belief_6.
- After idea_17: Depth=12 barely improved (0.9843). Depth scaling flattened. Optimal depth=10.
