"""Investigation: look at the training dynamics more carefully.
Compare the loss curve at different points for idea_9 (best) to understand
where the model is spending its training budget."""

import json

# Read idea_9 probes
probes = []
with open("state/probes/idea_9_probes.jsonl") as f:
    for line in f:
        probes.append(json.loads(line))

print(f"Total probes: {len(probes)}")
print(f"Total steps: {probes[-1]['step']}")
print()

# Look at loss improvement rate across the training
print("=== Loss improvement rate (per 100 steps) ===")
for i in range(0, len(probes)-10, 10):
    s1, s2 = probes[i], probes[min(i+10, len(probes)-1)]
    delta = s2['smooth_loss'] - s1['smooth_loss']
    print(f"Steps {s1['step']:5d}-{s2['step']:5d}: smooth_loss {s1['smooth_loss']:.4f} -> {s2['smooth_loss']:.4f} (delta={delta:+.4f}), lrm={s1['lrm']:.4f}, grad_norm={s1['grad_norm']:.6f}")

print()
print("=== Key observations ===")
# Where does warmdown start?
for p in probes:
    if p['lrm'] < 1.0:
        print(f"Warmdown starts at step {p['step']} (lrm={p['lrm']:.4f})")
        break

# What fraction of total loss improvement happens in warmdown?
warmdown_start = None
for p in probes:
    if p['lrm'] < 1.0:
        warmdown_start = p
        break

if warmdown_start:
    peak_lr_improvement = probes[1]['smooth_loss'] - warmdown_start['smooth_loss']
    warmdown_improvement = warmdown_start['smooth_loss'] - probes[-1]['smooth_loss']
    total = peak_lr_improvement + warmdown_improvement
    print(f"Loss improvement during peak LR: {peak_lr_improvement:.4f} ({100*peak_lr_improvement/total:.1f}%)")
    print(f"Loss improvement during warmdown: {warmdown_improvement:.4f} ({100*warmdown_improvement/total:.1f}%)")
    print(f"Peak LR phase: steps 0-{warmdown_start['step']}")
    print(f"Warmdown phase: steps {warmdown_start['step']}-{probes[-1]['step']}")

# Look at grad norms vs lrm
print()
print("=== Grad norm vs LR multiplier ===")
for p in probes[::20]:
    print(f"Step {p['step']:5d}: lrm={p['lrm']:.4f}, grad_norm={p['grad_norm']:.6f}, tok/sec={p['tok_per_sec']:,}")
