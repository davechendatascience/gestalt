# gestalt — brainstorm (day 0)

Living document. Captures the threads, the candidate directions, the honest
risks, and what the first falsifiable experiment should be. Nothing here is
committed; it's a thinking surface.

## The throughline

Everything points at the same triangle:

```
            PROCESS  (trace the entity over an internal clock — CTM)
              /  \
             /    \
   RELATION ------ SYMBOL
  (how parts      (interpretable invariant rule — csp/SR)
   co-vary —
   synchronization)
```

The bet: a representation that is **process-built, relational, and symbolically
read out** is more invariant to nuisance shift than a region-local feature
stack, because each leg of the triangle strips appearance:
- process → integrates evidence over a trajectory instead of a fixed window;
- relation → "which parts co-vary" survives shifts that change *what each part
  looks like*;
- symbol → a sparse rule can't memorize per-pixel render statistics.

## Candidate directions

### A. Trace-and-sync: CTM-style entity tracer → relational descriptor → csp
The flagship. Given a candidate mask/silhouette (from SAM on real data, or
ground-truth in sim), run a small **tick-recurrent attention loop** that walks
the entity (boundary / interior) over T internal steps. Read out a
**synchronization-style relational descriptor** (how the traced features
co-vary over the trajectory), then classify with csp.
- Directly mechanizes "observe the mask as an entity, not region locality."
- Tests the full triangle.
- Risk: the tracer itself is learned and could re-entangle appearance; must
  feed it *shape* (binary mask / boundary), not RGB, to keep it honest.

### B. Symbolic Neuron-Level Models (interpretable CTM)
CTM gives each neuron a private depth-1 MLP over its own pre-activation history.
**Replace that MLP with a csp-discovered symbolic temporal rule.** Result: a CTM
whose per-neuron update laws are readable equations, and whose synchronization
matrix is a structured object you can fit further symbolic rules on.
- This is tessera's sweet spot (small, clean-target, per-unit temporal fns).
- Pure-research; payoff is interpretability of "thinking as a process," and a
  possible regularization / OOD benefit from the symbolic bottleneck.
- Risk: csp NLMs may underperform MLP NLMs in-distribution (acceptable if they
  transfer better or reveal structure).

### C. Invariant shape-entity descriptors + csp (the strong baseline)
No process: just compute classical appearance-blind descriptors on the mask —
Fourier boundary descriptors, Hu/Zernike moments, medial-axis stats, a
persistence-diagram summary — and classify with csp. This is the **control**
that tells us how much the *process* (A) actually adds over a static structural
readout.
- Cheap, must-have baseline. If C already closes most of the gap, A/B need to
  justify their complexity.

### D. Analysis-by-synthesis shape programs (long horizon)
Represent each class as a compact **symbolic shape program** (contour grammar /
parametric program) and recognize by finding the program that reconstructs the
silhouette. Appearance-invariant by construction, compositional, "high-level"
in the strongest sense. csp searches the program space.
- Highest ceiling, highest risk; slow/brittle on complex real shapes. Park it
  until A–C give signal.

## The honest risks (write them down so we don't fool ourselves)

1. **Chicken-and-egg.** To read out an entity you need the entity. On real data
   that means a strong class-agnostic segmenter (SAM/SAM2) to *propose* the
   mask; gestalt then works on the proposed shape. We are improving
   **recognition/transfer of a given entity**, not solving cross-domain
   localization. Be explicit about which stage we claim.
2. **Shape-only ceiling.** Many classes are disambiguated by appearance, not
   silhouette (cup vs. can). A pure-shape representation may *transfer* far
   better in relative terms yet **cap lower in absolute** than an appearance
   model with a little real data. Measure the ceiling early.
3. **Topology is coarse.** Most rigid objects are genus-0 blobs; persistence
   helps only for structured shapes. Don't oversell TDA.
4. **"Process" can cheat.** A learned tracer fed RGB will just relearn
   appearance. Keep its input structural.
5. **Match-not-beat is the likely honest outcome in-distribution.** That's
   fine — the claim must be on the OOD/transfer axis, with a controlled shift.

## First experiment (falsifiable, cheap, synthetic)

Before touching the real segmentation data, build a **controlled shape-transfer
testbed** so we can measure the *transfer axis* cleanly:

- **Data:** procedurally generated 2D shapes from K classes (distinguishable by
  silhouette — e.g. polygons by #sides, star/gear/blob families). Render each
  with a "sim" appearance (one texture/lighting/noise regime) and a disjoint
  "real" appearance (different textures/lighting/noise, *same geometry*). This
  reproduces your sim2real setup in miniature with a knob on the gap.
- **Models:**
  - (baseline) small CNN on the RGB shape — expect high sim-val, low real-test
    (reproduce the collapse in miniature);
  - (C) descriptors-on-mask + csp;
  - (A) trace-and-sync + csp;
- **Metric:** the gap `sim-val − real-test`, and absolute real-test accuracy.
- **Win condition:** A and/or C have a *dramatically smaller* gap than the CNN,
  even if absolute accuracy is capped. If so, the triangle is real and we scale
  toward SAM-masks on actual data. If not, we learn that cheaply.

This testbed is the keystone: it lets us iterate on architecture against the
*transfer* number directly, without the cost/noise of the full pipeline.

## Day-0 result (control C + CNN baseline)

`experiments/run_transfer_testbed.py`, 6 silhouette-separable classes, sim/real
disjoint appearance, 3600 train / 1200 val / 1200 real-test:

| representation | sim-val | real-test | gap |
|---|---|---|---|
| CNN on RGB (region-local) | 0.800 | 0.176 | **+0.624** |
| descriptors+logistic (invariant) | 1.000 | 1.000 | **+0.000** |

- The collapse is **real and reproduced**: the CNN falls to ~chance (1/6) on
  real — pure texture reliance, zero transfer.
- Pure structure transfers with **zero gap** and hits 100% — because these
  classes are perfectly silhouette-separable.

**The catch (and the next move):** both anchors are pinned at the extremes
(CNN ≈ chance, descriptors = 100%), so there is **no headroom to demonstrate
that direction A/B adds anything.** To make this a real instrument we need a
**shape-separability knob**: introduce class pairs that are *silhouette-
ambiguous but appearance-distinct* (the cup-vs-can case), dropping the
descriptor ceiling below 100%. Then the live question becomes measurable:
*can a process/relational model that is allowed to use appearance in a
transfer-robust way beat the pure-shape ceiling without inheriting the CNN's
gap?* That gap-between-the-anchors is the whole game.

## Open questions for us to settle

- Start with the **flagship A** (more exciting, more risk) or the **control C**
  (cheap, tells us the baseline transfer) — or build the testbed first and run
  C immediately as the yardstick?
- Backbone for the "symbol" stage: tessera/csp from the start, or a plain
  linear/logistic readout first to isolate the *representation* effect from the
  *readout* effect?
- Real data: can we get a small dump of sim+real masks from the segmentation
  project to validate the testbed's conclusions, or stay fully synthetic until
  the idea proves out?
- Compute/framework: numpy+torch CPU-first (like tessera), or JAX for the
  tick-recurrence?
