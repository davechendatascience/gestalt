# gestalt

> *Perceive the entity as a whole, through a process, and read out its structure symbolically.*

**gestalt** is a research project on learning **high-level, transfer-robust
representations** that are *relational and structural* rather than
*region-local and textural*. It grows out of three observations that kept
colliding:

1. **The sim2real / OOD gap lives in appearance, not geometry.** When object
   geometry is faithful (e.g. sim built from real reconstructions) but a model
   trained on sim collapses on real, the failure is in nuisance factors —
   texture, lighting, sensor, local render statistics. Region-local CNN
   features *entangle* those nuisances into the representation, so they don't
   survive the shift. (Empirically: global style transfer barely moves the gap
   — the entanglement is local and structural, not a global recolor.)

2. **What transfers is relational structure.** Shape, topology, part–whole
   relations, and the *timing/co-variation* between features are one level of
   abstraction above the features themselves — and that level is far more
   invariant to the nuisance group. This is the honest home of symbolic /
   geometric / equivariant methods: they don't beat a CNN *in-distribution*
   (we have the receipts), but **out-of-distribution, structure that refuses
   to overfit the nuisance factors can win.**

3. **Perception of an entity is a process, not a single feed-forward pass.**
   Sakana's Continuous Thought Machine reframes representation as *the timing
   relationships between units over an internal clock* — and on mazes it
   literally *traces the path* over internal ticks. "Observe the mask as an
   entity" is naturally a **process**: attend, trace, accumulate — not a static
   region convolution.

## Thesis

> High-level features that transfer = **(process) × (relation) × (symbol)**.
> Trace the entity over an internal timeline (process), represent it by how its
> parts co-vary (relation), and read that out as an interpretable, invariant
> rule (symbol). Accuracy in-distribution is not the goal; **invariance and
> transfer are.**

## What this is / isn't

- **Is:** a testbed + small architectures to find out whether process-based,
  relational, symbolic readouts transfer better than region-local features on
  controlled OOD/sim2real shifts — and where they don't.
- **Isn't:** a bid to beat SOTA accuracy in-distribution. We expect to *match,
  not crush*, on clean data (same verdict we reached for KAN/SR/equivariant
  nets). The bet is the OOD axis.

## Relationship to tessera

[`tessera`](../tessera) (gradient-free symbolic regression, `tessera.search.csp`)
is an **optional** dependency: it's the natural "symbol" stage — fitting a
compact, interpretable rule on top of relational/shape descriptors, exactly the
small-dimensional clean-target regime where csp shines (and unlike raw
perception, where it doesn't beat a CNN). The core of gestalt does not require
it.

## Status

Day 0 — brainstorming. See [`docs/brainstorm.md`](docs/brainstorm.md) for the
open directions, the honest risks, and the first experiment.
