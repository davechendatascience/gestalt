# Recognition under viewpoint: what we found

A rigorous sweep of "can structure beat a CNN at recognizing objects under
viewpoint change," run on a procedural 14-object 3D testbed. The short version:
**structure wins decisively only when it gets better input or the complete model;
on a single 2D view, a CNN is at the information ceiling and structure ties or
loses.** The leverage is upstream of the classifier.

## The evidence

All on the same 14 objects, held-out poses (chance = 0.071).

| method | input | result | note |
|---|---|---|---|
| analytic invariant (D2 / SH power spectrum) | **3D point cloud** | **1.000** | closed-form, no training; `SO(3)` quotiented analytically |
| multi-view feature pool | **4 views** | **0.64** > 0.47 | more input information helps |
| STN canonicalization | single, 2D-rot extrapolation | **beats CNN** (0.88 vs 0.62) | removes a *present* nuisance it can't memorize |
| **generative match** (search pose over a bank) | single view + **exact models** | **0.85** > CNN 0.51 | the Bayesian posterior `max_g sim(x, render(o,g))` |
| CNN classifier | single view | 0.51–0.76 | the baseline; near the single-view ceiling |
| equivariant AE code | single view | 0.23 | great at synthesis, not discrimination |
| 3D render auxiliary | single view | −0.01 | auxiliary supervision ≠ input information |
| ungrounded 3D voxel fusion | multi-view | 0.11 | voxel frame meaningless without grounding |
| invariant-signature distillation | single view | 0.16 | target isn't a function of one lossy view |
| learned-latent pose-aware matcher | single view | 0.75 ≈ CNN | amortizing the bank back to a single view → CNN |

## The principle

> **Single-view recognition is input-limited.** A single 2D projection is lossy
> (hidden back) and breaks the clean `SO(3)` action (projection), so no
> restructuring of the *objective* — invariant features, equivariant codes, 3D
> auxiliaries, signature distillation, learned latents — beats a CE-trained CNN.
> Structure pays only where it changes the *input or the model*: 3D / multi-view
> (clean group action → analytic invariant = 1.00), the complete model + dense
> pose search (generative match = 0.85), or a *present* nuisance the CNN can't
> memorize (2D-rotation extrapolation → STN wins).

Two information losses bound single-view identity (Grenander pattern theory): the
**unknown pose** (marginalize `g`) and **self-occlusion** (the likelihood only
sees the visible surface). The generative posterior *handles* both (0.85, not
1.0); the invariant assumed them away (full observation) and the CNN amortizes
them from data.

## Deployable conclusion

- **Pose is free, no labels.** Generative matching searches pose over a sim-
  rendered bank; you never annotate or learn pose.
- **Appearance is the only real gap.** Pixel matching collapses sim→real
  (0.85→0.58); matching on **structure** (silhouette) is invariant (0.75→0.75),
  at a lower ceiling. Depth/normals/domain-randomized features are the upgrade.
- **For segmentation:** the models help via **render-and-compare** — the mask is
  the fitted model's silhouette (appearance-invariant → crosses sim→real), or a
  class-agnostic segmenter (SAM) + structural model-matching for labels. See
  `experiments/run_render_compare_seg.py`.

## Provenance

`run_invariant_signature.py`, `run_multiview_compare.py`, `run_viewpoint_mnist.py`,
`run_generative_recognition.py`, `run_generative_real.py`, `run_equivariant_ae.py`,
`run_multitask_compare.py`, `run_signature_distill.py`, `run_latent_classifier.py`.
