"""Generate a Colab notebook that CLONES the gestalt repo and runs 3D-RCNN on the
repo's OWN rendered dataset (render_camera images). Run:
    python notebooks/_make_3drcnn_colab.py
Writes notebooks/3d_rcnn_colab.ipynb."""
import json
from pathlib import Path

REPO = "https://github.com/davechendatascience/gestalt.git"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md(f"""# 3D-RCNN render-and-compare on the **gestalt** dataset

Clones [`{REPO.split('//')[1]}`]({REPO[:-4]}) and runs the 3D-RCNN miniature on the repo's own
rendered objects (`render_camera`, full `SO(3)`): a PCA shape basis (the shape *modes*), a
differentiable renderer, and a **render-and-compare** loss where the network's predicted shape
silhouette is fit to the rendered object's mask.

Honest setup: the *input* images come from the point-cloud renderer, while the model renders
*occupancy voxels* — a genuine model-vs-observation fit. There is **no pose label**; pose is
inferred purely by render-and-compare, anchored by the shape-code. Runs on Colab CPU/GPU.""")

code(f"""import os
if not os.path.isdir('gestalt'):
    !git clone -q {REPO}
else:
    !cd gestalt && git pull -q
import sys
sys.path.insert(0, 'gestalt/src')""")

code("""import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from gestalt.render3d import library, render_camera, rand_rotation
from gestalt.ae3d import rotate3d
from gestalt.r3dcnn import build_shape_basis, render_silhouette, quat_to_R, R3DCNN
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); print('device:', dev)""")

md("## Shape basis (PCA modes) over the repo's procedural objects")
code("""D, NC, H = 20, 10, 48
LIB = library(); NAMES = list(LIB); PTS = [LIB[k] for k in NAMES]   # (points, normals)
mean, basis, codes = build_shape_basis([p for p, _ in PTS], D, NC)
mean_t = torch.tensor(mean, device=dev); basis_t = torch.tensor(basis, device=dev)
codes_t = torch.tensor(codes, device=dev)
print('objects:', NAMES, '| shape modes:', NC)""")

md("## Dataset: the repo's RENDERED objects (render_camera, random SO(3))\\n"
   "Input = a shaded render; mask = its silhouette; label = which object (for the shape-code "
   "anchor). No pose label is stored.")
code("""def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, H, H), np.float32); masks = np.zeros((n, H, H), np.float32)
    oid = rng.integers(0, len(PTS), n)
    for i in range(n):
        pts, nrm = PTS[oid[i]]
        img = render_camera(pts, nrm, rand_rotation(rng), 4.0, 2.0, (0, 0), H, perspective=False)
        imgs[i] = img; masks[i] = (img > 0.05)
    return (torch.tensor(imgs, device=dev), torch.tensor(masks, device=dev),
            codes_t[torch.tensor(oid, device=dev)])
Xtr, Mtr, Ctr = make_dataset(900, 0)
Xte, Mte, Cte = make_dataset(160, 1)
print('train', tuple(Xtr.shape), 'test', tuple(Xte.shape))""")

md("## Render-and-compare training (2D silhouette + shape-code anchor; pose inferred)")
code("""net = R3DCNN(NC, H).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1.5e-3)
for ep in range(80):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 32):
        idx = perm[i:i+32]
        code, R = net(Xtr[idx].unsqueeze(1))
        sil = render_silhouette(code, R, mean_t, basis_t, D, H).clamp(1e-4, 1-1e-4)
        loss = F.binary_cross_entropy(sil, Mtr[idx]) + 0.3 * F.mse_loss(code, Ctr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 20 == 0:
        print(f'epoch {ep+1}/80  loss={loss.item():.4f}')""")

md("## Results — render-and-compare fits the model to the rendered objects")
code("""net.eval()
with torch.no_grad():
    code, R = net(Xte.unsqueeze(1)); sil = render_silhouette(code, R, mean_t, basis_t, D, H)
    inter = ((sil>0.5)&(Mte>0.5)).sum((-1,-2)).float(); union = ((sil>0.5)|(Mte>0.5)).sum((-1,-2)).float()
    print('render-and-compare silhouette IoU (test):', float((inter/(union+1e-6)).mean()))
    nv = quat_to_R(F.normalize(torch.tensor(np.random.default_rng(9).normal(size=(6,4)),
                                            dtype=torch.float32, device=dev), dim=1))
    novel = render_silhouette(code[:6], nv, mean_t, basis_t, D, H)
rows = [(Xte, 'input render'), (Mte, 'mask'), (sil, 'fitted silhouette (R&C)'), (novel, 'fitted shape, NOVEL pose')]
fig, ax = plt.subplots(4, 6, figsize=(9, 6))
for r, (A, lab) in enumerate(rows):
    for c in range(6): ax[r, c].imshow(A[c].cpu().numpy(), cmap='gray', vmin=0, vmax=1); ax[r, c].axis('off')
    ax[r, 0].set_title(lab, fontsize=8, loc='left')
plt.suptitle('3D-RCNN miniature on the gestalt dataset: render-and-compare'); plt.tight_layout(); plt.show()""")

md("""Row 3 (the network's fitted shape silhouette) is matched to row 2 (the rendered object's
mask) with **no pose label** — pose is found purely by render-and-compare, the shape-code anchor
keeps identity right. Row 4 renders the fitted shape from a **novel** pose. Honest limits: the
model's occupancy shape differs from the point-cloud render (real model-vs-observation gap), and a
single silhouette underdetermines 3D — so IoU is lower than the synthetic-data version, which is
the point.""")

nb = {"cells": [{"cell_type": t, "metadata": {},
                 **({"source": s} if t == "markdown" else
                    {"source": s, "outputs": [], "execution_count": None})}
                for (t, s) in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "3d_rcnn_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
