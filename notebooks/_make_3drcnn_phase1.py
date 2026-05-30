"""Generate the Phase-1 Colab notebook: verify a PyTorch3D differentiable
silhouette RENDER-AND-COMPARE on Colab (the renderer core of 3D-RCNN), before the
Monday Pascal3D+ build. Run:  python notebooks/_make_3drcnn_phase1.py
Writes notebooks/3d_rcnn_pytorch3d_phase1.ipynb.

NOTE: the PyTorch3D code is best-effort from the canonical silhouette-fitting API;
it is UNTESTED on this machine (no GPU / no PyTorch3D). Run on Colab and report
errors so we iterate."""
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# 3D-RCNN — Phase 1: verify the PyTorch3D **render-and-compare** core (Colab)

3D-RCNN learns instance 3D shape+pose with a *render-and-compare* loss: render the predicted
shape at the predicted pose, compare the silhouette to the 2D mask. This notebook retires the one
technical risk we can on Colab today — a **differentiable silhouette renderer** (PyTorch3D) that
we can backprop a pose/shape through.

**What this does:** render a target silhouette of an asymmetric mesh at a known pose, then recover
that pose purely by minimizing a silhouette render-and-compare loss. If the loss drops and the
rendered silhouette converges to the target, the core works and we build Pascal3D+ on top Monday.

> The PyTorch3D code is best-effort from the canonical API and **untested by the author** (no
> GPU/PyTorch3D locally). Please run and report any install/API errors. Use a **GPU runtime**.""")

md("## 0. Install PyTorch3D (Colab). If the wheel line fails, use the source-build fallback.")
code("""import sys, torch, subprocess
def have(m):
    try: __import__(m); return True
    except Exception: return False
if not have('pytorch3d'):
    !pip -q install fvcore iopath
    ver = torch.__version__.split('+')[0].replace('.', '')
    cu  = ('cu' + torch.version.cuda.replace('.', '')) if torch.cuda.is_available() else 'cpu'
    url = f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{sys.version_info.minor}_{cu}_pyt{ver}/download.html"
    print('trying prebuilt wheel:', url)
    r = subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-cache-dir','-f',url,'pytorch3d'])
    if r.returncode != 0 or not have('pytorch3d'):
        print('prebuilt wheel failed -> building from source (slow, needs GPU runtime)')
        !pip -q install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
import pytorch3d; print('pytorch3d', pytorch3d.__version__)""")

md("## 1. Imports + device")
code("""import numpy as np, torch
import matplotlib.pyplot as plt
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, BlendParams)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', dev)""")

md("## 2. An ASYMMETRIC mesh (so the silhouette actually depends on pose)")
code("""m = ico_sphere(4, dev)
v = m.verts_packed().clone()
v = v * torch.tensor([1.8, 0.6, 1.0], device=dev)            # stretch -> ellipsoid
v[:, 2] = v[:, 2] + 0.4 * torch.exp(-(v[:, 0]**2 + v[:, 1]**2))   # a bump -> asymmetric
mesh = Meshes(verts=[v], faces=[m.faces_packed()])
print('verts', v.shape[0], 'faces', m.faces_packed().shape[0])""")

md("## 3. A differentiable SILHOUETTE renderer (PyTorch3D)")
code("""blend = BlendParams(sigma=1e-4, gamma=1e-4)
raster = RasterizationSettings(image_size=128,
    blur_radius=np.log(1.0/1e-4 - 1.0) * blend.sigma, faces_per_pixel=50)
def silhouette_renderer(cameras):
    return MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
                        shader=SoftSilhouetteShader(blend_params=blend))
def render_sil(dist, elev, azim):
    R, T = look_at_view_transform(dist, elev, azim, device=dev)
    cam = FoVPerspectiveCameras(device=dev, R=R, T=T)
    return silhouette_renderer(cam)(mesh, cameras=cam)[..., 3]    # alpha = silhouette""")

md("## 4. Target silhouette at a known pose")
code("""with torch.no_grad():
    target = render_sil(torch.tensor([2.7], device=dev),
                        torch.tensor([35.0], device=dev), torch.tensor([60.0], device=dev))[0]
plt.imshow(target.cpu().numpy(), cmap='gray'); plt.title('target silhouette'); plt.axis('off'); plt.show()""")

md("## 5. RENDER-AND-COMPARE: recover the pose by silhouette matching only")
code("""p = torch.tensor([3.2, 5.0, 5.0], device=dev, requires_grad=True)   # dist, elev, azim guess
opt = torch.optim.Adam([p], lr=0.07)
losses = []
for it in range(220):
    sil = render_sil(p[0:1], p[1:2], p[2:3])[0]
    loss = ((sil - target) ** 2).mean()                # the render-and-compare loss
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if (it+1) % 40 == 0: print(f'iter {it+1}: loss={loss.item():.4f}  pose={p.detach().cpu().numpy().round(1)}')
print('recovered pose (dist,elev,azim):', p.detach().cpu().numpy().round(2), ' target = [2.7, 35, 60]')""")

md("## 6. Did it converge?")
code("""with torch.no_grad(): final = render_sil(p[0:1], p[1:2], p[2:3])[0]
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(target.cpu().numpy(), cmap='gray'); ax[0].set_title('target'); ax[0].axis('off')
ax[1].imshow(final.cpu().numpy(), cmap='gray'); ax[1].set_title('render-and-compare fit'); ax[1].axis('off')
ax[2].plot(losses); ax[2].set_title('loss'); ax[2].set_xlabel('iter')
plt.tight_layout(); plt.show()""")

md("""## Phase 2 (Monday, GPU + disk): Pascal3D+ build on top of this core

With the differentiable renderer verified, the remaining pieces are data + a net:
1. **Download** Pascal3D+ (`PASCAL3D+_release1.1`, ~7.5 GB) and pick a category (e.g. `car`).
2. **Parse** the `.mat` annotations -> per-instance (CAD index, azimuth/elevation/distance) and the
   GT segmentation masks; **load the category's CAD meshes**.
3. **Shape basis:** PCA over the category's CAD models -> a low-dim shape code (the shape modes);
   the predicted shape = mean + code . basis (a deformable mesh).
4. **Net:** a backbone on the image RoI -> heads for (shape code, allocentric pose), trained with
   the **render-and-compare** loss above (predicted shape+pose -> silhouette vs the GT mask) +
   direct shape/pose supervision where available.
5. Train on GPU. (Reference structure: the unofficial repo shahabty/3D-RCNN.)

That swaps the single asymmetric mesh here for a CAD-derived deformable shape basis and real
images/masks — the core render-and-compare loop is the same one verified above.""")

nb = {"cells": [{"cell_type": t, "metadata": {},
                 **({"source": s} if t == "markdown" else
                    {"source": s, "outputs": [], "execution_count": None})}
                for (t, s) in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "3d_rcnn_pytorch3d_phase1.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
