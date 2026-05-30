"""Generate the roadside 3D-RCNN skeleton notebook (PyTorch3D + torchvision Mask
R-CNN). Run: python notebooks/_make_roadside_skeleton.py
Writes notebooks/roadside_3drcnn_skeleton.ipynb.

UNTESTED on this machine (no PyTorch3D / GPU). Best-effort from the canonical
PyTorch3D + torchvision APIs; verify on a Colab GPU runtime and report errors."""
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md("""# Roadside instance 3D-RCNN skeleton (PyTorch3D + Mask R-CNN)

The real pipeline (vs the single-cow toy):

```
street image -> Mask R-CNN (COCO-pretrained)  -> per-instance box + class + 2D mask
            -> per-class mesh bank (car/person/bike; swap ShapeNet)
            -> PyTorch3D RENDER-AND-COMPARE: fit each mesh's pose+scale so its
               silhouette matches the instance mask
            -> 3D-consistent instance mask + 3D pose per object
```

**Scope (honest):** this runs end-to-end *inference* on one real street image — real
detection + per-instance render-and-compare 3D fit. The **appearance-robust** win is that
each refined mask is the **mesh silhouette** (geometry), not learned pixels. The amortized
**training** (a pose encoder on Cityscapes) and **ShapeNet** per-class CADs are documented at
the end — that's the scale-up. **Untested by the author; run on a GPU runtime.**""")

md("## 0. Install PyTorch3D (torchvision is already on Colab)")
code(r'''import sys, subprocess, torch
def _have():
    try:
        import pytorch3d  # noqa
        return True
    except ModuleNotFoundError:
        return False
def _src():
    subprocess.run([sys.executable,'-m','pip','install','-q','ninja','fvcore','iopath'], check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q',
                    'git+https://github.com/facebookresearch/pytorch3d.git@stable'], check=True)
if not _have():
    if torch.version.cuda is None:
        _src()
    else:
        pyt=torch.__version__.split('+')[0].replace('.',''); cu=torch.version.cuda.replace('.','')
        url=f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{sys.version_info.minor}_cu{cu}_pyt{pyt}/download.html'
        subprocess.run([sys.executable,'-m','pip','install','-q','fvcore','iopath'], check=True)
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-cache-dir','-q','-f',url,'pytorch3d'])
        if not _have(): _src()
import pytorch3d; print('pytorch3d', pytorch3d.__version__)''')

md("## 1. Imports")
code(r'''import numpy as np, torch, requests
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, BlendParams)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', dev)''')

md("## 2. Input street image (set IMG_URL, or upload your own)")
code(r'''IMG_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg/640px-Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg'
try:
    img = Image.open(BytesIO(requests.get(IMG_URL, timeout=20).content)).convert('RGB')
except Exception as e:
    print('download failed (%s); upload an image instead' % e)
    from google.colab import files
    up = files.upload(); img = Image.open(next(iter(up))).convert('RGB')
img = img.resize((640, int(640 * img.height / img.width)))
plt.figure(figsize=(8,5)); plt.imshow(img); plt.axis('off'); plt.title('input'); plt.show()
print('image size', img.size)''')

md("## 3. Detection: COCO Mask R-CNN -> boxes + classes + 2D masks for road objects")
code(r'''weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
det = maskrcnn_resnet50_fpn(weights=weights).eval().to(dev)
pre = weights.transforms()
COCO = weights.meta['categories']                      # index -> name
ROAD = {'person','bicycle','car','motorcycle','bus','truck'}
PROXY = {'person':'person','bicycle':'bike','car':'car','motorcycle':'bike','bus':'car','truck':'car'}

x = pre(img).to(dev)
with torch.no_grad():
    out = det([x])[0]
keep = [(b, COCO[l], float(s), m[0]) for b, l, s, m in
        zip(out['boxes'], out['labels'], out['scores'], out['masks'])
        if float(s) > 0.7 and COCO[l] in ROAD]
print(f'{len(keep)} road instances:', [(c, round(s,2)) for _, c, s, _ in keep])

import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(9,6)); ax.imshow(img)
ov = np.zeros((*img.size[::-1], 4))
for b, c, s, m in keep:
    mm = (m.cpu().numpy() > 0.5)
    ov[mm] = [0, 1, 0, 0.4]
    x0,y0,x1,y1 = b.cpu().numpy()
    ax.add_patch(mpatches.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, ec='r', lw=2))
    ax.text(x0, y0-4, f'{c} {s:.2f}', color='r', fontsize=9)
ax.imshow(ov); ax.axis('off'); ax.set_title('Mask R-CNN detections'); plt.show()''')

md("## 4. Per-class mesh bank (PROXY ellipsoids; swap ShapeNet CADs for real shapes)")
code(r'''def proxy(scale):
    m = ico_sphere(3, dev); v = m.verts_packed() * torch.tensor(scale, device=dev)
    return Meshes(verts=[v], faces=[m.faces_packed()])
MESHES = {                                    # crude placeholders -> replace with ShapeNet
    'car':    proxy((1.8, 0.6, 0.9)),
    'person': proxy((0.4, 1.0, 0.35)),
    'bike':   proxy((1.2, 0.7, 0.25)),
}
print('mesh bank:', {k: m.verts_packed().shape[0] for k, m in MESHES.items()})''')

md("## 5. Per-instance render-and-compare: fit pose+scale so the mesh silhouette matches the mask")
code(r'''S = 128
blend = BlendParams(sigma=1e-4, gamma=1e-4)
raster = RasterizationSettings(image_size=S, blur_radius=np.log(1./1e-4-1.)*blend.sigma, faces_per_pixel=50)
def sil_render(mesh, campos):
    R, T = look_at_view_transform(eye=campos[None], at=((0,0,0),), up=((0,1,0),), device=dev)
    cam = FoVPerspectiveCameras(device=dev, R=R, T=T)
    r = MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                     shader=SoftSilhouetteShader(blend_params=blend))
    return r(mesh.scale_verts(1.0))[..., 3]            # (1,S,S)

def crop_mask(m, box):
    x0,y0,x1,y1 = [int(v) for v in box.cpu().numpy()]
    sub = m[y0:y1, x0:x1].float()[None,None]
    return F.interpolate(sub, size=(S,S), mode='bilinear', align_corners=False)[0,0]

def fit(mesh, target, iters=150):
    campos = torch.tensor([0.0, 0.0, 2.7], device=dev, requires_grad=True)
    logs = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([campos, logs], lr=0.05)
    for _ in range(iters):
        m = mesh.scale_verts(torch.exp(logs))
        sil = sil_render(m, campos)[0]
        loss = ((sil - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sil = sil_render(mesh.scale_verts(torch.exp(logs)), campos)[0]
    return sil.detach(), campos.detach(), float(torch.exp(logs))

fits = []
for b, c, s, m in keep:
    tgt = (crop_mask((m.cpu() > 0.5).float().to(dev), b) > 0.5).float()
    sil, campos, sc = fit(MESHES[PROXY[c]], tgt)
    inter = ((sil>0.5)&(tgt>0.5)).sum().float(); union=((sil>0.5)|(tgt>0.5)).sum().float()
    fits.append((c, tgt.cpu(), sil.cpu(), float(inter/(union+1e-6))))
print('mean fit IoU:', np.mean([f[3] for f in fits]) if fits else 0)''')

md("## 6. Visualize per-instance fits: detection mask vs render-and-compare silhouette")
code(r'''n = len(fits)
if n:
    fig, ax = plt.subplots(2, n, figsize=(2.4*n, 5))
    ax = np.atleast_2d(ax)
    for i,(c,tgt,sil,iou) in enumerate(fits):
        ax[0,i].imshow(tgt, cmap='gray'); ax[0,i].set_title(f'{c}\nMaskRCNN', fontsize=8); ax[0,i].axis('off')
        ax[1,i].imshow(sil, cmap='gray'); ax[1,i].set_title(f'mesh fit\nIoU={iou:.2f}', fontsize=8); ax[1,i].axis('off')
    plt.suptitle('per-instance: detection mask (top) vs render-and-compare mesh silhouette (bottom)')
    plt.tight_layout(); plt.show()''')

md("""## 7. The scale-up (training on real data) — what makes this solve roadside seg

This notebook is **inference**: detection + per-instance *test-time* render-and-compare fit.
The full system replaces the per-instance optimization with a **trained pose/shape encoder** and
real supervision:

1. **Per-class meshes -> ShapeNet CADs.** Replace the proxy ellipsoids with category meshes
   (car `02958343`, etc.); build a **PCA shape basis per class** (the paper's 10-dim TSDF space)
   so shape is a code, not fixed.
2. **Amortized encoder.** A ResNet on each RoI crop -> (class is from Mask R-CNN; predict
   **allocentric pose** + **shape code**). Train it by **render-and-compare** against the
   instance masks (and pose where available) — turning the 150-iter per-instance fit into one
   forward pass.
3. **Real masks: Cityscapes.** `gtFine` instance masks for car/person/bicycle/... as the
   render-and-compare targets. Loader sketch:
   ```python
   # cityscapes/leftImg8bit/<split>/<city>/*.png  +  gtFine/<split>/<city>/*_instanceIds.png
   # for each instance id: class = id // 1000 ; mask = (instanceIds == id)
   ```
4. **Appearance bridge.** The masks come from the **mesh silhouette** (geometry), so they are
   appearance-invariant -> the part that crosses sim->real. (Detection is the only learned-
   appearance stage; Mask R-CNN is real-trained, so it already generalizes.)

Reference: 3D-RCNN (CVPR2018, `papers/`), Mesh R-CNN, and the repo's
`run_r3dcnn_multiclass.py` (ResNet encoder + class head + per-class shape, the amortized core).""")

nb = {"cells": [{"cell_type": t, "metadata": {},
                 **({"source": s} if t == "markdown" else
                    {"source": s, "outputs": [], "execution_count": None})}
                for (t, s) in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "roadside_3drcnn_skeleton.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
