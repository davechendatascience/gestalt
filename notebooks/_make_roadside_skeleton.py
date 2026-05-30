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

md("""## 4. Per-class mesh bank — real **ShapeNetCore** CADs (car/motorcycle) + proxy fallback

PyTorch3D's `ShapeNetCore` is a **loader over a local copy** — it does NOT download. Obtain
ShapeNetCore.v2 (free account at shapenet.org, or an HF mirror), then **mount Drive** or unzip
to `SHAPENET_DIR`. Synsets present: **car `02958343`, motorcycle `03790512`** — there is **no
person / bicycle** in ShapeNet, so those stay proxy ellipsoids. If the dir is absent every
class falls back to a proxy, so the notebook still runs end-to-end.""")
code(r'''import os
SHAPENET_DIR = os.environ.get('SHAPENET_DIR', '/content/ShapeNetCore.v2')
SYNSET = {'car': '02958343', 'bike': '03790512'}      # motorcycle stands in for bike
ELLIP  = {'car': (1.8, 0.6, 0.9), 'person': (0.4, 1.0, 0.35), 'bike': (1.2, 0.7, 0.25)}

def _norm(v): v = v - v.mean(0); return v / (v.abs().max() + 1e-8)   # center + unit-scale

def _proxy(name):
    m = ico_sphere(3, dev)
    return _norm(m.verts_packed()) * torch.tensor(ELLIP[name], device=dev), m.faces_packed()

def _shapenet(name):                                  # one canonical CAD per class (fixed shape)
    from pytorch3d.datasets import ShapeNetCore
    ds = ShapeNetCore(SHAPENET_DIR, synsets=[SYNSET[name]], version=2, load_textures=False)
    s = ds[0]
    return _norm(s['verts'].to(dev)), s['faces'].to(dev)

MV, MF = {}, {}                                       # per-class verts/faces (heterogeneous OK)
for name in ['car', 'person', 'bike']:
    try:
        if name not in SYNSET or not os.path.isdir(SHAPENET_DIR): raise FileNotFoundError
        MV[name], MF[name] = _shapenet(name); src = 'ShapeNet'
    except Exception as e:
        MV[name], MF[name] = _proxy(name); src = f'proxy ({type(e).__name__})'
    print(f'{name:7s} -> {src:20s} {MV[name].shape[0]} verts')
MESHES = {k: Meshes(verts=[MV[k]], faces=[MF[k]]) for k in MV}''')

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

md("""## 7. Data pull — real instance masks from **COCO** (car / person / bicycle)

COCO is downloadable without login and has per-instance segmentation masks for our road
classes (Cityscapes needs registration — swap its `gtFine` loader here later). We pull the
val2017 annotations, then download only the images we use, and build (crop, mask, class)
training samples.""")
code(r'''import os, zipfile, urllib.request
from PIL import Image
!pip install -q pycocotools
from pycocotools.coco import COCO

os.makedirs('coco', exist_ok=True)
ANN = 'coco/annotations/instances_val2017.json'
if not os.path.exists(ANN):
    z = 'coco/ann.zip'
    urllib.request.urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', z)
    with zipfile.ZipFile(z) as f: f.extractall('coco')
coco = COCO(ANN)

CLASSES = ['person', 'car', 'bicycle']
CID = {n: coco.getCatIds(catNms=[n])[0] for n in CLASSES}
N_PER = 150; S = 128
os.makedirs('coco/img', exist_ok=True)

crops, masks, labels = [], [], []
for ci, name in enumerate(CLASSES):
    aids = coco.getAnnIds(catIds=[CID[name]], iscrowd=False)
    got = 0
    for aid in aids:
        if got >= N_PER: break
        ann = coco.loadAnns([aid])[0]
        x, y, w, h = [int(v) for v in ann['bbox']]
        if w < 16 or h < 16: continue
        info = coco.loadImgs(ann['image_id'])[0]
        fn = 'coco/img/' + info['file_name']
        try:
            if not os.path.exists(fn):
                urllib.request.urlretrieve(info['coco_url'], fn)
            im = np.array(Image.open(fn).convert('RGB'))
        except Exception:
            continue
        m = coco.annToMask(ann)
        crop = np.array(Image.fromarray(im[y:y+h, x:x+w]).resize((S, S))) / 255.0
        mk = np.array(Image.fromarray((m[y:y+h, x:x+w] * 255).astype('uint8')).resize((S, S))) > 127
        crops.append(crop.astype('float32')); masks.append(mk.astype('float32')); labels.append(ci)
        got += 1
    print(f'{name}: {got} instances')
crops = torch.tensor(np.stack(crops)).permute(0, 3, 1, 2)      # (N,3,S,S)
masks = torch.tensor(np.stack(masks))                          # (N,S,S)
labels = torch.tensor(labels)
print('dataset:', crops.shape, masks.shape)''')

md("""## 8. Train the amortized pose encoder (ResNet) by **render-and-compare** on real masks

A ResNet maps each RoI crop -> (azimuth, elevation, log-scale). We render that class's mesh at
the predicted pose and compare its silhouette to the real instance mask. One forward pass
replaces the 150-iter per-instance optimization. (Class comes from the detector/COCO label.)""")
code(r'''import torch.nn as nn
enc = torchvision.models.resnet18(weights=None)
enc.fc = nn.Linear(512, 3)                          # az, el, log-scale
enc = enc.to(dev)

BANK = ['person', 'car', 'bike']                    # aligned to CLASSES = ['person','car','bicycle']
DIST = 2.7
blend = BlendParams(sigma=1e-4, gamma=1e-4)
raster = RasterizationSettings(image_size=S, blur_radius=np.log(1./1e-4-1.)*blend.sigma, faces_per_pixel=50)

def render_batch(az, el, logs, clsidx):                               # uses the real ShapeNet bank
    sc = torch.exp(logs)                                              # (B,) isotropic; mesh encodes shape
    keys = [BANK[int(c)] for c in clsidx]
    meshes = Meshes(verts=[MV[k] * sc[i] for i, k in enumerate(keys)],   # heterogeneous topology OK
                    faces=[MF[k] for k in keys])
    eye = torch.stack([DIST*torch.cos(el)*torch.sin(az), DIST*torch.sin(el),
                       DIST*torch.cos(el)*torch.cos(az)], 1)           # (B,3) on a sphere
    R, T = look_at_view_transform(eye=eye, at=((0,0,0),), up=((0,1,0),), device=dev)
    cam = FoVPerspectiveCameras(device=dev, R=R, T=T)
    r = MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                     shader=SoftSilhouetteShader(blend_params=blend))
    return r(meshes, cameras=cam)[..., 3]                             # (B,S,S)

opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
crops_d, masks_d, labels_d = crops.to(dev), masks.to(dev), labels.to(dev)
n, bs = len(crops_d), 16
for ep in range(15):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, bs):
        idx = perm[i:i+bs]
        out = enc(crops_d[idx])
        az = np.pi * torch.tanh(out[:, 0]); el = (np.pi/3) * torch.tanh(out[:, 1]); logs = out[:, 2]
        sil = render_batch(az, el, logs, labels_d[idx]).clamp(1e-4, 1-1e-4)
        loss = F.binary_cross_entropy(sil, masks_d[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f'epoch {ep+1}/15  loss={tot/(n//bs+1):.4f}')''')

md("## 9. Evaluate: predicted-pose mesh silhouette vs the real mask")
code(r'''enc.eval()
with torch.no_grad():
    out = enc(crops_d)
    az = np.pi*torch.tanh(out[:,0]); el=(np.pi/3)*torch.tanh(out[:,1]); logs=out[:,2]
    sil = render_batch(az, el, logs, labels_d)
    iou = (((sil>0.5)&(masks_d>0.5)).sum((-1,-2)).float() /
           (((sil>0.5)|(masks_d>0.5)).sum((-1,-2)).float()+1e-6))
print('mean fit IoU (mesh bank: ShapeNet car + proxy):', float(iou.mean()))
import numpy as _np
order = _np.argsort(-iou.cpu().numpy())[:6]            # show 6 best fits
fig, ax = plt.subplots(3, 6, figsize=(13, 6))
for j, k in enumerate(order):
    ax[0,j].imshow(crops[k].permute(1,2,0).numpy()); ax[0,j].set_title(CLASSES[labels[k]], fontsize=8)
    ax[1,j].imshow(masks[k].numpy(), cmap='gray'); ax[1,j].set_title('COCO mask', fontsize=8)
    ax[2,j].imshow(sil[k].cpu().numpy(), cmap='gray'); ax[2,j].set_title(f'mesh fit IoU={iou[k]:.2f}', fontsize=8)
for a in ax.ravel(): a.axis('off')
plt.suptitle('Amortized encoder: crop (top) | COCO mask (mid) | predicted-pose mesh silhouette (bottom)')
plt.tight_layout(); plt.show()''')

md("""## 10. To go further (the parts that lift the ceiling)

- **Per-class PCA shape basis (the paper's 10-dim TSDF space).** Cell 4 loads ONE canonical
  CAD per class. Load *many* (`ShapeNetCore(..., synsets=['02958343'])`, iterate), voxelize/TSDF
  each, PCA the intra-class variation -> a 10-dim basis, and add a **shape-code head** to the
  encoder so it predicts *pose + shape*, trained jointly by render-and-compare. This is the
  biggest single lever once real meshes are in.
- **person / bicycle aren't in ShapeNet.** Get them elsewhere (SMPL for person; a bicycle CAD
  from Objaverse/Free3D) or keep the proxy for those two.
- **Cityscapes** for the street domain: `gtFine/*_instanceIds.png`, `class = id // 1000`,
  `mask = (instanceIds == id)` — drop-in for the data-pull cell (also gated, free account).
- **Allocentric pose + the H-infinity RoI correction** (paper §5) for proper RoI equivariance.

Reference: 3D-RCNN (CVPR2018, `papers/`), Mesh R-CNN; the repo's `run_r3dcnn_multiclass.py`
is the same amortized core (ResNet -> class+pose, render-and-compare).""")

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
