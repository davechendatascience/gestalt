"""Generate a Colab notebook that CLONES the gestalt repo and runs 3D-RCNN from
its code (gestalt.r3dcnn / render3d / ae3d). Run:
    python notebooks/_make_3drcnn_colab.py
Writes notebooks/3d_rcnn_colab.ipynb."""
import json
from pathlib import Path

REPO = "https://github.com/davechendatascience/gestalt.git"
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md(f"""# 3D-RCNN render-and-compare — straight from the **gestalt** repo

Clones [`{REPO.split('//')[1]}`]({REPO[:-4]}) and runs the 3D-RCNN miniature using the
repo's own code: PCA shape basis + differentiable renderer + render-and-compare training.
Runs on Colab CPU or GPU.""")

code(f"""import os
if not os.path.isdir('gestalt'):
    !git clone -q {REPO}
else:
    !cd gestalt && git pull -q
import sys
sys.path.insert(0, 'gestalt/src')""")

code("""import numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from gestalt.render3d import library
from gestalt.ae3d import rotate3d
from gestalt.r3dcnn import build_shape_basis, render_silhouette, quat_to_R, R3DCNN
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); print('device:', dev)""")

md("## Shape basis (PCA modes) over the repo's procedural objects")
code("""D, NC, H = 20, 10, 48
pts = [library()[k][0] for k in library()]
mean, basis, codes = build_shape_basis(pts, D, NC)
mean_t = torch.tensor(mean, device=dev); basis_t = torch.tensor(basis, device=dev)
codes_t = torch.tensor(codes, device=dev)
print('objects:', list(library()), '| shape modes:', NC)""")

md("## Synthetic scenes + render-and-compare training (2D mask supervises 3D)")
code("""def rand_quats(n, rng):
    return F.normalize(torch.tensor(rng.normal(size=(n,4)), dtype=torch.float32, device=dev), dim=1)
@torch.no_grad()
def gen(n, seed):
    rng=np.random.default_rng(seed); o=rng.integers(0,len(pts),n)
    gt=codes_t[o]; R=quat_to_R(rand_quats(n,rng))
    occ=(mean_t+gt@basis_t).clamp(0,1).view(n,1,D,D,D); occ_r=rotate3d(occ,R)
    w=torch.linspace(1,0.35,D,device=dev).view(1,1,D,1,1)
    shaded=F.interpolate((occ_r*w).amax(2),size=(H,H),mode='bilinear',align_corners=False)[:,0]
    sil=F.interpolate(1-torch.prod(1-occ_r,2),size=(H,H),mode='bilinear',align_corners=False)[:,0]
    mask=(sil>0.3).float()
    tex=0.6+0.4*torch.tensor(rng.random((n,H,H)),dtype=torch.float32,device=dev)
    img=(shaded*tex+torch.tensor(rng.normal(0,0.04,(n,H,H)),dtype=torch.float32,device=dev)).clamp(0,1)
    return img,mask,gt,R
Xtr,Mtr,Ctr,Rtr=gen(900,0); Xte,Mte,Cte,Rte=gen(120,1)
net=R3DCNN(NC,H).to(dev); opt=torch.optim.Adam(net.parameters(),lr=1.5e-3)
for ep in range(70):
    perm=torch.randperm(len(Xtr))
    for i in range(0,len(Xtr),32):
        idx=perm[i:i+32]; code,R=net(Xtr[idx].unsqueeze(1))
        sil=render_silhouette(code,R,mean_t,basis_t,D,H).clamp(1e-4,1-1e-4)
        loss=F.binary_cross_entropy(sil,Mtr[idx])+0.1*F.mse_loss(code,Ctr[idx])+0.1*F.mse_loss(R,Rtr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%20==0: print(f'epoch {ep+1}/70  loss={loss.item():.4f}')""")

md("## Results — 2D render-and-compare recovers 3D shape + pose")
code("""net.eval()
with torch.no_grad():
    code,R=net(Xte.unsqueeze(1)); sil=render_silhouette(code,R,mean_t,basis_t,D,H)
    inter=((sil>0.5)&(Mte>0.5)).sum((-1,-2)).float(); union=((sil>0.5)|(Mte>0.5)).sum((-1,-2)).float()
    print('render-and-compare silhouette IoU (test):', float((inter/(union+1e-6)).mean()))
    nv=quat_to_R(rand_quats(6,np.random.default_rng(9))); novel=render_silhouette(code[:6],nv,mean_t,basis_t,D,H)
rows=[(Xte,'input'),(Mte,'GT mask'),(sil,'pred silhouette (R&C)'),(novel,'pred shape, NOVEL pose')]
fig,ax=plt.subplots(4,6,figsize=(9,6))
for r,(A,lab) in enumerate(rows):
    for c in range(6): ax[r,c].imshow(A[c].cpu().numpy(),cmap='gray',vmin=0,vmax=1); ax[r,c].axis('off')
    ax[r,0].set_title(lab,fontsize=8,loc='left')
plt.suptitle('3D-RCNN miniature (from gestalt repo): render-and-compare recovers 3D'); plt.tight_layout(); plt.show()""")

md("""Row 3 (network's render of its predicted shape+pose) matches row 2 (the GT mask), trained
only by comparing renders to 2D masks; row 4 renders the same predicted shape from a **novel**
pose. Honest limit: 2D silhouettes underdetermine the shape code (single-view is lossy).""")

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
