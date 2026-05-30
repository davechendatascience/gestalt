"""Fix 3d_mesh_latent_segmentation.ipynb: (1) PyTorch3D install cell (CPU-safe +
correct source-build command, no embedded quotes), (2) clean up the evaluation
cell (remove dead loop + redundant empty figures). Run: python fix_notebook.py"""
import json
from pathlib import Path

NB = Path(__file__).parent / "3d_mesh_latent_segmentation.ipynb"

INSTALL = r'''import sys, subprocess, torch
print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}')

def _have():
    try:
        import pytorch3d  # noqa
        return True
    except ModuleNotFoundError:
        return False

def _build_from_source():
    print('Building PyTorch3D from source (slow ~5 min; needs a GPU runtime for CUDA)...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ninja', 'fvcore', 'iopath'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'git+https://github.com/facebookresearch/pytorch3d.git@stable'], check=True)

if not _have():
    if torch.version.cuda is None:
        # CPU runtime: no matching prebuilt wheel -> source build (a GPU runtime is recommended)
        _build_from_source()
    else:
        pyt = torch.__version__.split('+')[0].replace('.', '')
        cu = torch.version.cuda.replace('.', '')
        ver = f'py3{sys.version_info.minor}_cu{cu}_pyt{pyt}'
        url = f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{ver}/download.html'
        print(f'Trying prebuilt wheel: {ver}')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'fvcore', 'iopath'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
                        '-q', '-f', url, 'pytorch3d'])
        if not _have():
            print('Prebuilt wheel not found for this torch/CUDA -> source build.')
            _build_from_source()
import pytorch3d
print(f'pytorch3d installed: {pytorch3d.__version__}')'''

EVAL = r'''# Project the learned 3D mesh latent -> 2D masks for every view; report IoU.
with torch.no_grad():
    final_mesh = src_mesh.offset_verts(deform_verts)

iou_scores = []
ncol = 4
nrow = (N_VIEWS + ncol - 1) // ncol
fig, big_axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
big_axes = big_axes.ravel()

for idx in range(N_VIEWS):
    cam = gt_cameras_list[idx]
    R, T = cam['R'].to(device), cam['T'].to(device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    sil_renderer = make_silhouette_renderer(cameras)
    with torch.no_grad():
        pred_sil = sil_renderer(final_mesh)[..., 3].squeeze(0).cpu().numpy()
    gt_sil_np = gt_silhouettes[idx].numpy()

    pred_bin = pred_sil > 0.5
    gt_bin = gt_sil_np > 0.5
    inter = float((pred_bin & gt_bin).sum())
    union = float((pred_bin | gt_bin).sum())
    iou_scores.append(inter / (union + 1e-6))

    ax = big_axes[idx]
    ax.imshow(np.concatenate([gt_sil_np, pred_sil], axis=1), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'azim={azimuths[idx]:.0f} deg  IoU={iou_scores[-1]:.3f}\n'
                 f'left: GT mask  |  right: 3D->2D pred', fontsize=8)
    ax.axis('off')
for k in range(N_VIEWS, len(big_axes)):
    big_axes[k].axis('off')

plt.suptitle('Learned 3D mesh latent projected to 2D masks  (left=GT, right=pred)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('segmentation_results.png', dpi=100, bbox_inches='tight')
plt.show()
print(f'\nMean IoU across {N_VIEWS} views: {np.mean(iou_scores):.4f}')
print('Per-view IoU:', [f'{s:.3f}' for s in iou_scores])'''


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    fixed = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if "pytorch3d.git@stable" in src:
            c["source"] = INSTALL; fixed.append("install")
        elif "big_axes" in src or "ax_row" in src:
            c["source"] = EVAL; fixed.append("eval")
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("fixed cells:", fixed)


if __name__ == "__main__":
    main()
