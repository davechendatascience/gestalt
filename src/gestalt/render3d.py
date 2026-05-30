"""Procedural 3D multi-view renderer (pure numpy, no engine/GPU).

Each object is a parametric solid sampled to a surface point cloud + analytic
normals. A viewpoint is a rotation R(az, el); we rotate the object, project
ORTHOGRAPHICALLY (linear algebra: drop z, keep it for depth), back-face cull,
and splat with a painter's z-order into a clean, noise-free grayscale image
shaded by the camera-facing normal component.

The point: same object, KNOWN camera pose, many views -> exactly the multi-view
supervision that equivariant / canonicalisation methods want and rarely get.
No lighting model beyond the deterministic normal shading (we do NOT build a
light into the inductive bias).
"""
from __future__ import annotations
import numpy as np


# ---------------- parametric surfaces -> (points (M,3), normals (M,3)) -------

def _grid(nu, nv):
    u = (np.arange(nu) + 0.5) / nu
    v = (np.arange(nv) + 0.5) / nv
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return uu.ravel(), vv.ravel()


def _unit(N):
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)


def sphere(n=64):
    u, v = _grid(n, n)
    th, ph = 2 * np.pi * u, np.pi * v
    p = np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], 1)
    return p, p.copy()


def ellipsoid(a, b, c, n=64):
    p, _ = sphere(n)
    s = p * np.array([a, b, c])
    return s, _unit(s / np.array([a, b, c]) ** 2)


def box(sx, sy, sz, n=40):
    u, v = _grid(n, n)
    a, b = u * 2 - 1, v * 2 - 1
    Ps, Ns = [], []
    for axis in range(3):
        for sgn in (-1, 1):
            P = np.zeros((len(a), 3)); N = np.zeros_like(P)
            others = [i for i in range(3) if i != axis]
            P[:, axis] = sgn; P[:, others[0]] = a; P[:, others[1]] = b
            N[:, axis] = sgn
            Ps.append(P * np.array([sx, sy, sz])); Ns.append(N)
    return np.concatenate(Ps), np.concatenate(Ns)


def cylinder(r, h, n=64):
    u, v = _grid(n, n)
    th, y = 2 * np.pi * u, (v * 2 - 1) * h
    sP = np.stack([r * np.cos(th), y, r * np.sin(th)], 1)
    sN = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
    ur, ut = _grid(n // 2, n)
    rr, ct = np.sqrt(ur) * r, 2 * np.pi * ut
    cx, cz = rr * np.cos(ct), rr * np.sin(ct)
    tP = np.stack([cx, np.full(len(rr), h), cz], 1); tN = np.tile([0, 1, 0], (len(rr), 1))
    bP = np.stack([cx, np.full(len(rr), -h), cz], 1); bN = np.tile([0, -1, 0], (len(rr), 1))
    return np.concatenate([sP, tP, bP]), np.concatenate([sN, tN, bN]).astype(float)


def cone(r, h, n=64):
    u, v = _grid(n, n)
    th, t = 2 * np.pi * u, v
    rad, y = r * (1 - t), (t * 2 - 1) * h
    P = np.stack([rad * np.cos(th), y, rad * np.sin(th)], 1)
    slope = r / (2 * h)
    N = _unit(np.stack([np.cos(th), slope * np.ones_like(th), np.sin(th)], 1))
    ur, ut = _grid(n // 2, n)
    rr, ct = np.sqrt(ur) * r, 2 * np.pi * ut
    bP = np.stack([rr * np.cos(ct), np.full(len(rr), -h), rr * np.sin(ct)], 1)
    bN = np.tile([0, -1, 0], (len(rr), 1))
    return np.concatenate([P, bP]), np.concatenate([N, bN]).astype(float)


def torus(R, r, n=72):
    u, v = _grid(n, n)
    a, b = 2 * np.pi * u, 2 * np.pi * v
    rb = R + r * np.cos(b)
    P = np.stack([rb * np.cos(a), r * np.sin(b), rb * np.sin(a)], 1)
    ctr = np.stack([R * np.cos(a), np.zeros_like(a), R * np.sin(a)], 1)
    return P, _unit(P - ctr)


def _shift(po, d):
    p, n = po
    return p + np.array(d), n


def dumbbell():
    s1 = _shift(ellipsoid(.55, .55, .55, 40), [0, .85, 0])
    s2 = _shift(ellipsoid(.55, .55, .55, 40), [0, -.85, 0])
    c = cylinder(.22, .55, 40)
    return (np.concatenate([s1[0], s2[0], c[0]]), np.concatenate([s1[1], s2[1], c[1]]))


def lshape():
    b1 = box(1.0, 0.35, 0.5, 40)
    b2 = _shift(box(0.35, 0.9, 0.5, 40), [-0.65, 0.55, 0])
    return (np.concatenate([b1[0], b2[0]]), np.concatenate([b1[1], b2[1]]))


def capsule(r, h):
    c = cylinder(r, h, 56)
    t = _shift(ellipsoid(r, r, r, 36), [0, h, 0])
    b = _shift(ellipsoid(r, r, r, 36), [0, -h, 0])
    return (np.concatenate([c[0], t[0], b[0]]), np.concatenate([c[1], t[1], b[1]]))


def library():
    """The procedural 'asset library': name -> (points, normals)."""
    return {
        "sphere": sphere(), "ellipsoid_tall": ellipsoid(.7, 1.3, .7),
        "ellipsoid_flat": ellipsoid(1.3, .6, 1.0), "cube": box(.9, .9, .9),
        "slab": box(1.3, .35, .9), "rod": box(.32, 1.35, .32),
        "cylinder": cylinder(.7, 1.3), "disc": cylinder(1.2, .32),
        "cone": cone(.95, 1.4), "torus": torus(.9, .35),
        "torus_thin": torus(1.05, .2), "dumbbell": dumbbell(),
        "Lshape": lshape(), "capsule": capsule(.45, .9),
    }


# ---------------- camera + orthographic splat renderer ----------------------

def rotation(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Ry


def render(pts, normals, R, H=64, scale=0.40):
    """Orthographic, back-face-culled, painter's-order normal-shaded render."""
    P = pts @ R.T
    Ncz = (normals @ R.T)[:, 2]
    vis = Ncz > 0.02                                  # camera faces +Z; cull back faces
    P, Ncz = P[vis], Ncz[vis]
    u = np.clip(((P[:, 0] * scale + 0.5) * H).astype(int), 0, H - 1)
    v = np.clip(((0.5 - P[:, 1] * scale) * H).astype(int), 0, H - 1)
    order = np.argsort(P[:, 2])                        # far (small z) first; near overwrites
    img = np.zeros((H, H), np.float32)
    sh = np.clip(0.25 + 0.75 * Ncz, 0, 1)
    for dy in (-1, 0, 1):                              # 3x3 splat to fill pinholes
        for dx in (-1, 0, 1):
            vv = np.clip(v + dy, 0, H - 1); uu = np.clip(u + dx, 0, H - 1)
            img[vv[order], uu[order]] = sh[order]
    return img


def render_views(po, azimuths, elevations, H=64):
    """Render an object at the cartesian product of azimuths x elevations.
    Returns imgs (K,H,W), az (K,), el (K,), R (K,3,3)."""
    pts, nrm = po
    imgs, A, E, Rs = [], [], [], []
    for el in elevations:
        for az in azimuths:
            Rm = rotation(az, el)
            imgs.append(render(pts, nrm, Rm, H))
            A.append(az); E.append(el); Rs.append(Rm)
    return (np.stack(imgs), np.array(A, float), np.array(E, float), np.stack(Rs))
