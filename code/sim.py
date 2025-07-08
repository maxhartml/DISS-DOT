#!/usr/bin/env python3
"""
simple_fd_dot_simulator.py
==========================

A *compact* script that:

1. builds a 2-D circular mesh,
2. sprinkles the mesh with an optional absorbing blob,
3. slides a tiny probe (one light source, three detectors) across the top,
4. runs a 140 MHz frequency-domain forward solve for every probe position, and
5. saves amplitude and phase for every channel into an HDF5 file.

All it needs is the **full** NIRFASTer-FF repo on your PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

#   ╭────────────────────────────────────────────────╮
#   │  0. import nirfaster-FF (commit 0fb52ae)       │
#   ╰────────────────────────────────────────────────╯
import nirfasterff as ff

# ──────────────────────────────────────────────────────────────────────────────
# 1. Mesh utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_or_scale_disc(radius_mm: float = 50.0,
                       mua: float = 0.012,
                       musp: float = 1.0) -> ff.base.stndmesh:
    """
    Load the built-in circular mesh that ships with nirfaster-FF and
    scale it so its outer radius matches `radius_mm`.

    Parameters
    ----------
    radius_mm : desired disc radius after scaling (mm)
    mua, musp : background optical coefficients (mm⁻¹)

    Returns
    -------
    mesh : nirfaster `stndmesh` ready for simulation
    """
    repo = Path(ff.__file__).resolve().parents[1]
    mesh_base = repo / "meshes" / "standard" / "circle2000_86_stnd"

    mesh = ff.base.stndmesh()
    mesh.from_file(str(mesh_base))          # loads .elem/.node/.param/...

    # scale nodes if you asked for a radius other than the default ≈45 mm
    native_r = np.max(np.linalg.norm(mesh.nodes[:, :2], axis=1))
    if abs(native_r - radius_mm) > 1e-6:
        mesh.nodes[:, :2] *= radius_mm / native_r

    return mesh


def add_circular_blob(mesh: ff.base.stndmesh,
                      centre: Tuple[float, float],
                      radius: float,
                      mua_factor: float = 2.0) -> None:
    """Darken μa inside a little circle (simulated tumour / inclusion)."""
    node_dist = np.linalg.norm(mesh.nodes[:, :2] - centre, axis=1)
    inside_node = node_dist <= radius
    inside_elem = np.all(inside_node[mesh.elem], axis=1)
    mesh.prop.mua[inside_elem] *= mua_factor

# ──────────────────────────────────────────────────────────────────────────────
# 2. Probe helpers
# ──────────────────────────────────────────────────────────────────────────────
def rigid_probe(source_xy: Tuple[float, float],
                det_offsets: Tuple[float, float, float] = (20.0, 30.0, 40.0)
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays for one source + three right-hand detectors."""
    sx, sy = source_xy
    src = np.array([[sx, sy]], dtype=np.float32)
    det = np.array([[sx + d, sy] for d in det_offsets], dtype=np.float32)
    return src, det


def probe_positions(radius: float,
                    n: int = 16,
                    pitch: float = 4.0) -> List[Tuple[float, float]]:
    """
    Evenly spaced x-positions (y=0) so that the furthest detector (+40 mm) stays
    inside the circle.
    """
    half = (n - 1) * pitch / 2
    xs = np.linspace(-half, half, n)
    return [(x, 0.0) for x in xs if x + 40.0 <= radius - 1e-3]

# ──────────────────────────────────────────────────────────────────────────────
# 3. One phantom
# ──────────────────────────────────────────────────────────────────────────────
def make_phantom(radius: float,
                 fd_freq: float,
                 src_positions: List[Tuple[float, float]],
                 rng: np.random.Generator,
                 blob_chance: float = 0.7
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one phantom – returns (amplitude, phase) channel vectors."""
    mesh = load_or_scale_disc(radius)

    if rng.random() < blob_chance:
        add_circular_blob(mesh,
                          centre=rng.uniform(-15, 15, 2),
                          radius=rng.uniform(5, 12),
                          mua_factor=rng.uniform(1.5, 3.0))

    amps, phases = [], []

    for src_xy in src_positions:
        S, D = rigid_probe(src_xy)
        mesh.source = ff.base.optode(S)
        mesh.meas = ff.base.optode(D)

        mesh.touch_optodes()  # update mesh with new source/detector positions

        data, _ = mesh.femdata(fd_freq)         # forward solve
        m = data.bnd[:, 0]                      # 3 detectors

        amps.extend(np.abs(m))
        phases.extend(np.angle(m))

    return np.array(amps, dtype=np.float32), np.array(phases, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Dataset driver
# ──────────────────────────────────────────────────────────────────────────────
def bake_dataset(out_file: str = "fd2d_phantoms.h5",
                 n_samples: int = 50,
                 *,
                 radius: float = 50.0,
                 fd_freq: float = 140e6,
                 n_scan: int = 16,
                 pitch: float = 4.0,
                 blob_chance: float = 0.7,
                 seed: int = 42) -> None:
    """Generate `n_samples` phantoms and write them to an HDF5 file."""
    rng = np.random.default_rng(seed)
    src_row = probe_positions(radius, n_scan, pitch)
    n_channels = len(src_row) * 3

    with h5py.File(out_file, "w") as h5:
        d_amp = h5.create_dataset("amplitude", (n_samples, n_channels), "f4")
        d_phs = h5.create_dataset("phase",     (n_samples, n_channels), "f4")

        for idx in tqdm(range(n_samples), desc="simulating"):
            amp, phs = make_phantom(radius, fd_freq, src_row, rng, blob_chance)
            d_amp[idx] = amp
            d_phs[idx] = phs

    print(f"✔ wrote {n_samples} phantoms to {out_file}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bake_dataset(
        out_file="fd2d_phantoms.h5",
        n_samples=50,         # smoke-test; raise to thousands later
        radius=50.0,
        fd_freq=140e6,
        n_scan=16,
        pitch=4.0,
        blob_chance=0.7,
        seed=42,
    )
