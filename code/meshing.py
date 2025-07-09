#!/usr/bin/env python3
"""
Generates a breast phantom with 0–5 randomly placed ellipsoidal tumours,
meshes it using NIRFASTer-FF, assigns distinct optical properties,
places a rigid probe (1 source + 3 detectors) and raster scans it
across the phantom, running FD simulations at each position.

Outputs:
- mesh visualisation showing tumours + first probe
- FD scan plots of amplitude & phase vs scan position
"""

import numpy as np
import nirfasterff as ff
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# STEP 1 - Create breast volume with random ellipsoidal tumours
# --------------------------------------------------------------
def build_breast_volume_with_random_tumours(shape=(60, 60, 50),
                                            max_tumours=5,
                                            tumour_radius_range=(4, 10),
                                            background_label=1,
                                            tumour_start_label=2,
                                            rng_seed=None):
    """
    Creates a 3D phantom volume:
    - all voxels start as healthy tissue (label 1)
    - inserts 0–5 random ellipsoidal tumours with labels 2,3,4,...

    Returns
    -------
    vol : np.ndarray
        3D array (Nx x Ny x Nz) where values are region labels.
    """
    Nx, Ny, Nz = shape
    vol = np.full(shape, background_label, dtype=np.uint8)

    rng = np.random.default_rng(rng_seed)
    n_tumours = rng.integers(0, max_tumours + 1)  # sample number of tumours (0 to max)

    print(f"Generating phantom with {n_tumours} tumour(s).")

    current_label = tumour_start_label
    # Precompute meshgrid for ellipsoid formula
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')

    for _ in range(n_tumours):
        # Random tumour centre, keep away from edges
        cx = rng.integers(10, Nx - 10)
        cy = rng.integers(10, Ny - 10)
        cz = rng.integers(5, Nz - 5)

        # Random ellipsoid radii
        rx = rng.integers(*tumour_radius_range)
        ry = rng.integers(*tumour_radius_range)
        rz = rng.integers(3, 8)

        # Create ellipsoid mask and update volume labels
        ellipsoid = (((X - cx)/rx)**2 + ((Y - cy)/ry)**2 + ((Z - cz)/rz)**2) <= 1
        vol[ellipsoid] = current_label
        current_label += 1

    return vol

# --------------------------------------------------------------
# STEP 2 - Mesh the volume into tetrahedra
# --------------------------------------------------------------
def mesh_volume(volume):
    """
    Converts the labelled voxel volume into a tetrahedral FEM mesh.

    Uses CGAL mesher via NIRFASTer-FF. Assigns finer mesh cell sizes
    inside tumour regions for higher numerical fidelity.

    Returns
    -------
    ele, nodes : element connectivity + node coordinate arrays
    """
    params = ff.utils.MeshingParams()
    params.general_cell_size = 2.0  # fallback for any unspecified region
    max_label = np.max(volume)
    # Ensure tumour regions have slightly finer mesh
    params.subdomain = np.array([[1, 2.0]] + [[lbl, 1.5] for lbl in range(2, max_label+1)])

    ele, nodes = ff.meshing.RunCGALMeshGenerator(volume, opt=params)
    ff.meshing.CheckMesh3D(ele, nodes)
    return ele, nodes

def create_stndmesh(ele, nodes):
    """
    Wraps nodes & elements into a NIRFASTer standard mesh object.
    """
    mesh = ff.base.stndmesh()
    mesh.from_solid(ele, nodes)
    return mesh

# --------------------------------------------------------------
# STEP 3 - Set optical properties for each region
# --------------------------------------------------------------
def assign_optical_properties(mesh, rng_seed=None):
    """
    Assigns μₐ and μ′s to each region in the mesh. Background has
    lower absorption/scattering, while tumours are more absorbing & scattering.

    Uses stochastic sampling to vary optical parameters.
    """
    rng = np.random.default_rng(rng_seed)
    regions = np.unique(mesh.region)
    prop = []

    for region in regions:
        if region == 1:
            # Healthy tissue
            mua = rng.uniform(0.01, 0.02)
            musp = rng.uniform(0.9, 1.2)
        else:
            # Tumour: more absorbing/scattering
            mua = rng.uniform(0.025, 0.045)
            musp = rng.uniform(1.0, 1.6)
        prop.append([region, mua, musp, 1.33])

    prop = np.array(prop)
    mesh.set_prop(prop)
    print("Optical properties set:\n", prop)
    return mesh

# --------------------------------------------------------------
# STEP 4 - Build rigid probe geometry
# --------------------------------------------------------------
def make_probe_line(source_xy, offsets=(20, 30, 40)):
    """
    Returns coordinates for a probe:
    - 1 source at source_xy
    - 3 detectors offset along +x by 20, 30, 40 mm
    """
    sx, sy = source_xy
    src = np.array([[sx, sy, 0]])
    det = np.array([[sx + d, sy, 0] for d in offsets])
    return src, det

# --------------------------------------------------------------
# STEP 5 - Visualisation: mesh + one probe
# --------------------------------------------------------------
def visualize_mesh_with_probe(mesh, src, det, save_as="mesh_with_probe.png"):
    """
    Uses matplotlib 3D scatter to show:
    - sparse green dots for healthy tissue nodes
    - dense red dots for tumour nodes
    - optode positions overlaid
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    regions = np.unique(mesh.region)

    for reg in regions:
        nodes = mesh.nodes[mesh.region == reg]
        if reg == 1:
            ax.scatter(nodes[::30,0], nodes[::30,1], nodes[::30,2],
                       color='green', s=1, alpha=0.3)
        else:
            ax.scatter(nodes[::5,0], nodes[::5,1], nodes[::5,2],
                       color='red', s=6, alpha=0.8, label=f'Tumour {reg}')

    ax.scatter(src[:,0], src[:,1], src[:,2], c='yellow', s=80, edgecolor='black', label='Source')
    ax.scatter(det[:,0], det[:,1], det[:,2], c='blue', s=80, edgecolor='black', label='Detectors')
    ax.set_title("Mesh with example probe position")
    ax.legend()
    ax.set_box_aspect([np.ptp(mesh.nodes[:,0]),
                       np.ptp(mesh.nodes[:,1]),
                       np.ptp(mesh.nodes[:,2])])
    plt.savefig(save_as, dpi=200)
    plt.show()
    print(f"Saved mesh figure to {save_as}")

# --------------------------------------------------------------
# STEP 6 - Raster scan probe across x, solving FD each time
# --------------------------------------------------------------
def raster_scan_fd(mesh, scan_x_range, scan_y_fixed=30, fd_freq_hz=140e6):
    """
    Moves the rigid probe in x across the surface. At each position:
    - places source + detectors
    - sets link matrix
    - runs frequency-domain FEM solve

    Returns
    -------
    arrays of amplitude & phase across all scan positions
    """
    all_amplitudes = []
    all_phases = []

    for x in scan_x_range:
        src, det = make_probe_line((x, scan_y_fixed))
        mesh.source = ff.base.optode(src)
        mesh.meas   = ff.base.optode(det)
        mesh.link = np.array([[1,1,1], [1,2,1], [1,3,1]])  # rigid probe
        mesh.touch_optodes()

        # Solve forward problem at this position
        data, info = mesh.femdata(fd_freq_hz)
        amplitude = data.amplitude
        phase = np.degrees(data.phase)  # convert rad → deg

        all_amplitudes.append(amplitude)
        all_phases.append(phase)

    return np.array(all_amplitudes), np.array(all_phases)

# --------------------------------------------------------------
# STEP 7 - Plot amplitude & phase across scan positions
# --------------------------------------------------------------
def plot_fd_scan_results(amplitudes, phases, scan_x_range, save_as="fd_scan.png"):
    """
    Plots amplitude & phase curves for each detector across raster positions.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14,5))
    for ch in range(amplitudes.shape[1]):
        axs[0].plot(scan_x_range, amplitudes[:,ch], 'o-', label=f"Det {ch+1}")
        axs[1].plot(scan_x_range, phases[:,ch], 'o-', label=f"Det {ch+1}")

    axs[0].set_title("Amplitude vs scan position")
    axs[0].set_xlabel("Scan x-position (mm)")
    axs[0].set_ylabel("Amplitude (a.u.)")
    axs[0].grid(True)

    axs[1].set_title("Phase vs scan position")
    axs[1].set_xlabel("Scan x-position (mm)")
    axs[1].set_ylabel("Phase (degrees)")
    axs[1].grid(True)

    plt.suptitle("Frequency-domain scan across breast phantom")
    axs[0].legend()
    axs[1].legend()
    plt.savefig(save_as, dpi=200)
    plt.show()
    print(f"Saved scan results plot to {save_as}")

# --------------------------------------------------------------
# Main program orchestrating the entire pipeline
# --------------------------------------------------------------
def main():
    # Create phantom volume
    vol = build_breast_volume_with_random_tumours(rng_seed=44)
    ele, nodes = mesh_volume(vol)
    mesh = create_stndmesh(ele, nodes)
    mesh = assign_optical_properties(mesh, rng_seed=42)

    # Sanity check: show one probe
    src, det = make_probe_line((15,30))
    visualize_mesh_with_probe(mesh, src, det)

    # Raster scan along x from 10 to 30 in 2mm steps
    scan_x_range = np.arange(10, 31, 2)
    amplitudes, phases = raster_scan_fd(mesh, scan_x_range, scan_y_fixed=30)

    # Plot results
    plot_fd_scan_results(amplitudes, phases, scan_x_range)
    print("All done.")

if __name__ == "__main__":
    main()
