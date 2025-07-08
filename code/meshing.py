#!/usr/bin/env python3
"""
Generates a cylinder FEM mesh with an inner tumour cylinder using
NIRFASTer-FF, sets distinct optical properties, places optodes,
and visualises everything, saving to disk.
"""


import numpy as np
import nirfasterff as ff
import matplotlib.pyplot as plt

def build_box_with_cylinder_tumour(shape=(60,60,50),
                                   tumour_radius=8,
                                   tumour_center_offset=(5,5,0)):
    """
    Builds a cuboid volume with:
    - region 1: entire box
    - region 2: a small cylinder (tumour) inside the box
    """
    Nx, Ny, Nz = shape
    cx, cy = Nx // 2, Ny // 2
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    # start with all voxels in region 1
    vol = np.ones(shape, dtype=np.uint8)

    # add small cylinder tumour (region 2)
    tx, ty, tz = cx + tumour_center_offset[0], cy + tumour_center_offset[1], Nz//2
    mask_tumour = (X - tx)**2 + (Y - ty)**2 < tumour_radius**2
    vol[mask_tumour, :] = 2

    return vol


def mesh_volume(volume):
    ele, nodes = ff.meshing.RunCGALMeshGenerator(volume)
    ff.meshing.CheckMesh3D(ele, nodes)
    return ele, nodes


def create_stndmesh(ele, nodes):
    mesh = ff.base.stndmesh()
    mesh.from_solid(ele, nodes)
    return mesh


def set_optical_properties(mesh):
    """
    Sets optical properties so that region 2 (tumour) is more absorbing/scattering.
    """
    prop = np.array([
        [1, 0.01, 1.0, 1.33],  # background cylinder
        [2, 0.03, 1.5, 1.33]   # tumour inclusion
    ])
    mesh.set_prop(prop)
    return mesh


def place_optodes(mesh):
    """
    Places one source and 4 detectors around the cylinder.
    """
    mesh.link = np.atleast_2d([1,1,1,1,1])  # dummy link matrix for 1 src, 4 det

    src = np.array([[20,30,0]])
    det = np.array([
        [40,30,0],
        [35,35,0],
        [35,25,0],
        [45,30,0]
    ])

    mesh.source = ff.base.optode(src)
    mesh.meas   = ff.base.optode(det)
    mesh.touch_optodes()
    return src, det


def visualize_mesh_with_optodes(mesh, src, det, save_as="mesh_with_optodes.png"):
    """
    Visualises mesh + optodes using direct matplotlib overlay
    """

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot tumour nodes by directly filtering mesh.region==2
    tumour_nodes = mesh.nodes[mesh.region==2]
    ax.scatter(tumour_nodes[:,0], tumour_nodes[:,1], tumour_nodes[:,2],
               c='red', s=1, alpha=0.6, label='Tumour region')

    # Also plot a subset of tissue nodes (region==1) for background
    tissue_nodes = mesh.nodes[mesh.region==1]
    ax.scatter(tissue_nodes[::50,0], tissue_nodes[::50,1], tissue_nodes[::50,2],
               c='green', s=1, alpha=0.6, label='Tissue region (sampled)')

    # Plot optodes
    ax.scatter(src[:,0], src[:,1], src[:,2], c='yellow', s=80, label='Source')
    ax.scatter(det[:,0], det[:,1], det[:,2], c='blue', s=80, label='Detectors')

    ax.set_title("Mesh with tumour + optodes")
    ax.legend()
    plt.savefig(save_as)
    plt.show()
    print(f"Saved figure to {save_as}")


def run_tpsf_simulation(mesh, total_time=10e-9, dt=1e-11, beautify=False, save_as="tpsf_plot.png"):
    """
    Runs a time-resolved FEM forward solve and plots the TPSF at the first detector.
    """
    print("Running TPSF simulation...")
    tpsf_result = mesh.femdata_tpsf(total_time, dt, beautify=beautify)[0]
    
    time = tpsf_result.time * 1e9  # convert to ns for plotting
    tpsf_curve = tpsf_result.tpsf[0,:]

    plt.figure(figsize=(8,5))
    plt.plot(time, tpsf_curve, color='purple')
    plt.xlabel("Time (ns)")
    plt.ylabel("TPSF (a.u.)")
    plt.title("Temporal Point Spread Function at detector 1")
    plt.grid(True)
    plt.savefig(save_as)
    plt.show()
    print(f"Saved TPSF plot to {save_as}")



def main():
    vol = build_box_with_cylinder_tumour()
    print(f"Volume shape: {vol.shape}, unique regions: {np.unique(vol)}")

    ele, nodes = mesh_volume(vol)
    print(f"Mesh: {nodes.shape[0]} nodes, {ele.shape[0]} elements")

    mesh = create_stndmesh(ele, nodes)
    mesh = set_optical_properties(mesh)
    src, det = place_optodes(mesh)

    visualize_mesh_with_optodes(mesh, src, det)

    run_tpsf_simulation(mesh)

    print("All done.")


if __name__ == "__main__":
    main()
