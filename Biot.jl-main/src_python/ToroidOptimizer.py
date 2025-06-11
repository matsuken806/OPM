import numpy as np
import matplotlib.pyplot as plt

from FieldCalc import field_map_point_path, make_ellip
from SaddleCoils import make_rect_point_path

def dcore_geom(r1, r2, dr=1e-4, npts=None, plot_on=False, upsample_points=1e4):
    """
    Port of DCoreGeom(r₁, r₂; dr=0.0001, NPts=nothing, PlotOn=false, UpsamplePoints=1e4)
    Returns (r_vals, z_vals) arrays describing the D-core cross‐section.
    """
    # derivative dz/dr
    def dZdr(r):
        return np.log(np.sqrt(r1 * r2) / r) / np.sqrt(np.log(r / r1) * np.log(r2 / r))

    # if npts specified, override dr for uniform upsampling
    if npts is not None:
        dr = (r2 - r1) / upsample_points

    # initialize
    ri = r1 + dr/10.0
    r_list = [ri]
    z_list = [0.0]
    num_iters = int(np.floor((r2 - r1) / dr))
    for _ in range(1, num_iters):
        dz = dZdr(ri) * dr
        ri = ri + dr
        z_next = z_list[-1] + dz
        r_list.append(ri)
        z_list.append(z_next)

    r_arr = np.array(r_list)
    z_arr = np.array(z_list)
    # enforce boundary condition z(end)=0
    z_arr = z_arr - z_arr[-1]
    z_arr[0] = 0.0

    # flatten the top, then mirror to form full profile
    z_flat_height = z_arr[1]
    start_pts = int(np.ceil(z_flat_height / dr))
    for i in range(1, start_pts + 1):
        z_arr = np.insert(z_arr, 1 + i, z_arr[1 + i - 1] * i / start_pts)
        r_arr = np.insert(r_arr, 1 + i, r_arr[0])

    # mirror (excluding first and last duplicated points)
    z_full = np.concatenate([z_arr, -z_arr[1:-1][::-1]])
    r_full = np.concatenate([r_arr,  r_arr[1:-1][::-1]])

    # downsample to npts if requested
    if npts is not None:
        step = max(1, int(round(upsample_points / npts)))
        r_full = r_full[::step]
        z_full = z_full[::step]

    if plot_on:
        plt.plot(r_full, z_full)
        plt.xlabel('r')
        plt.ylabel('z')
        plt.title('D-core Geometry')
        plt.axis('equal')
        plt.show()

    return r_full, z_full


def dcore_point_path(r1, r2, npts=100):
    """
    Port of DCore_PointPath(r₁, r₂; NPts=100)
    Returns an (M×3) ndarray of points along the D-core loop.
    """
    # sample finely, then re‐interpolate to npts
    r_vals, z_vals = dcore_geom(r1, r2, npts=npts*10)
    # total contour length
    seg_lengths = np.sqrt(np.diff(r_vals)**2 + np.diff(z_vals)**2)
    total_len = seg_lengths.sum()
    target = total_len / (npts + 1)

    pts_r = [r_vals[0]]
    pts_z = [z_vals[0]]
    acc = 0.0
    for i in range(len(seg_lengths)):
        acc += seg_lengths[i]
        if acc > target:
            pts_r.append(r_vals[i])
            pts_z.append(z_vals[i])
            acc = 0.0

    pts_r = np.array(pts_r)
    pts_z = np.array(pts_z)
    zeros = np.zeros_like(pts_r)
    return np.column_stack((pts_r, pts_z, zeros))


def field_map_d_toroid(r1, r2, npts=100, test_layers=20):
    """
    Port of FieldMapDToroid(r₁, r₂; NPts=100, TestLayers=20)
    Generates and plots the field‐map of a D-core toroid.
    """
    pp = dcore_point_path(r1, r2, npts=npts)
    return field_map_point_path(pp, test_layers, weight_radius=True, inv_weights=True)


def compare_d_circ_rect_toroid(dr1, dr2, rect_coords, cr1, cr2, circ_center, test_layers=20):
    """
    Port of CompareD_Circ_Rect_Toroid(Dr₁,Dr₂,RectCoords,Cr₁,Cr₂,CircCenter;TestLayers=20)
    Compares field maps of D-toroid, rectangular saddle, and circular coil.
    """
    # D-core toroid
    plt.figure(75)
    field_map_d_toroid(dr1, dr2, npts=100, test_layers=test_layers)

    # rectangular saddle coil
    plt.figure(76)
    rpp = make_rect_point_path(rect_coords, nelement=50, plot_on=False)
    field_map_point_path(rpp, test_layers, weight_radius=True, inv_weights=True)

    # circular coil
    plt.figure(77)
    cpp = make_ellip(cr1, cr2, center=circ_center, n_pts=100)
    cpp = cpp[::-1]
    field_map_point_path(cpp, test_layers, weight_radius=True, inv_weights=True)
