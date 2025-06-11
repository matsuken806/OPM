import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Assumed external ports of these Julia functions:
#   DCore_PointPath, MakeEllip, BiotSav, mutual_l_two_loops
from ToroidOptimizer import dcore_point_path       # Port of DCore_PointPath(r1, r2; dr)
from FieldCalc import make_ellip        # Port of MakeEllip(r1, r2; Center, NPts)
from BiotSav import biot_sav            # Port of BiotSav(PointPath, r; Current, MinThreshold)
from MutualindductanceLoops import mutual_l_two_loops  # Port of Mutual_L_TwoLoops(...) :contentReference[oaicite:0]{index=0}


def eval_induct_d_toroid(IRad, ORad, N,
                         down_sample_fac=1,
                         plot_on=False,
                         n_pts_path=100,
                         n_layers=20,
                         wire_radius=0.001):
    """
    Port of eval_Induct_DToroid(IRad, ORad, N; ...)
    Returns (L_mat, total_L)
    """
    theta_per_turn = 360.0 / N
    # single D‐core loop point path
    pp0 = DCore_PointPath(IRad, ORad, dr=(ORad-IRad)/100, n_pts=n_pts_path)
    # slice area for weighting (unused here but computed in Julia)
    slice_area = symmetric_pp_to_slice_area(pp0)

    # initialize inductance matrix
    L_mat = np.zeros((N, N), dtype=float)

    # first neighbor turn
    pp1 = rot_point_path_y(pp0, theta_per_turn)
    if plot_on:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(pp0[:,0], pp0[:,1], pp0[:,2], s=2)
        ax.scatter(pp1[:,0], pp1[:,1], pp1[:,2], s=2)

    mut, self_L, b_self = mutual_l_two_loops(
        pp0, pp1,
        down_sample_fac=down_sample_fac,
        meas_layers=n_layers,
        min_threshold=1e-10,
        wire_radius=wire_radius,
        include_wire_induct=True,
        save_phi1=True
    )
    L_mat += self_L * off_diag_ones(N, 1)
    L_mat += mut    * off_diag_ones(N, 2)
    L_mat += mut    * off_diag_ones(N, N)

    # remaining turns up to half‐turn symmetry
    for i in range(2, int(np.ceil((N-1)/2)) + 1):
        new_pp = rot_point_path_y(pp0, i * theta_per_turn)
        if plot_on:
            ax.scatter(new_pp[:,0], new_pp[:,1], new_pp[:,2], s=2)
        mut_i, _, _ = mutual_l_two_loops(
            pp0, new_pp, b_self,
            down_sample_fac=down_sample_fac,
            meas_layers=n_layers,
            min_threshold=1e-10,
            wire_radius=wire_radius
        )
        L_mat += mut_i * off_diag_ones(N, i+1)
        L_mat += mut_i * off_diag_ones(N, N-(i-1))

    L_mat[-1, -1] = L_mat[0, 0]
    total_L = L_mat.sum()
    return L_mat, total_L


def eval_induct_circ_toroid(IRad, ORad, N,
                            down_sample_fac=1,
                            plot_on=False,
                            n_pts_path=100,
                            n_layers=20,
                            wire_radius=None):
    """
    Port of eval_Induct_CircToroid(IRad, ORad, N; ...)
    Returns (L_mat, total_L)
    """
    theta_per_turn = 360.0 / N
    core_rad    = (ORad - IRad) / 2.0
    core_off    = (ORad + IRad) / 2.0
    # circular‐core loop
    pp0 = make_ellip(core_rad, core_rad, center=[core_off, 0, 0], n_pts=n_pts_path)
    L_mat = np.zeros((N, N), dtype=float)
    if plot_on:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(pp0[:,0], pp0[:,1], pp0[:,2], s=2)

    # build off‐diagonals
    for i in range(1, N):
        new_pp = rot_point_path_y(pp0, i * theta_per_turn)
        if plot_on:
            ax.scatter(new_pp[:,0], new_pp[:,1], new_pp[:,2], s=2)
        mut, self_L, _ = mutual_l_two_loops(
            pp0, new_pp,
            down_sample_fac=down_sample_fac,
            meas_layers=n_layers,
            min_threshold=1e-10,
            wire_radius=wire_radius,
            include_wire_induct=True
        )
        L_mat[i-1, i-1] = self_L
        L_mat += mut * off_diag_ones(N, i+1)

    L_mat[-1, -1] = L_mat[0, 0]
    return L_mat, L_mat.sum()


def eval_induct_d_toroid_transformer(IRad, ORad, N1, N2,
                                     down_sample_fac=1,
                                     plot_on=False,
                                     n_pts_path=100,
                                     n_layers=20,
                                     self_induct_scale=1.0):
    """
    Port of eval_Induct_DToroidTransformer(IRad, ORad, N₁, N₂; ...)
    Returns (L_mat, idx_array, L11, L22, Lm, Lmu, Lleak1, Lleak2, k)
    """
    N = N1 + N2
    theta_per_turn = 360.0 / N

    # build indexing array for primary/secondary
    idx_primary = np.concatenate([
        np.ones(N1, dtype=int),
        np.zeros(N2, dtype=int)
    ])
    # interleave and sort by turn order
    order = np.argsort(idx_primary)  # mimic Julia's sortslices
    idx_array = np.vstack((idx_primary, order)).T

    pp0 = DCore_PointPath(IRad, ORad, dr=(ORad-IRad)/100, n_pts=n_pts_path)
    L_mat = np.zeros((N, N), dtype=float)
    if plot_on:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(pp0[:,0], pp0[:,1], pp0[:,2], c='b', s=3)

    # first neighbor
    pp1 = rot_point_path_y(pp0, theta_per_turn)
    mut, self_L, b_self = mutual_l_two_loops(
        pp0, pp1,
        down_sample_fac=down_sample_fac,
        meas_layers=n_layers,
        min_threshold=1e-10,
        include_wire_induct=True,
        save_phi1=True
    )
    L_mat += self_L * off_diag_ones(N, 1)
    L_mat += mut    * off_diag_ones(N, 2)
    L_mat += mut    * off_diag_ones(N, N)

    # remaining symmetry pairs
    for i in range(2, int(np.ceil((N-1)/2)) + 1):
        is_primary = bool(idx_array[i,0])
        new_pp = rot_point_path_y(pp0, i*theta_per_turn)
        if plot_on:
            color = 'b' if is_primary else 'r'
            ax.scatter(new_pp[:,0], new_pp[:,1], new_pp[:,2], c=color, s=2)
        mut_i, _, _ = mutual_l_two_loops(
            pp0, new_pp, b_self,
            down_sample_fac=down_sample_fac,
            meas_layers=n_layers,
            min_threshold=1e-10
        )
        L_mat += mut_i * off_diag_ones(N, i+1)
        L_mat += mut_i * off_diag_ones(N, N-(i-1))

    L_mat[-1, -1] = L_mat[0, 0]

    # extract sub‐matrices by indexing
    prim_mask = idx_array[:,0] == 1
    sec_mask  = ~prim_mask
    L11 = L_mat[np.ix_(prim_mask, prim_mask)] * self_induct_scale
    L22 = L_mat[np.ix_(sec_mask , sec_mask )] * self_induct_scale
    M   = L_mat[np.ix_(prim_mask, sec_mask)]
    Lmu = M.sum() * N1/N2 / 2.0
    lleak1 = L11.sum() - Lmu
    lleak2 = L22.sum() - Lmu*(N2/N1)**2
    k = (M.sum()/2.0) / np.sqrt(L11.sum()*L22.sum())
    return L_mat, idx_array, L11, L22, M, Lmu, lleak1, lleak2, k


def eval_induct_d_toroid_split_windings(IRad, ORad, N1_half):
    """
    Port of eval_Induct_DToroid_SplitWindings(IRad, ORad, N₁Half)
    """
    return eval_induct_d_toroid_transformer(
        IRad, ORad, N1_half, N1_half,
        down_sample_fac=10,
        plot_on=True,
        n_layers=45
    )


def build_rot_matrix_deg_z(theta):
    θ = np.deg2rad(theta)
    return np.array([
        [ np.cos(θ), -np.sin(θ), 0],
        [ np.sin(θ),  np.cos(θ), 0],
        [         0,          0, 1]
    ])


def build_rot_matrix_deg_y(theta):
    θ = np.deg2rad(theta)
    return np.array([
        [ np.cos(θ), 0, -np.sin(θ)],
        [         0, 1,           0],
        [ np.sin(θ), 0,  np.cos(θ)]
    ])


def rot_point_path_y(point_path, theta):
    """
    Rotate point_path (M×3) around Y axis by theta degrees.
    """
    M = build_rot_matrix_deg_y(theta)
    return (M @ point_path.T).T


def rot_point_path_z(point_path, theta):
    """
    Rotate point_path (M×3) around Z axis by theta degrees.
    """
    M = build_rot_matrix_deg_z(theta)
    return (M @ point_path.T).T


def symmetric_pp_to_slice_area(pp):
    """
    Port of SymmetricPP2SliceArea(PP)
    """
    total = 0.0
    half = pp.shape[0]//2
    for i in range(half):
        dx = pp[i+1,0] - pp[i,0]
        mean_y = (pp[i+1,1] + pp[i,1]) / 2.0
        total += dx * mean_y
    return total * 2.0


def off_diag_ones(N, dist):
    """
    Port of OffDiagOnes(FullSize, Dist)
    """
    M = np.zeros((N, N), dtype=float)
    offset = dist - 1
    for i in range(N-offset):
        M[i, i+offset] = 1.0
        M[i+offset, i] = 1.0
    return M


def lmat_mutual_indexing(N_total, idx_vec):
    """
    Port of LMatMutualIndexing(NTotal, IndexingVector_Primary)
    Returns boolean masks (primary_self, secondary_self, mutual)
    """
    prim = np.equal.outer(idx_vec, 1)
    sec  = np.equal.outer(idx_vec, 0)
    mutual = ~(prim | sec)
    return prim, sec, mutual


def mutual_l_circ_two_loops_separation(D, separation_vec,
                                       down_sample_fac=1,
                                       meas_layers=55,
                                       min_threshold=1e-15,
                                       n_pts=100,
                                       plot_on=False):
    """
    Port of Mutual_L_CircTwoLoops_Separation(D, SeparationVec; ...)
    Returns array of mutual inductances for each separation.
    """
    pp1 = make_ellip(D/2, D/2, n_pts=n_pts)
    slice_area = np.pi/4 * D**2
    results = []
    for sep in separation_vec:
        pp2 = make_ellip(D/2, D/2, center=[0,0,sep], n_pts=n_pts)
        if plot_on:
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(pp1[:,0], pp1[:,1], pp1[:,2], c='r')
            ax.scatter(pp2[:,0], pp2[:,1], pp2[:,2], c='b')
        tp, weights, dr, dθ, r = pp_2_testpoints_drd_theta(pp1, meas_layers, wire_radius=min_threshold, down_sample_fac=down_sample_fac)
        np_pts_slice = meas_layers * pp1.shape[0]
        weights = weights / weights.sum() * np_pts_slice
        phi21 = 0.0
        for i, pt in enumerate(tp):
            b_self = biot_sav(pp1, pt, min_threshold=min_threshold) * weights[i] * slice_area / np_pts_slice
            b2     = biot_sav(pp2, pt, min_threshold=min_threshold) * weights[i] * slice_area / np_pts_slice
            phi21 += np.dot(b_self, b2) / np.linalg.norm(b_self)
        results.append(phi21)
    return np.array(results)


def self_l_circ_loop(D,
                     down_sample_fac=1,
                     meas_layers=55,
                     min_threshold=1e-15,
                     n_pts=100,
                     plot_on=False,
                     wire_radius=0.0005,
                     include_wire_induct=True):
    """
    Port of Self_L_CircLoop(D; ...)
    Returns self‐inductance Φ₁₁
    """
    pp1 = make_ellip(D/2, D/2, n_pts=n_pts)
    slice_area = np.pi/4 * D**2
    tp, weights = pp_2_testpoints(pp1, meas_layers, down_sample_fac=down_sample_fac, wire_radius=wire_radius)
    np_pts_slice = meas_layers * pp1.shape[0]
    weights = weights / sum(weights) * np_pts_slice
    phi11 = 0.0
    for i, pt in enumerate(tp):
        b_self = biot_sav(pp1, pt, min_threshold=min_threshold) * weights[i] * slice_area / np_pts_slice
        phi11 += np.linalg.norm(b_self)
    if include_wire_induct:
        dist_between = D/2 / meas_layers
        if dist_between < wire_radius:
            w_ind = 0.0
        else:
            w_ind = internal_inductance(np.pi * D)
        phi11 += w_ind
    return phi11


def internal_inductance(L):
    """
    Port of internalInductance(L)
    """
    mu0 = 4 * np.pi * 1e-7
    return mu0 * L / (8 * np.pi)


def make_index_vec(basis_vec, scale):
    """
    Port of makeIndexinVec(BasisVec, Scale)
    """
    new_vec = []
    for b in basis_vec:
        new_vec.extend([b]*scale)
    return np.array(new_vec[:len(basis_vec)], dtype=basis_vec.dtype)


def scatter_plot_toroid(IRad, ORad, N1, N2, bundle_scale,
                        n_pts_path=100,
                        n_layers=20):
    """
    Port of ScatterPlotToroid(...)
    """
    pp0 = DCore_PointPath(IRad, ORad, dr=(ORad-IRad)/100, n_pts=n_pts_path)
    prim_idx = make_index_vec(
        np.concatenate([np.ones(N1), np.zeros(N2)]),
        bundle_scale
    )
    theta_per_turn = 360.0 / (N1+N2)
    ax = plt.figure().add_subplot(projection='3d')
    for i, flag in enumerate(prim_idx, start=1):
        new_pp = rot_point_path_y(pp0, i*theta_per_turn)
        color = 'b' if flag else 'r'
        ax.scatter(new_pp[:,0], new_pp[:,1], new_pp[:,2], c=color, s=3)
    plt.show()


def pp_2_testpoints_drd_theta(PP, layers, wire_radius=0.001, down_sample_fac=1):
    """
    Port of PP_2_TestPoints_drdΘ(PP, Layers; ...)
    Returns (TestPoints, Weights, dr, dθ, r)
    """
    PP_ds = PP[::down_sample_fac]
    mean_pt = PP_ds.mean(axis=0)
    PP_cent = PP_ds - mean_pt
    # build concentric layers
    layers_pts = [PP_cent * np.sqrt(i/(layers+1)) for i in range(1, layers+1)]
    pts_cat = np.vstack(layers_pts[::-1] + [PP_cent * ((np.linalg.norm(PP_cent, axis=1).max()-wire_radius)/np.linalg.norm(PP_cent, axis=1).max())])
    polar = cart2polar(pts_cat)
    L = PP_cent.shape[0]
    dr = np.zeros(len(polar))
    dθ = np.zeros(len(polar))
    # compute dθ and dr per layer
    idx = 0
    for j in range(layers+1):
        for k in range(L):
            if k==0:
                dθ[idx] = abs_ang_between(polar[idx,1], polar[idx+1,1])
            elif k==L-1:
                dθ[idx] = abs_ang_between(polar[idx,1], polar[idx-1,1])
            else:
                dθ[idx] = abs_ang_between(polar[idx-1,1], polar[idx+1,1]) / 2.0
            idx+=1
        # dr for all but last layer
        start = j*L
        stop  = start+L
        if j<layers:
            dr[start:stop] = polar[start:stop,0] - polar[stop:stop+L,0]
        else:
            dr[start:stop] = polar[start:stop,0]
    # re-center
    tp = pts_cat + mean_pt
    weights = np.abs(polar[:,0] * dr * dθ)
    r = polar[:,0]
    return tp, weights, dr, dθ, r


def abs_ang_between(t1, t2):
    """
    Port of AbsAngBetw(θ₁, θ₂)
    """
    diff = abs(t1 - t2)
    if diff == 0:
        return 0.0
    if diff > np.pi:
        return 2*np.pi - (diff % (2*np.pi))
    return diff


def pp_2_testpoints(PP, layers, down_sample_fac=1, wire_radius=None):
    """
    Port of PP_2_TestPoints(PP, Layers; ...)
    Returns (TestPoints, Weights)
    """
    PP_ds = PP[::down_sample_fac]
    mean_pt = PP_ds.mean(axis=0)
    PP_cent = PP_ds - mean_pt
    layers_pts = [PP_cent * (i/(layers+1)) for i in range(1, layers+1)]
    if wire_radius is not None:
        maxd = np.linalg.norm(PP_cent, axis=1).max()
        layers_pts.append(PP_cent * ((maxd-wire_radius)/maxd))
    tp = np.vstack(layers_pts)
    tp += mean_pt
    weights = np.linalg.norm(tp-mean_pt, axis=1)
    return tp, weights


def cart2polar(arr):
    """
    Port of Cart2Polar(Arr)
    """
    r = np.linalg.norm(arr[:,:2], axis=1)
    theta = np.arctan2(arr[:,1], arr[:,0])
    z = arr[:,2]
    return np.column_stack((r, theta, z))


def pp_2_testpoints_arb_plane(PP, layers, wire_radius=0.001, down_sample_fac=1):
    """
    Port of PP_2_TestPoints_arbPlane(PP, Layers; ...)
    Returns (TestPoints, Weights)
    """
    PP_ds = PP[::down_sample_fac]
    mean_pt = PP_ds.mean(axis=0)
    PP_cent = PP_ds - mean_pt
    layers_pts = [PP_cent * np.sqrt(i/(layers+1)) for i in range(1, layers+1)]
    pts_cat = np.vstack(layers_pts[::-1] + [PP_cent * ((np.linalg.norm(PP_cent, axis=1).max()-wire_radius)/np.linalg.norm(PP_cent, axis=1).max())])
    tp = pts_cat + mean_pt
    total_area, weights = effective_area(tp, np.arange(PP.shape[0]))
    return tp, weights


def effective_area(points, boundary_indices):
    """
    Port of effective_area(points, boundary_indices)
    Returns (total_area, effective_areas)
    """
    boundary_pts = points[boundary_indices]
    # compute normal
    v1 = boundary_pts[1] - boundary_pts[0]
    # find non-collinear point
    for j in range(2, len(boundary_pts)):
        v2 = boundary_pts[j] - boundary_pts[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 0:
            break
    normal /= np.linalg.norm(normal)
    # orthonormal basis u, v
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    # project boundary to 2D
    proj = np.array([[np.dot(p-boundary_pts[0], u), np.dot(p-boundary_pts[0], v)]
                     for p in boundary_pts])
    x, y = proj[:,0], proj[:,1]
    n = len(x)
    area = 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # inverse distance weights
    dists = np.array([np.sum(1.0/np.linalg.norm(points[i]-points[np.arange(len(points))!=i], axis=1))
                      for i in range(len(points))])
    weights = dists / dists.sum()
    eff_areas = area * weights
    return area, eff_areas
