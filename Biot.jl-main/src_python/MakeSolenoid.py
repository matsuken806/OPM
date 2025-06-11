import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

from BiotSav import biot_sav
from MutualindductanceLoops import mutual_l_two_loops, vec_dist
from FieldCalc import make_ellip

def off_diag_ones(N, k):
    """
    OffDiagOnes(N, k): 
    k=1 → identity matrix; 
    k>=2 → ones on ±(k-1) off-diagonals.
    """
    M = np.zeros((N, N), dtype=float)
    offset = k - 1
    for i in range(N - offset):
        M[i, i + offset] = 1.0
        if offset != 0:
            M[i + offset, i] = 1.0
    return M

def eval_solenoid_induct(N, r1, r2, L, *,
                         down_sample_fac=1,
                         plot_on=False,
                         n_pts_path=100,
                         n_layers=20):
    """
    eval_Solenoid_Induct(N, r1, r2, L;
                         DownSampleFac=1, PlotOn=False,
                         NPtsPath=100, NLayers=20)
    → (L_mat, total_L)
    """
    # １ターン目ループ点列
    pp0 = make_ellip(r1, r2, n_pts=n_pts_path)

    # 全体行列の初期化
    L_mat = np.zeros((N, N), dtype=float)

    if plot_on:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pp0[:,0], pp0[:,1], pp0[:,2], s=2)
        ax.set_title("Turn 1")

    # ２ターン目
    z2 = (1 / N) * L
    pp1 = make_ellip(r1, r2, center=[0, 0, z2], n_pts=n_pts_path)
    if plot_on:
        ax.scatter(pp1[:,0], pp1[:,1], pp1[:,2], s=2)

    # Self & first mutual
    mut, self_L, saved_phi1 = mutual_l_two_loops(
        pp0, pp1,
        down_sample_fac=down_sample_fac,
        meas_layers=n_layers,
        min_threshold=1e-10,
        include_wire_induct=False,
        save_phi1=True
    )
    L_mat += self_L * off_diag_ones(N, 1)
    L_mat += mut    * off_diag_ones(N, 2)

    # 残りのターン（3..N）
    for i in range(3, N+1):
        zi = ((i - 1) / N) * L
        ppi = make_ellip(r1, r2, center=[0, 0, zi], n_pts=n_pts_path)
        if plot_on:
            ax.scatter(ppi[:,0], ppi[:,1], ppi[:,2], s=2)

        # saved_phi1 を使った mutual
        mut = mutual_l_two_loops(
            pp0, ppi, saved_phi1,
            down_sample_fac=down_sample_fac,
            meas_layers=n_layers,
            min_threshold=1e-10
        )
        L_mat += mut * off_diag_ones(N, i)

    # 最後の対角要素を最初と揃える
    L_mat[N-1, N-1] = L_mat[0, 0]

    total_L = np.sum(L_mat)
    return L_mat, total_L

def eval_field_centered(N, r1, r2, L, *, n_pts_coil=100):
    """
    EvalField_Centered(N, r1, r2, L; NPts_Coil=100) → sum of B_z at center
    """
    vals = []
    for wi in range(1, N+1):
        zi = (wi - 1) / N * L
        pp = make_ellip(r1, r2, center=[0, 0, zi], n_pts=n_pts_coil)
        B = biot_sav(pp.tolist(), [0, 0, L/2], min_threshold=0.001)
        vals.append(B[2])
    return sum(vals)

def make_ellip_solenoid_point_path(N, r1, r2, full_length, zoffset, *,
                                   n_pts_path=100):
    """
    MakeEllipSolenoidPointPath(N, r1, r2, FullLength, Zoffset;
                               NPtsPath=100) → (N·NPtsPath×3) ndarray
    """
    dL = full_length / (N - 1)
    zs = np.linspace(-full_length/2, full_length/2, N)
    loops = [
        make_ellip(r1, r2, center=[0, 0, z + zoffset], n_pts=n_pts_path)
        for z in zs
    ]
    return np.vstack(loops)

def make_concentric_solenoids(N_vec, r1_vec, r2_vec, L_vec, zoffset_vec, *,
                              down_sample_fac=1,
                              plot_on=False,
                              n_pts_path=50,
                              n_layers=15):
    """
    makeConcentricSolenoids(NVec, r1Vec, r2Vec, LVec, ZOffsetVec;
                            DownSampleFac=1, PlotOn=False,
                            NPtsPath=50, NLayers=15)
    → (L_mat, pp_list)
    """
    N_coils = len(N_vec)
    pp_list = [
        make_ellip_solenoid_point_path(
            N_vec[i],
            r1_vec[i],
            r2_vec[i],
            L_vec[i],
            zoffset_vec[i],
            n_pts_path=n_pts_path
        )
        for i in range(N_coils)
    ]
    L_mat = np.zeros((N_coils, N_coils), dtype=float)

    if plot_on:
        fig = plt.figure()
        for j, pp in enumerate(pp_list, start=1):
            ax = fig.add_subplot(1, N_coils, j, projection='3d')
            ax.plot(pp[:,0], pp[:,1], pp[:,2], lw=1)

    for j in range(N_coils):
        for k in range(N_coils):
            if j == k:
                continue
            mut, self_L, _ = mutual_l_two_loops(
                pp_list[j], pp_list[k],
                down_sample_fac=down_sample_fac,
                meas_layers=n_layers,
                min_threshold=1e-10,
                include_wire_induct=False,
                save_phi1=False
            )
            L_mat[j, j] = self_L
            L_mat[j, k] = mut
            L_mat[k, j] = mut  # 対称

    return L_mat, pp_list
