import numpy as np
import time
from MutualindductanceLoops import mutual_l_two_loops, mutual_l_two_loops_prior
from Toroid_Inductance import pp_2_testpoints_arb_plane  # Port of PP_2_TestPoints_arbPlane :contentReference[oaicite:0]{index=0}
from BiotSav import biot_sav

def getPPArea(PP, meas_layers=55, wire_rad=0.0002):
    """
    Port of getPPArea(PP; MeasLayers=55, WireRad=0.2e-3)
    Returns the slice area (sum of weights) for a single PointPath.
    """
    test_points, weights = pp_2_testpoints_arb_plane(PP, layers=meas_layers, wire_radius=wire_rad)
    return weights.sum()

def SortPPByArea(PP_All, meas_layers=55, wire_rad=0.0002):
    """
    Port of SortPPByArea(PP_All; MeasLayers=55, WireRad=0.2e-3)
    Sorts a list of PointPaths by their effective slice area, ascending.
    Returns the sorted list and the original indices.
    """
    areas = np.array([getPPArea(PP, meas_layers=meas_layers, wire_rad=wire_rad) for PP in PP_All])
    new_indexes = np.argsort(areas)
    PP_All_new = [PP_All[i] for i in new_indexes]
    return PP_All_new, new_indexes

def eval_LMat_PP_List(
    PP_All,
    down_sample_fac=1,
    plot_on=False,
    n_pts_path=100,
    n_layers=20,
    wire_radius=0.001
):
    """
    Port of eval_LMat_PP_List(PP_All; DownSampleFac=1, PlotOn=false,
                              NPtsPath=100, NLayers=20, WireRadius=0.001)
    Computes the inductance matrix LMat for a list of magnetically coupled loops.
    Returns:
      LMat: (N×N) ndarray of mutual/self inductances
      new_indexes: ndarray of original indices after sorting by area
    :contentReference[oaicite:1]{index=1}
    """
    ρ_cu = 1.72e-8
    wire_area = np.pi * wire_radius**2
    # Sort loops by area
    PP_sorted, new_indexes = SortPPByArea(PP_All, meas_layers=n_layers, wire_rad=wire_radius)
    N = len(PP_sorted)

    # Initialize
    LMat = np.zeros((N, N), dtype=float)
    saved_b_self = None
    test_points = None
    weights = None

    # First mutual/self between last & first
    mut, self_L, saved_b_self, test_points, weights = mutual_l_two_loops(
        PP_sorted[-1], PP_sorted[0],
        down_sample_fac=down_sample_fac,
        meas_layers=n_layers,
        min_threshold=1e-10,
        wire_radius=wire_radius,
        include_wire_induct=False,
        save_phi1=True
    )
    LMat[-1, -1] = self_L

    # Loop over unique pairs
    start_time = time.time()
    total_iters = N * (N - 1) / 2
    completed = 0

    for j in range(N):
        for k in range(j + 1, N):
            completed += 1
            if k == j + 1:
                # Estimate remaining time
                elapsed = time.time() - start_time
                avg = elapsed / completed
                rem = avg * (total_iters - completed)
                m, s = divmod(int(rem), 60)
                print(f"Estimating mutual L ({completed}/{int(total_iters)}): ~{m}m{s}s remaining")
                mut, self_L, saved_b_self, test_points, weights = mutual_l_two_loops(
                    PP_sorted[j], PP_sorted[k],
                    down_sample_fac=down_sample_fac,
                    meas_layers=n_layers,
                    min_threshold=1e-10,
                    wire_radius=wire_radius,
                    include_wire_induct=False,
                    save_phi1=True
                )
            else:
                # Use prior B_self and points to compute only mutual term
                mut = mutual_l_two_loops_prior(
                    PP_sorted[j], PP_sorted[k],
                    saved_b_self, test_points, weights,
                    down_sample_fac=down_sample_fac,
                    meas_layers=n_layers,
                    min_threshold=1e-10,
                    wire_radius=wire_radius
                )
            # Fill matrix (self_L reused for diagonal)
            LMat[j, j] = self_L
            LMat[j, k] = mut
            LMat[k, j] = mut

    return LMat, new_indexes
