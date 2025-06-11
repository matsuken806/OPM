import numpy as np
from Toroid_Inductance import pp_2_testpoints_arb_plane, internal_inductance
from BiotSav import vec_dist, biot_sav

def mutual_l_two_loops(
    pp1: np.ndarray,
    pp2: np.ndarray,
    *,
    down_sample_fac: int = 1,
    meas_layers: int = 55,
    min_threshold: float = 1e-6,
    wire_radius: float = 0.001,
    include_wire_induct: bool = False,
    save_phi1: bool = False
) -> tuple:
    """
    Port of Mutual_L_TwoLoops(PP₁, PP₂; DownSampleFac, MeasLayers, MinThreshold,
                              WireRadius, IncludeWireInduct, SaveΦ₁) :contentReference[oaicite:0]{index=0}

    Returns
    -------
    phi21 : float
        Mutual inductance term Φ₂₁
    phi11 : float
        Self-inductance term Φ₁₁ (including internal inductance if requested)
    b_self : (M×3) ndarray
        Weighted Biot-Savart field of PP₁ at each test point (only valid if save_phi1)
    test_points : (M×3) ndarray
        The interior test points generated around PP₁
    weights : (M,) ndarray
        The raw weights before normalization
    """
    # generate test points & weights
    test_points, weights = pp_2_testpoints_arbplane(
        pp1, meas_layers, wire_radius, down_sample_fac=down_sample_fac
    )
    # avoid zero-weight issues
    weights = weights + np.finfo(weights.dtype).eps
    pts_per_layer = pp1.shape[0]
    n_pts_slice = meas_layers * pts_per_layer
    slice_area = weights.sum()
    # normalize
    weights_norm = weights / weights.sum() * n_pts_slice

    if save_phi1:
        b_self = np.zeros((test_points.shape[0], 3), dtype=float)
        phi11 = 0.0
    else:
        b_self = None
        phi11 = 0.0

    phi21 = 0.0

    for i, tp in enumerate(test_points):
        # field from loop 1
        b1 = biot_sav(pp1, tp, min_threshold=min_threshold)
        w = weights_norm[i] * slice_area / n_pts_slice
        b1_weighted = b1 * w

        if save_phi1:
            b_self[i] = b1_weighted
        phi11 += np.sqrt((b1_weighted**2).sum())

        # field from loop 2
        b2 = biot_sav(pp2, tp, min_threshold=min_threshold) * w
        phi21 += np.dot(b1_weighted, b2) / np.sqrt((b1_weighted**2).sum())

    if include_wire_induct:
        # add internal inductance of PP₁
        cum_dist = sum(
            np.linalg.norm(pp1[i] - pp1[i+1])
            for i in range(pp1.shape[0] - 1)
        )
        phi11 += internal_inductance(cum_dist)

    return phi21, phi11, b_self, test_points, weights


def mutual_l_two_loops_prior(
    pp1: np.ndarray,
    pp2: np.ndarray,
    prior_b_self: np.ndarray,
    *,
    down_sample_fac: int = 1,
    meas_layers: int = 55,
    min_threshold: float = 1e-6,
    wire_radius: float = 0.001
) -> float:
    """
    Port of Mutual_L_TwoLoops(PP₁, PP₂, PriorBSelf; …) :contentReference[oaicite:1]{index=1}

    Returns
    -------
    phi21 : float
        Mutual inductance term Φ₂₁ using the provided prior B_self array.
    """
    # regenerate test points & weights
    test_points, weights = pp_2_testpoints_arbplane(
        pp1, meas_layers, wire_radius, down_sample_fac=down_sample_fac
    )
    pts_per_layer = pp1.shape[0]
    n_pts_slice = meas_layers * pts_per_layer
    slice_area = weights.sum()
    weights_norm = weights / weights.sum() * n_pts_slice

    # prepare pp2 as closed loop
    pp2_closed = np.vstack([pp2, pp2[0]])
    dL = [vec_dist(pp2_closed[i-1:i+1]) for i in range(1, pp2_closed.shape[0])]
    L = len(dL) + 1

    phi21 = 0.0
    for i, tp in enumerate(test_points):
        b2 = biot_sav(pp2_closed, dL, tp, L, min_threshold=min_threshold)
        w = weights_norm[i] * slice_area / n_pts_slice
        phi21 += np.dot(prior_b_self[i], b2 * w) / np.sqrt((prior_b_self[i]**2).sum())

    return phi21


def mutual_l_two_loops_with_points(
    pp1: np.ndarray,
    pp2: np.ndarray,
    prior_b_self: np.ndarray,
    test_points: np.ndarray,
    weights: np.ndarray,
    *,
    down_sample_fac: int = 1,
    meas_layers: int = 55,
    min_threshold: float = 1e-6,
    wire_radius: float = 0.001
) -> float:
    """
    Port of Mutual_L_TwoLoops(PP₁, PP₂, PriorBSelf, TestPoints, Weights; …) :contentReference[oaicite:2]{index=2}

    Returns
    -------
    phi21 : float
        Mutual inductance term Φ₂₁ using provided test points & weights.
    """
    pts_per_layer = pp1.shape[0]
    n_pts_slice = meas_layers * pts_per_layer
    slice_area = weights.sum()
    weights_norm = weights / weights.sum() * n_pts_slice

    # prepare pp2 as closed loop
    pp2_closed = np.vstack([pp2, pp2[0]])
    dL = [vec_dist(pp2_closed[i-1:i+1]) for i in range(1, pp2_closed.shape[0])]
    L = len(dL) + 1

    phi21 = 0.0
    for i, tp in enumerate(test_points):
        b2 = biot_sav(pp2_closed, dL, tp, L, min_threshold=min_threshold)
        w = weights_norm[i] * slice_area / n_pts_slice
        phi21 += np.dot(prior_b_self[i], b2 * w) / np.sqrt((prior_b_self[i]**2).sum())

    return phi21
