import numpy as np
import time
from numpy.linalg import pinv
from MutualindductanceLoops import mutual_l_two_loops
from BiotSav import biot_sav, vec_dist
# from biot_sav_module import vec_dist, biot_sav, mutual_l_two_loops

def eval_eddy_current_pp_list(
    PP_All,
    current_input_list,
    f,
    test_points=None,
    open_turns=None,
    down_sample_fac=1,
    plot_on=False,
    n_pts_path=100,
    n_layers=20,
    wire_radius=0.001,
    quasistatic=True,
    node_node_cap=1e-15,
):
    """
    Evaluate the equivalent circuit (eddy currents, voltages, etc.) for a system of coupled loops.

    Parameters
    ----------
    PP_All : list of (N_i×3) arrays
        Each element is an array of 3D points defining one closed wire loop.
    current_input_list : array-like, length = len(PP_All)
        Currents injected into each loop (set to zero for loops you wish to solve for).
    f : array-like of shape (N_f,)
        Frequencies at which to evaluate the circuit.
    test_points : (M×3) array, optional
        Points at which to evaluate the magnetic flux (default: [[0,0,0]]).
    open_turns : array-like of ints (0 or 1), length = len(PP_All)
        0 = closed loop, 1 = open (non-conducting) turn (default: all closed).
    down_sample_fac : int, optional
    plot_on : bool, optional
        If True, plot Φ magnitude vs. test point coordinates.
    n_pts_path : int, optional
    n_layers : int, optional
    wire_radius : float, optional
    quasistatic : bool, optional
    node_node_cap : float, optional

    Returns
    -------
    LMat : (N×N) ndarray
        Self‐ and mutual‐inductance matrix.
    GMat_all_freq : (S×S×N_f) ndarray, complex
        Complete circuit admittance matrices at each frequency (S = 2·N + N_in).
    circ_outputs_all_freq : (S×N_f) ndarray, complex
    circ_inputs_all_freq : (S×N_f) ndarray, complex
    circ_output_key : list of str, length = S
    rx_inds : (S,) bool ndarray
    phi : (M×3×N_f) ndarray, complex
        Magnetic flux vectors at each test point and frequency.

    Notes
    -----
    Relies on helper functions ported from Julia:
      - `mutual_l_two_loops()`  — Mutual_L_TwoLoops
      - `vec_dist()`            — VecDist
      - `biot_sav()`            — BiotSav
    :contentReference[oaicite:0]{index=0}
    """
    # --- Defaults ---
    if test_points is None:
        test_points = np.array([[0.0, 0.0, 0.0]])
    else:
        test_points = np.asarray(test_points, dtype=float)

    if open_turns is None:
        open_turns = np.zeros(len(PP_All), dtype=int)
    else:
        open_turns = np.asarray(open_turns, dtype=int)

    rho_cu = 1.72e-8
    wire_area = np.pi * wire_radius**2
    N = len(PP_All)

    # Prepare inductance matrix
    LMat = np.zeros((N, N), dtype=float)
    # Initial mutual/self between last & first
    mut, self_L, saved_b_self_arr = mutual_l_two_loops(
        PP_All[-1], PP_All[0],
        down_sample_fac=down_sample_fac,
        meas_layers=n_layers,
        min_threshold=1e-10,
        wire_radius=wire_radius,
        include_wire_induct=True,
        save_phi1=True
    )
    LMat[-1, -1] = self_L

    # --- Build full LMat ---
    start_time = time.time()
    total_iters = N * (N - 1) / 2
    done = 0
    for j in range(N):
        for k in range(j + 1, N):
            done += 1
            if k == j + 1:
                # Progress estimate
                elapsed = time.time() - start_time
                avg = elapsed / done
                rem = avg * (total_iters - done)
                m, s = int(rem // 60), int(rem % 60)
                print(f"Mutual L between loop {j+1} & {k+1} ({done}/{int(total_iters)}) — est {m}m{s}s")
                mut, self_L, saved_b_self_arr = mutual_l_two_loops(
                    PP_All[j], PP_All[k],
                    down_sample_fac=down_sample_fac,
                    meas_layers=n_layers,
                    min_threshold=1e-10,
                    wire_radius=wire_radius,
                    include_wire_induct=True,
                    save_phi1=True
                )
            else:
                mut = mutual_l_two_loops(
                    PP_All[j], PP_All[k],
                    saved_b_self_arr,
                    down_sample_fac=down_sample_fac,
                    meas_layers=n_layers,
                    min_threshold=1e-10,
                    wire_radius=wire_radius
                )
            LMat[j, j] = self_L
            LMat[j, k] = mut
            LMat[k, j] = mut

    # --- Wire conductance & length ---
    wire_conductance = np.zeros((N, N), dtype=float)
    wire_length = np.zeros(N, dtype=float)
    eye = np.zeros((N, N), dtype=float)
    for jj in range(N):
        PP = np.vstack([PP_All[jj], PP_All[jj][0]])
        dL = [vec_dist(PP[i-1:i+1]) for i in range(1, PP.shape[0])]
        if open_turns[jj] == 0:
            wire_length[jj] = sum(np.linalg.norm(seg) for seg in dL)
            wire_conductance[jj, jj] = 1 / (wire_length[jj] * rho_cu / wire_area)
        else:
            wire_conductance[jj, jj] = 0.0
        eye[jj, jj] = 1.0

    # --- Circuit assembly for each frequency ---
    N_f = len(f)
    currs = np.array(current_input_list, copy=True)
    N_in = int(np.count_nonzero(currs != 0))
    input_inds = np.nonzero(currs != 0)[0]
    S = 2 * N + N_in

    circ_output_key = [""] * S
    rx_inds = np.concatenate([
        np.zeros(N_in, dtype=bool),
        open_turns.astype(bool),
        np.zeros(N, dtype=bool)
    ])
    not_rx_inds = np.concatenate([
        np.zeros(N_in, dtype=bool),
        (~open_turns.astype(bool)),
        np.zeros(N, dtype=bool)
    ])

    circ_output_key[:N_in] = ["Voltage on Current Source"] * N_in
    circ_output_key[N_in:N_in+rx_inds.sum()] = ["Voltage on Rx"] * rx_inds.sum()
    circ_output_key[N_in+rx_inds.sum():N_in+rx_inds.sum()+not_rx_inds.sum()] = \
        ["Voltage on Current-Carrying Wire"] * not_rx_inds.sum()
    circ_output_key[N_in+N:] = ["Current in Wire"] * N

    GMat_all_freq = np.zeros((S, S, N_f), dtype=complex)
    circ_outputs_all_freq = np.zeros((S, N_f), dtype=complex)
    circ_inputs_all_freq  = np.zeros((S, N_f), dtype=complex)
    phi = np.zeros((test_points.shape[0], 3, N_f), dtype=complex)

    for ff, freq in enumerate(f):
        lam = 3e8 / freq
        if not quasistatic:
            for idx in input_inds:
                # phase shift along wire
                currs[idx] *= np.exp(wire_length[idx]/lam * 2*np.pi*1j)

        # Build GMat
        G = np.zeros((S, S), dtype=complex)
        G[N_in:N_in+N, N_in:N_in+N] = wire_conductance
        G[N_in+N:,   N_in+N:]   = -LMat * (2*np.pi*freq*1j)
        G[N_in:N_in+N, N_in+N:]  = -eye
        G[N_in+N:,   N_in:N_in+N] = -eye

        for kk, I in enumerate(input_inds):
            G[kk, kk] = wire_conductance[I, I]
            G[N_in+I, N_in+I] = wire_conductance[I, I]
            G[kk, N_in+I] = -wire_conductance[I, I]
            G[N_in+I, kk] = -wire_conductance[I, I]

        GMat_all_freq[:, :, ff] = G
        circ_in = np.zeros(S, dtype=complex)
        circ_in[input_inds] = currs[input_inds]
        circ_inputs_all_freq[:, ff] = circ_in

        circ_out = pinv(G) @ circ_in
        circ_outputs_all_freq[:, ff] = circ_out

        # Compute flux Φ at each test point
        wires = np.vstack(PP_All)
        for jj in range(N):
            PP = np.vstack([PP_All[jj], PP_All[jj][0]])
            dL = [vec_dist(PP[i-1:i+1]) for i in range(1, PP.shape[0])]
            PP_rs = [PP[i] for i in range(PP.shape[0])]
            for i, pt in enumerate(test_points):
                min_d = np.min(np.linalg.norm(wires - pt, axis=1))
                if min_d >= 0.5 * wire_radius:
                    phi[i, :, ff] += biot_sav(PP_rs, dL, pt, len(PP_rs)) * circ_out[N_in+N+jj]
                else:
                    print("Point too close to a wire")
        # Clip outliers
        mag = np.sqrt((phi[:, :, ff].real**2).sum(axis=1))
        mean_mag = mag.mean()
        mag[mag > 3*mean_mag] = 3*mean_mag

        if plot_on:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(test_points[:, 0], mag)
            plt.plot(test_points[:, 1], mag)
            plt.plot(test_points[:, 2], mag)
            plt.show()

    return (
        LMat,
        GMat_all_freq,
        circ_outputs_all_freq,
        circ_inputs_all_freq,
        circ_output_key,
        rx_inds,
        phi
    )
