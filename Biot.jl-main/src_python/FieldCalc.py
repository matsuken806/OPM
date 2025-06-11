import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用
from BiotSav import biot_sav

def make_ellip(r1, r2, center=None, n_pts=100):
    """
    MakeEllip(r₁, r₂; Center=[0,0,0], NPts=100) の移植版
    楕円状の閉ループ上の点列 (N×3 ndarray) を返す
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    X = r1 * np.cos(theta) + center[0]
    Y = r2 * np.sin(theta) + center[1]
    Z = np.full(n_pts, center[2])
    return np.vstack([X, Y, Z]).T


def field_on_axis_circ(R, z, I=1.0):
    """
    FieldOnAxis_Circ(R, z; I=1) の移植版
    円形コイルの軸上磁場 Bz を返す
    """
    mu0 = 4 * np.pi * 1e-7
    return mu0 * 2*np.pi * R**2 * I / (4*np.pi * (z**2 + R**2)**1.5)


def make_ellip_test_points(r1, r2, center=None, n_pts=6, layers=3):
    """
    MakeEllipTestPoints(r₁, r₂; Center=[0,0,0], NPts=6, Layers=3) の移植版
    複数レイヤーの楕円状テスト点列を X, Y, Z の３つの 1D 配列で返す
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=float)

    # レイヤー 1
    X = np.array([r1*1/(layers+1)*np.cos(i/n_pts*2*np.pi)+center[0] for i in range(n_pts)])
    Y = np.array([r2*1/(layers+1)*np.sin(i/n_pts*2*np.pi)+center[1] for i in range(n_pts)])
    Z = np.full(n_pts, center[2])

    # レイヤー 2..layers
    for ll in range(2, layers+1):
        x1 = np.array([r1*ll/(layers+1)*np.cos(i/n_pts*2*np.pi)+center[0] for i in range(n_pts)])
        y1 = np.array([r2*ll/(layers+1)*np.sin(i/n_pts*2*np.pi)+center[1] for i in range(n_pts)])
        z1 = np.full(n_pts, center[2])
        X = np.concatenate([X, x1])
        Y = np.concatenate([Y, y1])
        Z = np.concatenate([Z, z1])

    return X, Y, Z


def field_map_point_path(point_path, num_layers,
                         weight_radius=False, inv_weights=False,
                         plot_axes=None):
    """
    FieldMapPointPath(PointPath, NumLayers;
                      WeightRadius=false, InvWeights=false, PlotAxes=nothing) の移植版

    ・PointPath: (N×3) ndarray
    ・NumLayers: 内部テスト点レイヤ数
    ・weight_radius / inv_weights: 重み付けオプション
    ・plot_axes: matplotlib の Axes3D オブジェクトを渡せばそこにプロット
    """
    pts = np.asarray(point_path, dtype=float)
    x = pts[:,0]; y = pts[:,1]
    xTP = x.copy(); yTP = y.copy()

    # 内部レイヤー点を追加
    xmid, ymid = (x.min()+x.max())/2, (y.min()+y.max())/2
    for i in range(1, num_layers-1):
        factor = i/num_layers
        xTP = np.concatenate([xTP, (x - xmid)*factor + xmid])
        yTP = np.concatenate([yTP, (y - ymid)*factor + ymid])

    # テスト点リスト作成
    TP_list = np.vstack([xTP, yTP, np.zeros_like(xTP)]).T

    # 重み配列の生成
    if weight_radius:
        weights = x.copy()
    else:
        weights = np.ones_like(x)
    if inv_weights:
        weights = 1.0 / weights
    # 閉ループ用に先頭要素を末尾に追加
    weights = np.concatenate([weights, weights[:1]])

    # 各テスト点で BiotSav を呼び出し、Z成分の絶対値を収集
    Bmag = np.array([
        abs(biot_sav(pts.tolist(), tp.tolist(),
                     current=weights, min_threshold=1e-8)[2])
        for tp in TP_list
    ])

    # 0 でない点のみ抽出してプロット
    mask = (Bmag != 0)
    Xp, Yp, Bp = xTP[mask], yTP[mask], Bmag[mask]

    # 3Dサーフェスプロット
    if plot_axes is not None:
        ax = plot_axes
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xp, Yp, Bp, cmap='jet')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('|B_z|')


def wire_near_field_B(r, wire_R, I=1.0):
    """
    WireNearField_B(r, Wire_R; I=1) の移植版
    距離 r での近傍磁場 B を返す（スカラー or ndarray 対応）
    """
    mu0 = 4 * np.pi * 1e-7
    r_arr = np.asarray(r)
    # r < wire_R
    B = np.where(r_arr < wire_R,
                 mu0 * r_arr * I / (2*np.pi * wire_R**2),
    # r == wire_R
                 np.where(r_arr == wire_R,
                          mu0 * I / (2*np.pi * wire_R),
    # r > wire_R
                          mu0 * I / (2*np.pi * r_arr)))
    return B if B.shape else float(B)


def deriv_field(r, wire_R=0.001, I=1.0):
    """
    DerivField(r, Wire_R=0.001; I=1) の移植版
    磁場の半径微分 dB/dr を返す
    """
    mu0 = 4 * np.pi * 1e-7
    r_arr = np.asarray(r)
    # r > wire_R のみ微分計算、それ以外は 0
    Bp = np.where(r_arr > wire_R,
                  -mu0 * I / (2*np.pi * r_arr**2),
                  0.0)
    return Bp if Bp.shape else float(Bp)
