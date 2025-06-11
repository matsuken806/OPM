import numpy as np

def vec_dist(X):
    """
    VecDist(X::Array) の移植版
    X: array-like of shape (2,3)
    戻り値: [x2-x1, y2-y1, z2-z1]
    """
    X = np.asarray(X)
    return np.array([X[1,0] - X[0,0],
                     X[1,1] - X[0,1],
                     X[1,2] - X[0,2]])


def biot_sav_closed_loop(PointPath, r, current=1.0, min_threshold=0.001):
    """
    BiotSav(PointPath, r; Current=1, MinThreshold)
    閉じたループを前提とし、PointPath の最初の点を末尾に追加して計算します。
    PointPath: ndarray of shape (N,3)
    r: array-like of length 3
    current: スカラーまたは配列
    min_threshold: ワイヤーとの最小距離
    """
    Path = np.vstack([PointPath, PointPath[0]])
    NPts = Path.shape[0]
    dB = np.zeros(3)
    r = np.asarray(r).flatten()

    # 距離チェック
    dists = np.linalg.norm(Path - r, axis=1)
    if dists.min() >= min_threshold:
        for i in range(1, NPts):
            segment = Path[i-1:i+1]     # shape (2,3)
            dL = vec_dist(segment)
            mean_pt = segment.mean(axis=0)
            Rprime = r - mean_pt
            Rdist = np.linalg.norm(Rprime)
            Rhat = Rprime / Rdist

            # current が配列かスカラーかで処理を分岐
            if hasattr(current, "__len__") and not isinstance(current, str):
                Curr = current[i]
            else:
                Curr = current

            dB += 1e-7 * Curr * np.cross(dL, Rhat) / (Rdist**2)

    return dB


def biot_sav_matrix(PointPath, dL, r, L, min_threshold=1e-5):
    """
    BiotSav(PointPath::Matrix, dL, r::Vector, L::Int; MinThreshold)
    PointPath: ndarray of shape (N,3)
    dL: ndarray of shape (N-1,3)（各セグメントのベクトル）
    r: array-like of length 3
    L: int, PointPath の行数
    min_threshold: ワイヤーとの最小距離
    """
    PointPath = np.asarray(PointPath)
    dL = np.asarray(dL)
    r = np.asarray(r).flatten()
    dB = np.zeros(3)

    # 各セグメントの中点
    mean_pts = (PointPath[1:] + PointPath[:-1]) / 2

    if min_threshold is not None:
        dists = np.linalg.norm(PointPath - r, axis=1)
        if dists.min() < min_threshold:
            return dB

    for i in range(1, L):
        Rprime = r - mean_pts[i-1]
        Rdist_sq = Rprime.dot(Rprime)
        Rhat = Rprime / np.sqrt(Rdist_sq)
        dB += 1e-7 * np.cross(dL[i-1], Rhat) / Rdist_sq

    return dB


def biot_sav_vector(PointPath, dL, r, L, min_threshold=1e-5):
    """
    BiotSav(PointPath::Vector, dL, r::Vector, L::Int; MinThreshold)
    PointPath: list of length-N vector
    dL:    list of length-(N-1) vector
    他は biot_sav_matrix と同等
    """
    # NumPy array に変換
    pts = [np.asarray(p).flatten() for p in PointPath]
    dL_arr = [np.asarray(dl).flatten() for dl in dL]
    r = np.asarray(r).flatten()
    dB = np.zeros(3)

    mean_pts = [(pts[i] + pts[i-1]) / 2 for i in range(1, len(pts))]

    if min_threshold is not None:
        dists = [np.linalg.norm(p - r) for p in pts]
        if min(dists) < min_threshold:
            return dB

    for i in range(1, L):
        Rprime = r - mean_pts[i-1]
        Rdist_sq = Rprime.dot(Rprime)
        Rhat = Rprime / np.sqrt(Rdist_sq)
        dB += 1e-7 * np.cross(dL_arr[i-1], Rhat) / Rdist_sq

    return dB


def biot_sav_matrix_wrapper(PointPath, dL, r_matrix, L, min_threshold=1e-5):
    """
    BiotSav(PointPath, dL, r::Matrix, L; MinThreshold)
    r が行列として渡される場合のラッパー
    """
    r_flat = np.asarray(r_matrix).flatten()
    return biot_sav_matrix(PointPath, dL, r_flat, L, min_threshold=min_threshold)


def biot_sav_r_matrix(PointPath, r_matrix, current=1.0, min_threshold=0.001):
    """
    BiotSav(PointPath, r::Matrix)
    r が行列として渡される場合の閉ループ版ラッパー
    """
    r_flat = np.asarray(r_matrix).flatten()
    return biot_sav_closed_loop(PointPath, r_flat,
                                current=current,
                                min_threshold=min_threshold)

# biot_sav_closed_loop, biot_sav_matrix, biot_sav_vector, biot_sav_matrix_wrapper, biot_sav_r_matrix の統合エントリポイント
import numpy as np

def biot_sav(PointPath, *args):
    """
    統合エントリポイント関数
    以下の既存関数を引数パターンに応じて呼び分けます：
      • biot_sav_closed_loop(PointPath, r, current, min_threshold)
      • biot_sav_r_matrix(PointPath, r_matrix, current, min_threshold)
      • biot_sav_matrix(PointPath, dL, r, L, min_threshold)
      • biot_sav_matrix_wrapper(PointPath, dL, r_matrix, L, min_threshold)
      • biot_sav_vector(PointPath_list, dL_list, r, L, min_threshold)
    """
    # Pythonリストならベクタ版
    if isinstance(PointPath, list):
        return biot_sav_vector(PointPath, *args)

    # NumPy配列なら次へ
    if not isinstance(PointPath, np.ndarray):
        raise TypeError("PointPath は list か numpy.ndarray である必要があります")

    total_args = 1 + len(args)  # PointPath を含めた引数総数

    # 4引数: closed_loop か r_matrix 版
    if total_args == 4:
        r_candidate, param2, param3 = args
        arr = np.asarray(r_candidate).flatten()
        # 長さ 3 → 閉ループ版
        if arr.size == 3:
            r = r_candidate
            current = param2
            min_threshold = param3
            return biot_sav_closed_loop(PointPath, r,
                                        current=current,
                                        min_threshold=min_threshold)
        # 長さ ≠3 → r_matrix ラッパー版
        else:
            r_matrix = r_candidate
            current = param2
            min_threshold = param3
            return biot_sav_r_matrix(PointPath, r_matrix,
                                     current=current,
                                     min_threshold=min_threshold)

    # 5引数: matrix 版 or matrix_wrapper 版
    elif total_args == 5:
        dL, r_candidate, L, min_threshold = args
        arr = np.asarray(r_candidate).flatten()
        # 長さ 3 → 行列版
        if arr.size == 3:
            return biot_sav_matrix(PointPath, dL, r_candidate,
                                   L, min_threshold=min_threshold)
        # 長さ ≠3 → matrix_wrapper 版
        else:
            return biot_sav_matrix_wrapper(PointPath, dL, r_candidate,
                                           L, min_threshold=min_threshold)

    else:
        raise ValueError(f"biot_sav: 不正な引数パターン (total_args={total_args})")


    