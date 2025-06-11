import numpy as np

def _unique_rows(a):
    """
    Preserve-order unique rows for an (M×N) array.
    """
    seen = set()
    unique = []
    for row in a:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return np.array(unique)

def make_rect_point_path(coords, nelement=50, plot_on=False):
    """
    Port of MakeRectPointPath(Coords; NElement=50, PlotOn=false)
    Returns an (M×3) ndarray of evenly‐sampled points along the closed rectangular path.
    """
    coords = np.asarray(coords, dtype=float)
    # Close the loop
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    x_up = [x[0]]; y_up = [y[0]]; z_up = [z[0]]
    for i in range(len(x) - 1):
        xs = np.linspace(x[i],   x[i+1], nelement)
        ys = np.linspace(y[i],   y[i+1], nelement)
        zs = np.linspace(z[i],   z[i+1], nelement)
        x_up.extend(xs.tolist()); x_up.append(x[i+1])
        y_up.extend(ys.tolist()); y_up.append(y[i+1])
        z_up.extend(zs.tolist()); z_up.append(z[i+1])

    pts = np.column_stack((x_up, y_up, z_up))
    point_path = _unique_rows(pts)

    if plot_on:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_path[:,0], point_path[:,1], point_path[:,2])
        plt.show()

    return point_path

def make_rect_saddle_point_path(coords, nelement=50, plot_on=False, connect_first_last_points=True):
    """
    Port of MakeRectSaddlePointPath(Coords; NElement=50, PlotOn=false, ConnectFirstLastPoints=true)
    Returns an (M×3) ndarray of Cartesian coordinates for a rectangular saddle coil.
    Input Coords: (N×3) array of [θ (deg), r, z].
    """
    coords = np.asarray(coords, dtype=float)
    if connect_first_last_points and not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    θ = coords[:,0]
    r = coords[:,1]
    z = coords[:,2]

    θ_up = [θ[0]]; r_up = [r[0]]; z_up = [z[0]]
    for i in range(len(θ) - 1):
        thetas = np.linspace(θ[i],   θ[i+1], nelement)
        rs     = np.linspace(r[i],   r[i+1], nelement)
        zs     = np.linspace(z[i],   z[i+1], nelement)
        θ_up.extend(thetas.tolist()); θ_up.append(θ[i+1])
        r_up.extend(rs.tolist());     r_up.append(r[i+1])
        z_up.extend(zs.tolist());     z_up.append(z[i+1])

    pts_polar = np.column_stack((θ_up, r_up, z_up))
    pts_polar = _unique_rows(pts_polar)

    # Convert polar → Cartesian
    θ_rad = np.deg2rad(pts_polar[:,0])
    x = pts_polar[:,1] * np.cos(θ_rad)
    y = pts_polar[:,1] * np.sin(θ_rad)
    z_coord = pts_polar[:,2]

    point_path = np.column_stack((x, y, z_coord))

    if plot_on:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_path[:,0], point_path[:,1], point_path[:,2])
        plt.show()

    return point_path

# Alias for convenience
make_polar_rect_point_path = make_rect_saddle_point_path

def field_saddle_centered(h, D, phi):
    """
    Port of FieldSaddleCentered(h,D,ϕ)
    Returns the centered saddle‐coil field B₀ for height h, diameter D, and angle φ (radians).
    """
    mu0 = 4 * np.pi * 1e-7
    s = 1 + (h / D)**2
    return 4 / np.pi * mu0 * (h / D**2) * (s**(-0.5) + s**(-1.5)) * np.sin(phi / 2)
