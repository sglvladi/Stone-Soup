from functools import partial
import math

import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag
import pyproj
from shapely import Polygon
from shapely.geometry import Point
from shapely.ops import transform
from geopy.distance import geodesic
from geopy import Point as GeoPoint


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')


def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    return transform(project, buf)


def predict_state_to_two_state(old_mean, old_cov, tx_model, dt):
    A = tx_model.matrix(time_interval=dt)
    Q = tx_model.covar(time_interval=dt)
    statedim = A.shape[0]
    AA = np.concatenate((np.eye(statedim), A))
    QQ = block_diag(np.zeros((statedim, statedim)), Q)
    return AA @ old_mean, AA @ old_cov @ AA.T + QQ


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def cover_rectangle_with_minimum_overlapping_circles(x1, y1, x2, y2, radius):
    """
    https://ieeexplore.ieee.org/document/1343643

    """
    width = x2 - x1
    height = y2 - y1

    p = Point(x1 + width/2, y1 + height/2).buffer(radius)
    pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersection = p.intersection(pol)
    if intersection.area >= 0.9*pol.area:
        return [(x1 + width/2, y1 + height/2)]

    # if width <= np.sqrt(3)/2*radius and height <= np.sqrt(3)/2*radius:
    z1 = height / (np.sqrt(3) * radius)
    re1 = z1 - math.floor(z1)
    n = math.floor(z1)
    if re1 <= 1/2:
        n += 1
    else:
        n += 2

    z2 = width / (3/2 * radius)
    re2 = z2 - math.floor(z2)
    m = math.floor(z2)
    if re2 <= 2/3:
        m += 1
    else:
        m += 2

    centers = []

    for k in range(1, n+1):
        for l in range(1, m+1):
            if l % 2 == 1:
                center = ((0.5 + (l-1) * 3/2) * radius, (k-1)*np.sqrt(3)*radius)
            else:
                center = ((0.5 + (l-1) * 3/2) * radius, (k-1)*np.sqrt(3)*radius + np.sqrt(3)/2*radius)
            offset_center = (center[0] + x1, center[1] + y1)
            cp = Point(offset_center)
            if cp.distance(pol) <= np.sqrt(3)/2*radius:
                centers.append(offset_center)
    return centers


def calculate_bearing(lon1, lat1, lon2, lat2):
    """Calculate the initial bearing from geo point A to point B"""
    # bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']
    # bearing = (bearing + 360) % 360

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360  # Normalize to 0-360 degrees

    return bearing


def compute_reachable_point(lon1, lat1, lon2, lat2, max_speed, time):
    """
    Compute the furthest point reachable within a given time and maximum speed
    between two latitude/longitude positions.

    Parameters
    ----------
    lon1 : float
        Longitude of the starting point.
    lat1 : float
        Latitude of the starting point.
    lon2 : float
        Longitude of the destination point.
    lat2 : float
        Latitude of the destination point.
    max_speed : float
        Maximum speed of the vehicle in m/s.
    time : float
        Time limit in seconds.

    Returns
    -------
    float, float
        Longitude and latitude of the reachable point.
    """
    # Calculate the distance that can be traveled within the time limit
    max_distance = max_speed * time / 1000  # distance = speed * time, in kilometers

    # Starting and destination points
    start = GeoPoint(lat1, lon1)
    destination = GeoPoint(lat2, lon2)

    # Calculate the total distance between the start and destination
    total_distance = geodesic(start, destination).kilometers

    # If max_distance is greater than or equal to total_distance, return the destination
    if np.isclose(max_distance, total_distance) or max_distance > total_distance:
        return lon2, lat2

    # Calculate the bearing from start to destination
    bearing = calculate_bearing(lon1, lat1, lon2, lat2)
    print(bearing)

    # Calculate the reachable point
    reachable_point = geodesic(kilometers=max_distance).destination(start, bearing)

    return reachable_point.longitude, reachable_point.latitude