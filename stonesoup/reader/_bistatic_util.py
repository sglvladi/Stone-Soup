"""
MIT License

© Crown Copyright 2025 Defence Science and Technology Laboratory UK


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: cmcnicol
"""


import numpy as np
import logging
from scipy.optimize import fsolve
import collections

from geographiclib.geodesic import Geodesic



# SOME USEFUL GEODESIC CALCULATIONS


def calc_geo_range(geo_pos_one, geo_pos_two):
    """
    Calculates range in metres between two geographic locations.
    Parameters
    ----------
    geo_pos_one: tuple / numpy.ndarray
        Geographic position as (lat, long, (depth-optional))
    geo_pos_two: tuple / numpy.ndarray
        Geographic position as (lat, long, (depth-optional))
    Returns
    -------
    range: float [m]
    """
    plan_distance = abs(Geodesic.WGS84.Inverse(geo_pos_one[0], geo_pos_one[1], geo_pos_two[0], geo_pos_two[1])['s12'])
    tot_distance = np.sqrt(plan_distance**2 + (geo_pos_two[2] - geo_pos_one[2])**2)
    return tot_distance


def angle_diff_deg(theta1, theta2):
    """
    Calculate the difference between two bearings in degrees.
    Parameters
    ----------
    theta1: float [degrees]
    theta2: float [degrees]
    Returns
    -------
    delta_theta: float [degrees]
    """
    delta_theta = abs((theta2 % 360) - (theta1 % 360))
    return delta_theta if delta_theta < 180 else delta_theta-180



def geo_to_cartesian_pos(geo_pos, origin_pos):
    """
    Calculate a cartesian position from a geographic position and an origin. +ve X is east, +ve y is north. Depth is unchanged.
    Parameters
    ----------
    geo_pos: tuple / numpy.ndarray
        Geographic position of "target" as (lat, long, depth)
    origin_pos: tuple / numpy.ndarray
        Geographic position of origin as (lat, long, depth)
    Returns
    -------
    x,y,z: float, float, float [m]
    """
    inv_result = Geodesic.WGS84.Inverse(origin_pos[0], origin_pos[1], geo_pos[0], geo_pos[1])
    bearing, distance = inv_result['azi1'], inv_result['s12']
    x, y, z = distance*np.sin(np.deg2rad(bearing)), distance*np.cos(np.deg2rad(bearing)), geo_pos[2]
    return x, y, z

def cartesian_to_geo(cart_pos, origin_pos):
    """
    Calculate a geographic position from a cartesian one, given an origin position.  +ve X is east, +ve y is north. Depth is unchanged.
    Parameters
    ----------
    cart_pos: tuple / numpy.ndarray
        Cartesian position as (X, Y, depth)
    origin_pos: tuple / numpy.ndarray
        Geographic position of origin as (lat, long, depth)

    Returns
    -------

    """
    azi, dist = np.rad2deg(np.arctan2(cart_pos[0], cart_pos[1])), np.sqrt(cart_pos[0]**2 + cart_pos[1]**2)
    result = Geodesic.WGS84.Direct(origin_pos[0], origin_pos[1], azi, dist)
    lat, lon, depth = result['lat2'], result['lon2'], cart_pos[2]
    return lat, lon, depth


def geo_pos_from_delta_pos(ref_pos, delta_north_metres, delta_east_metres):
    """
    Calculate a geographic position from a reference position, and a delta in north and east.
    Parameters
    ----------
    ref_pos: tuple / numpy.ndarray
        Geographic position of origin as (lat, long, (depth-optional))
    delta_north_metres: float [m]
        Number of metres north of reference position (+ve is north, -ve is south)
    delta_east_metres: float [m]
        Number of metres east of reference position (+ve is east, -ve is west)
    Returns
    -------
    lat, long : float
    """
    azi = np.arctan2(delta_east_metres, delta_north_metres)
    delta_range = np.sqrt(delta_east_metres**2 + delta_north_metres**2)
    result = Geodesic.WGS84.Direct(ref_pos[0], ref_pos[1], azi, delta_range)
    return result['lat2'], result['lon2']





# BISTATIC CALCULATIONS



def bistatic(c,t,d,a):
    """
    Parameters
    ----------
    c : float
        speed of sound (m/s)
    t : ndarray
        time delay (s)
        if t < 0 it is assumed as (minus) time after direct blast arrival
        if t >= 0 it is time after transmission occurred
    d: ndarray
        range from Tx to Rx (m)
    a: ndarray
        angle in triangle described as Tx-Tx-Contact (radians)
    

    Returns
    -------
    r : ndarray
        range from receiver Rx to contact
        
    To localise contact, you can use the calculated range with Rx location and 
    
    Note, this is an instantaneous calculation and hasn't accounted for movement of platforms/contacts while sound has travelled from TX to RX
    Could use TX location at time of transmission and RX location at time of receive to calculate with this formula, noting it is not exact location on contact due to motion
    """
    
    if np.any(t>=0):
        # use conventional bistatic equation (cosine rule)
        r = (c**2.*t**2 - d**2)/(2*(c*t - d*np.cos(a)))
    else:
        # this assumes that t is time after direct blast
        t = -t + d/c
        r = (c**2.*t**2 - d**2)/(2*(c*t - d*np.cos(a)))
        
    return r


def bistatic_range(true_brg_deg, tx_pos, rx_pos, delta_time, ref_sound_spd):
    """
    Calculate the bistatic range from a true bearing, and tx/rx cartesian positions and a delta time after the direct blast detection.
    Parameters
    ----------
    true_brg_deg: float [degrees]
        True bearing of detection from rx_pos
    tx_pos: tuple / numpy.ndarray
        Cartesian position of transmitter as (X, Y, (depth-optional)) in metres
    rx_pos: tuple / numpy.ndarray
        Cartesian position of receiver as (X, Y, (depth-optional)) in metres
    delta_time: float [s]
        Difference in time between detection and detection of direct blast
    ref_sound_spd: float [ms^-1]
        Speed of sound (default 1500ms^-1)
    Returns
    -------
    bistatic_range: float [m]
    """

    vec = tx_pos - rx_pos
    rx2tx_bearing = np.rad2deg(np.arctan2(vec[0], vec[1])) % 360
    rx2tx_dist = np.linalg.norm(vec)

    bistat_angle = rx2tx_bearing - true_brg_deg
    cosine_angle = np.cos(np.deg2rad(bistat_angle))
    # bistatic_range = ((ref_sound_spd**2 * delta_time**2) - rx2tx_dist**2)/(2*(ref_sound_spd*delta_time - rx2tx_dist*cosine_angle))
    cdt = ref_sound_spd * delta_time
    return 0.5 * (cdt + (2 * rx2tx_dist)) / (1 + (rx2tx_dist / cdt) * (1 - cosine_angle))


def two_d_bistatic(true_brg_deg, tx_pos, rx_pos, delta_time, ref_sound_spd):
    """
    Calculate cartesian position of detection using 2D bistatic calculation
    Parameters
    ----------
    true_brg_deg: float [degrees]
        True bearing of detection from rx_pos
    tx_pos: tuple / numpy.ndarray
        Cartesian position of transmitter as (X, Y, depth) in metres
    rx_pos: tuple / numpy.ndarray
        Cartesian position of receiver as (X, Y, depth) in metres
    delta_time: float [s]
        Difference in time between detection and detection of direct blast
    ref_sound_spd: float [ms^-1]
        Speed of sound (default 1500ms^-1)
    Returns
    -------
    detection_pos: numpy.ndarray
        Detection position as (X, Y, depth) in metres
    """
    bistat_range = bistatic_range(true_brg_deg, tx_pos, rx_pos, delta_time, ref_sound_spd)

    return np.array([rx_pos[0] + bistat_range * np.sin(np.deg2rad(true_brg_deg)),
                  rx_pos[1] + bistat_range * np.cos(np.deg2rad(true_brg_deg)),
                  rx_pos[2]])


def get_cone_angle(rel_vec, ta_hdg, truth_rel_bearing=None):
    """
    Get cone angle between a position and the line vector of a towed array.
    Parameters
    ----------
    rel_vec: numpy.ndarray
        Cartesian position vector relative to RX position
    ta_hdg: float [degrees]
        Towed array heading in degrees
    truth_rel_bearing: float [degrees]
        Detected relative bearing - used for disambiguating left-right
    Returns
    -------
    cone_angle: float [degrees]
    """
    ta_orientation = np.array([np.sin(np.deg2rad(ta_hdg)), np.cos(np.deg2rad(ta_hdg)),0])
    inner_prod = np.dot(rel_vec / np.linalg.norm(rel_vec), ta_orientation / np.linalg.norm(ta_orientation))
    cone_angle = np.rad2deg(np.arccos(inner_prod)) % 360
    other_cone_angle = 360 - cone_angle
    return min(cone_angle, other_cone_angle) if (truth_rel_bearing is None or truth_rel_bearing <= 180) else max(cone_angle, other_cone_angle)

def get_rel_bearing_deg(contact_pos, rx_pos, ta_orientation):
    """
    Get relative bearing in degrees between given a towed-array line vector
    Parameters
    ----------
    contact_pos: numpy.ndarray
        Cartesian position of detection relative to rx_pos
    rx_pos: numpy.ndarray
        Cartesian position of receiver
    ta_orientation: numpy.ndarray
        Normalised cartesian line vector of towed array
    Returns
    -------
    rel_brg: float [degrees]
    """
    rel_vec = contact_pos - rx_pos
    true_bearing = np.arctan2(rel_vec[0], rel_vec[1])
    ta_hdg = np.arctan2(ta_orientation[0], ta_orientation[1])
    return np.rad2deg(true_bearing - ta_hdg) % 360

def three_d_bistatic(rel_brg_deg, depth_hypo, tx_pos, rx_pos, ref_sound_spd, delta_time, rx_hdg, use_conical_beams=False):
    """
    Calculate the cartesian position of detection using a 3D bistatic calculation
    Parameters
    ----------
    rel_brg_deg: float [degrees]
        Relative bearing of detection from RX platform, in degrees
    depth_hypo: float [degrees]
        Hypothesis depth of target.
    tx_pos: tuple / numpy.ndarray
        Cartesian position of transmitter as (X, Y, depth) in metres
    rx_pos: tuple / numpy.ndarray
        Cartesian position of transmitter as (X, Y, depth) in metres
    ref_sound_spd: float [ms^-1]
        Speed of sound
    delta_time: float [s]
        Difference in time between detection and detection of direct blast
    rx_hdg: float [degrees]
        Heading of RX platform in degrees
    use_conical_beams: bool
        Option to correct for coning errors of conical beams. Default False

    Returns
    -------
    detection_pos: numpy.ndarray
        Detection position as (X, Y, depth) in metres.
    """
    true_brg_deg = (rel_brg_deg + rx_hdg) % 360
    approx_pos = two_d_bistatic(true_brg_deg=true_brg_deg, tx_pos=tx_pos, rx_pos=rx_pos,
                                           delta_time=delta_time, ref_sound_spd=1500.0)
    x_approx, y_approx = approx_pos[0], approx_pos[1]
    ta_orientation = np.array([np.sin(np.deg2rad(rx_hdg)), np.cos(np.deg2rad(rx_hdg)), 0])

    ## Ellipse parameters
    foci_vector = tx_pos - rx_pos
    # delta_time is difference in direct blast and echo time (*not* the total propagation time)
    #tot_range = ref_sound_spd * delta_time
    tot_range = np.linalg.norm(foci_vector) + ref_sound_spd*delta_time

    ellipse_a = 0.5 * tot_range
    ellipse_b = 0.5 * np.sqrt(tot_range**2 - np.linalg.norm(foci_vector)**2)
    ellipse_centre = (tx_pos + rx_pos ) /2
    # tx_pos_in_ellipse_coords = tx_pos - ellipse_centre
    rx_pos_in_ellipse_coords = rx_pos - ellipse_centre

    ## Coordinate transforms
    alpha = np.arctan2(foci_vector[1], foci_vector[0])
    beta = np.arctan2(float(rx_pos_in_ellipse_coords[2]), np.sqrt(rx_pos_in_ellipse_coords[0 ]**2 + rx_pos_in_ellipse_coords[1 ]**2))
    sin_a, cos_a, sin_bet, cos_bet = np.sin(alpha), np.cos(alpha), np.sin(beta), np.cos(beta)
    rot_mat = np.array([[cos_a * cos_bet, -sin_a, cos_a * sin_bet],
                        [sin_a * cos_bet, cos_a, sin_a * sin_bet],
                        [-sin_bet, 0, cos_bet]])

    def solve_ellipse_and_cone(x):
        x_prime, y_prime = x[0], x[1]
        r_prime = np.array([x_prime, y_prime, depth_hypo], dtype=object)
        r_prime_ellipse = r_prime + rx_pos_in_ellipse_coords
        #r_prime_ellipse = r_prime
        r_ellipse = np.dot(np.linalg.inv(rot_mat), r_prime_ellipse)
        ellipse_mat = np.array([[ 1 /ellipse_a**2, 0, 0], [0, 1/ ellipse_b ** 2, 0], [0, 0, 1 / ellipse_b ** 2]])
        ellipse_val = np.dot(r_ellipse.T, np.dot(ellipse_mat, r_ellipse)) - 1.0
        tot_dist = np.linalg.norm(r_prime - tx_pos) + np.linalg.norm(r_prime - rx_pos)
        #ellipse_val = abs(tot_range - tot_dist)

        if use_conical_beams:
            #intersect_val = angle_diff_deg(np.rad2deg(get_cone_angle(r_prime, ta_orientation)[0]), rel_brg_deg)
            intersect_val = angle_diff_deg(get_cone_angle(r_prime, rx_hdg, rel_brg_deg), rel_brg_deg)
        else:
            intersect_val = angle_diff_deg(get_rel_bearing_deg(r_prime, rx_pos, ta_orientation=ta_orientation), rel_brg_deg)
        return ellipse_val, intersect_val

    roots = fsolve(solve_ellipse_and_cone, [x_approx, y_approx])
    return np.array([roots[0], roots[1], depth_hypo]) + rx_pos



def bistatic_geo_coords(tx_pos_geo, rx_pos_geo, ref_pos=None, delta_time=0, rel_brg_deg=0, depth_hypo = None, rx_hdg = None,
                        use_conical_beams=False, ref_sound_spd=1500.0, max_tx_rx_sep_km=200.0, use_three_d_bistatic=False,
                        range_only=False):
    """
    Calculate geographic position of a bistatic detection from the transmitter and receiver positions, relative bearing and delta time.
    Parameters
    ----------
    tx_pos_geo: tuple/ numpy.ndarray
        Geographic position of transmitter as (lat, long, depth)
    rx_pos_geo: tuple / numpy.ndarray
        Geographic position of receiver as (lat, long, depth)
    ref_pos: tuple / numpy.ndarray
        Reference position to use for cartesian conversion. Default is None, in which case rx_pos_geo is used.
    delta_time: float [s]
        Difference in time between detection and direct blast detection
    rel_brg_deg: float / List[float] [degrees]
        Relative bearing of detection from RX platform
    depth_hypo: float [m]
        Depth hypothesis of target. Default is None in which case depth of receiver is used.
    rx_hdg: float [degrees]
        Heading of receiving platform
    use_conical_beams: bool
        Whether to correct for coning errors. Default False
    ref_sound_spd: float [ms^-1]
        Speed of sound. Default 1500.0ms^-1
    max_tx_rx_sep_km: float [km]
        Maximum possible separation between transmitter and receiver, if the calculated difference is larger, skip the calculation
    use_three_d_bistatic: bool
        Option to use a 3D bistatic calculation. Default False
    range_only: bool
        Option to extract only the bistatic range, rather than the full position
    Returns
    -------
    detection_loc: numpy.ndarray
        Geographic position of detection


    """
    #Sanity check, if distance between source and receiver too large, skip
    logging.debug(f"Bistatic calculation TX {tx_pos_geo}  RX {rx_pos_geo}  DeltaTime {delta_time}    RelBrg  {rel_brg_deg}   RxHdg {rx_hdg}")
    tx_rx_sep = calc_geo_range(tx_pos_geo, rx_pos_geo)
    ref_pos = rx_pos_geo if ref_pos is None else ref_pos
    if tx_rx_sep > max_tx_rx_sep_km*1000:
        logging.warning(f"Requested Large Tx-Rx separation ({tx_rx_sep/1000:.1} km), perhaps your positions are wrong? Skipping")
        return []

    if delta_time < 0:
        logging.warning(f"Unphysical, negative delta time requested in Bistatic Calculation, skipping")
        return []
    
    # If only a single value of rel_brg_deg given, treat it as a list
    if isinstance(rel_brg_deg, (int, float)):
        rel_brg_deg_list = [rel_brg_deg]
    elif isinstance(rel_brg_deg, (collections.Sequence, np.ndarray)):
        rel_brg_deg_list = rel_brg_deg

    result_list = []
    for rel_brg_deg in rel_brg_deg_list:
        rx_pos_cart = np.array(geo_to_cartesian_pos(rx_pos_geo, ref_pos))
        tx_pos_cart = np.array(geo_to_cartesian_pos(tx_pos_geo, ref_pos))
        true_brg_deg = (rel_brg_deg + rx_hdg) % 360

        if range_only:
            result_list.append(bistatic_range(true_brg_deg=true_brg_deg, tx_pos=tx_pos_cart, rx_pos=rx_pos_cart,
                                delta_time=delta_time, ref_sound_spd= ref_sound_spd))

        elif use_three_d_bistatic:
            depth_hypo = rx_pos_geo[2] if depth_hypo is None else depth_hypo
            det_loc = three_d_bistatic(rel_brg_deg, depth_hypo, tx_pos_cart, rx_pos_cart, ref_sound_spd, delta_time,
                                                rx_hdg, use_conical_beams)
            result_list.append(np.array(cartesian_to_geo(det_loc, ref_pos)))
        else:
            det_loc = two_d_bistatic(true_brg_deg=true_brg_deg, tx_pos=tx_pos_cart, rx_pos=rx_pos_cart,
                                    delta_time=delta_time, ref_sound_spd=ref_sound_spd)
            result_list.append(np.array(cartesian_to_geo(det_loc, ref_pos)))
        
    
    return result_list