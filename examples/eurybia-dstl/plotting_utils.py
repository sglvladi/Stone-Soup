import numpy as np
import pymap3d as pm
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def plot_gnd(tracks, ref_lat, ref_lon, ax, coord='gps'):
    for track in tracks:
        # x = [state.state_vector[0] for state in track.states]
        # y = [state.state_vector[2] for state in track.states]
        x = track.state_vector[0]
        y = track.state_vector[2]

        lat = x.copy()
        lon = y.copy()

        if coord == 'gps':
            for ii in range(0, len(x)):
                out = pm.enu2geodetic(x[ii], y[ii], 0, ref_lat, ref_lon, 0)
                lat[ii] = out[0]
                lon[ii] = out[1]

            ax.plot(lon, lat, 'k.', linewidth=5, label='Target Truth')

        elif coord == 'xyz':
            ax.plot(x, y, 'k.', linewidth=5, label='Target Truth')


def plot_platform(states, ref_lat, ref_lon, ax, coord='gps', color='b-', lab='platform'):
    x = [state[0] for state in states]
    y = [state[1] for state in states]

    lat = x.copy()
    lon = y.copy()

    if coord == 'gps':
        for ii in range(0, len(x)):
            out = pm.enu2geodetic(x[ii], y[ii], 0, ref_lat, ref_lon, 0)
            lat[ii] = out[0]
            lon[ii] = out[1]

        # plt.plot(lon, lat, color, label=lab, linewidth = 4)
        ax.plot(lon, lat, 'o', color=color, label=lab, markersize=4)

    elif coord == 'xyz':
        # plt.plot(x, y, color, label=lab, linewidth = 4)
        ax.plot(x, y, 'o', color=color, label=lab, markersize=4)


def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_tracks(tracks, show_error=True, ax=None):
    for track in tracks:
        data = np.array([state.state_vector for state in track.states])
        if ax is not None:
            ax.plot(data[:, 0], data[:, 2], 'r-')
        else:
            plt.plot(data[:, 0], data[:, 2], 'r-')
        if show_error:
            plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                             track.state.mean[[0, 2], :], edgecolor='r',
                             facecolor='none', ax=ax)