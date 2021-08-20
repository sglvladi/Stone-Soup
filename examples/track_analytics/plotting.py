import numpy as np
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_map(timestamp, limits, res, target):
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = limits
    # Mercator projection map
    m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                urcrnrlat=LAT_MAX, projection='merc', resolution=res)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    m.drawparallels(np.arange(-90., 91., 20.), labels=[1, 1, 0, 0])
    m.drawmeridians(np.arange(-180., 181., 20.), labels=[0, 0, 0, 1])
    plt.title(
        'Exact Earth AIS dataset Tracking\n'
        + "({})\n".format(target)
        + timestamp.strftime('%H:%M:%S %d/%m/%Y'))


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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
                    **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_tracks(tracks, limits, res, show_mmsis=False, show_probs=False,
                show_error=False, show_map=False):

    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = limits

    # Mercator projection map
    if show_map:
        m = Basemap(llcrnrlon=LON_MIN, llcrnrlat=LAT_MIN, urcrnrlon=LON_MAX,
                    urcrnrlat=LAT_MAX, projection='merc', resolution=res)
    if show_mmsis:
        mmsis = list({track.metadata["MMSI"] for track in tracks})
    for track in tracks:
        states = [state.state_vector for state in track.states]
        if len(states) == 0:
            continue
        data = np.array(states)
        lat = np.rad2deg(data[:, 2].astype(float))
        lon = np.rad2deg(data[:, 0].astype(float))
        if show_map:
            x, y = m(lon, lat)
            m.plot(x, y, 'b-o', linewidth=1, markersize=1)
            m.plot(x[-1], y[-1], 'ro', markersize=1)
            # plt.text(x[-1], y[-1], track.metadata["Vessel_Name"], fontsize=12)
            if show_probs:
                plt.text(x[-1], y[-1],
                         np.around(track.last_update.weights[0, 0], 2),
                         fontsize=6)
            elif show_mmsis:
                ind = mmsis.index(track.metadata["MMSI"])
                plt.text(x[-1], y[-1],
                         track.metadata["MMSI"],
                         fontsize=6)
        else:
            plt.plot(data[:, 0], data[:, 2], 'b-o', linewidth=1, markersize=1, label="AIS Tracks")
            plt.plot(data[-1, 0], data[-1, 2], 'ro', markersize=1)
            if show_error:
                plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                                 track.state.mean[[0, 2], :], edgecolor='r',
                                 facecolor='none')

    shared_mmsi = dict()
    examined_tracks = set()
    for track in tracks:
        examined_tracks.add(track)
        for track2 in tracks - examined_tracks:
            if track.metadata["MMSI"] == track2.metadata["MMSI"]:
                mmsi = track.metadata["MMSI"]
                if mmsi in shared_mmsi:
                    if track not in shared_mmsi[mmsi]:
                        shared_mmsi[mmsi].append(track)
                    if track2 not in shared_mmsi[mmsi]:
                        shared_mmsi[mmsi].append(track2)
                else:
                    shared_mmsi[mmsi] = [track, track2]

    for mmsi in shared_mmsi:
        states = [track.state.state_vector for track in shared_mmsi[mmsi]]
        data = np.array(states)
        lat = np.rad2deg(data[:, 2].astype(float))
        lon = np.rad2deg(data[:, 0].astype(float))
        if show_map:
            x, y = m(lon, lat)
            m.plot(x, y, 'y-o', linewidth=0.5, markersize=1)
        else:
            plt.plot(data[:, 0], data[:, 2], 'g-o', label="AIS Tracks")
            plt.text(data[0, 0], data[0, 2], mmsi, fontsize=12)


def plot_data(detections=None):
    if len(detections) > 0:
        x = [s.state_vector[0] for s in detections]
        y = [s.state_vector[1] for s in detections]
        plt.plot(x, y, linestyle='', marker='x')