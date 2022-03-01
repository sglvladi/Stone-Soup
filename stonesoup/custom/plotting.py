import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from stonesoup.base import Base, Property

# class CustomPlotter(Base):
#     meta = Property(dict, default=None)
#
#     def __init__(self, *args, **kwargs):
#         if self.meta is None:




def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_network(G, ax, node_size=0.1, width=0.5, with_labels=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx(G, pos, arrows=False, ax=ax, node_size=node_size, width=width, with_labels=with_labels)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.axis('on')
    # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


def highlight_nodes(G, ax,  nodes, node_size=0.1, node_color='m', node_shape='s', label=None):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=node_size,
                                  node_color=node_color, node_shape=node_shape, label=label)

def highlight_edges(G, ax, edges_idx, width= 2.0, edge_color='m', style='solid', arrows=False, label=None):
    edges = G.edges_by_idx(edges_idx)
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=width, edge_color=edge_color,
                                  style=style, arrows=arrows, label=label)

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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def remove_artists(artists):
    for artist in artists:
        artist.remove()


def plot_polygons(polygons, ax=None, zorder=2):
    if not ax:
        ax = plt.gca()
    lines = []
    for polygon in polygons:
        x, y = polygon.exterior.xy
        (ln,) = ax.plot(x, y, 'k-', zorder=zorder)
        lines.append(ln)
    return lines


def plot_short_paths_e(spaths, *args, **kwargs):
    edges = set()
    for key, value in spaths.items():
        edges |= set(value)
    highlight_edges(*args, edges_idx=list(edges), **kwargs)



class BlitManager:
    def __init__(self, canvas, plot_data):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in plot_data['base']:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()