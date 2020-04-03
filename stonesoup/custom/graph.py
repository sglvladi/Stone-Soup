import numpy as np
import networkx as nx
from scipy.io import loadmat

from stonesoup.functions import cart2pol, pol2cart

class CustomDiGraph(nx.DiGraph):
    def edges_by_idx(self, idx):
        if not isinstance(idx, list):
            idx = list([idx])
        edges = [e for i, e in enumerate(self.edges) if i in idx]
        return edges

    def weights_by_idx(self, idx):
        edges = self.edges_by_idx(idx)
        weights = [self.adj[e[0]][e[1]]['weight'] for e in edges]
        return weights


def load_graph_dict(path):
    wp = loadmat(path)
    s = wp['S'][0, 0]
    S = dict()
    S['Edges'] = dict()
    S['Edges']['EndNodes'] = s['Edges']['EndNodes'][0, 0] - 1
    S['Edges']['Weight'] = s['Edges']['Weight'][0, 0].ravel()
    S['Nodes'] = dict()
    S['Nodes']['Longitude'] = s['Nodes']['Longitude'][0, 0].ravel()
    S['Nodes']['Latitude'] = s['Nodes']['Latitude'][0, 0].ravel()
    return S


def dict_to_graph(S):
    num_edges = len(S['Edges']['Weight'])
    num_nodes = len(S['Nodes']['Longitude'])

    # Create empty graph object
    G = CustomDiGraph()

    # Add nodes to graph
    for i in range(num_nodes):
        G.add_node(i, pos=(S['Nodes']['Longitude'][i],
                           S['Nodes']['Latitude'][i]))

    # Add edges to graph
    for i in range(num_edges):
        edge = (S['Edges']['EndNodes'][i, 0],
                S['Edges']['EndNodes'][i, 1])
        G.add_edge(*edge, weight=S['Edges']['Weight'][i])

    return G


def shortest_path(G, source, target):
    node_path = nx.shortest_path(G, source, target, 'weight')
    path_edges = zip(node_path, node_path[1:])
    edges = [e for e in G.edges]

    edge_path = [edges.index(edge) for edge in path_edges]

    return node_path, edge_path


def get_xy_from_range_edge(r, e, G):

    r = np.atleast_1d(r)
    e = np.atleast_1d(e).astype(int)

    endnodes = G['Edges']['EndNodes'][e, :]

    # Get endnode coordinates
    p1 = np.array([G['Nodes']['Longitude'][endnodes[:,0]], G['Nodes']['Latitude'][endnodes[:,0]]])
    p2 = np.array([G['Nodes']['Longitude'][endnodes[:,1]], G['Nodes']['Latitude'][endnodes[:,1]]])

    # Normalise coordinates of p2, assuming p1 is the origin
    p2norm = p2-p1

    # Compute angle between p2 and p1
    _, theta = cart2pol(p2norm[0, :], p2norm[1, :])

    # Compute XY normalised, assuming p1 is the origin
    x_norm, y_norm = pol2cart(r, theta)
    xy_norm = np.array([x_norm, y_norm])

    # Compute transformed XY
    xy = p1 + xy_norm

    return xy
