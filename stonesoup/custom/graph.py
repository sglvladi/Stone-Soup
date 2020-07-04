import pickle

import numpy as np
import networkx as nx
from scipy.io import loadmat

from stonesoup.functions import cart2pol, pol2cart
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

from shapely.geometry import LineString
from shapely.geometry import Point
import sympy

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


def shortest_path(G, sources=None, targets=None, cache_filepath=None):
    edges = [e for e in G.edges]

    short_paths_n = dict()
    short_paths_e = dict()

    multi_source = isinstance(sources, list)
    multi_target = isinstance(targets, list)

    if cache_filepath is None:
        if sources is None or targets is None:
            edges_index = {edge: edges.index(edge) for edge in edges}
            if sources is None and targets is None:
                paths = nx.shortest_path(G, weight='weight')
                for s, dic in paths.items():
                    for t, node_path in dic.items():
                        path_edges = zip(node_path, node_path[1:])
                        edge_path = [edges_index[edge] for edge in path_edges]
                        short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
            elif sources is None:
                if multi_target:
                    for t in targets:
                        paths = nx.shortest_path(G, target=t, weight='weight')
                        for s, node_path in paths.items():
                            path_edges = zip(node_path, node_path[1:])
                            edge_path = [edges_index[edge] for edge in path_edges]
                            short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
                else:
                    t = targets
                    paths = nx.shortest_path(G, target=t, weight='weight')
                    for s, node_path in paths.items():
                        path_edges = zip(node_path, node_path[1:])
                        edge_path = [edges_index[edge] for edge in path_edges]
                        short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
            elif targets is None:
                if multi_source:
                    for s in sources:
                        paths = nx.shortest_path(G, source=s, weight='weight')
                        for t, node_path in paths.items():
                            path_edges = zip(node_path, node_path[1:])
                            edge_path = [edges_index[edge] for edge in path_edges]
                            short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
                else:
                    s = sources
                    paths = nx.shortest_path(G, source=s, weight='weight')
                    for t, node_path in paths.items():
                        path_edges = zip(node_path, node_path[1:])
                        edge_path = [edges_index[edge] for edge in path_edges]
                        short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
        else:
            if multi_source and multi_target:
                edges_index = {edge: edges.index(edge) for edge in edges}
                for s in sources:
                    for t in targets:
                        try:
                            node_path = nx.shortest_path(G, s, t, 'weight')
                            path_edges = zip(node_path, node_path[1:])
                            edge_path = [edges_index[edge] for edge in path_edges]
                        except (nx.NetworkXNoPath):
                            node_path = []
                            edge_path = []
                        short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
            elif multi_source:
                edges_index = {edge: edges.index(edge) for edge in edges}
                t = targets
                for s in sources:
                    try:
                        node_path = nx.shortest_path(G, s, t, 'weight')
                        path_edges = zip(node_path, node_path[1:])
                        edge_path = [edges_index[edge] for edge in path_edges]
                    except (nx.NetworkXNoPath):
                        node_path = []
                        edge_path = []
                    short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
            elif multi_target:
                edges_index = {edge: edges.index(edge) for edge in edges}
                s = sources
                for t in targets:
                    try:
                        node_path = nx.shortest_path(G, s, t, 'weight')
                        path_edges = zip(node_path, node_path[1:])
                        edge_path = [edges_index[edge] for edge in path_edges]
                    except (nx.NetworkXNoPath):
                        node_path = []
                        edge_path = []
                    short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
            else:
                s = sources
                t = targets
                try:
                    node_path = nx.shortest_path(G, s, t, 'weight')
                    path_edges = zip(node_path, node_path[1:])
                    edge_path = [edges.index(edge) for edge in path_edges]
                except (nx.NetworkXNoPath):
                    node_path = []
                    edge_path = []
                return node_path, edge_path
    else:
        with open(cache_filepath, 'rb') as f:
            s_paths_e, s_paths_n = pickle.load(f)

        if sources is None or targets is None:
            if sources is None and targets is None:
                return s_paths_n, short_paths_e
            elif sources is None:
                if multi_target:
                    keys = [key for key in s_paths_e if key[1] in targets]
                else:
                    t = targets
                    keys = [key for key in s_paths_e if key[1]==t]
            elif targets is None:
                if multi_source:
                    keys = [key for key in s_paths_e if key[0] in sources]
                else:
                    s = sources
                    keys = [key for key in s_paths_e if key[0] == s]
        else:
            if multi_source and multi_target:
                keys = [key for key in s_paths_e if (key[0] in sources and key[1] in targets)]
            elif multi_source:
                t = targets
                keys = [key for key in s_paths_e if (key[0] in sources and key[1] == t)]
            elif multi_target:
                s = sources
                keys = [key for key in s_paths_e if (key[0] == s and key[1] in targets)]
            else:
                s = sources
                t = targets
                short_paths_n = s_paths_n[(s,t)]
                short_paths_e = s_paths_e[(s,t)]
                return short_paths_n, short_paths_e

        short_paths_n = {key: value for key, value in s_paths_n if key in keys}
        short_paths_e = {key: value for key, value in s_paths_e if key in keys}
    return short_paths_n, short_paths_e


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


def get_xy_from_sv(sv: StateVector, short_paths, G):

    r = np.atleast_1d(sv[0, :])
    e = np.atleast_1d(sv[2, :]).astype(int)
    d = np.atleast_1d(sv[3, :]).astype(int)
    s = np.atleast_1d(sv[4, :]).astype(int)

    for i in range(sv.shape[1]):
        r[i], e[i], d[i], s[i] = normalise_re(r[i], e[i], d[i], s[i], short_paths, G)

    endnodes = G['Edges']['EndNodes'][e, :]

    # Get endnode coordinates
    p1 = np.array(
        [G['Nodes']['Longitude'][endnodes[:, 0]], G['Nodes']['Latitude'][endnodes[:, 0]]])
    p2 = np.array(
        [G['Nodes']['Longitude'][endnodes[:, 1]], G['Nodes']['Latitude'][endnodes[:, 1]]])

    # Normalise coordinates of p2, assuming p1 is the origin
    p2norm = p2 - p1

    # Compute angle between p2 and p1
    _, theta = cart2pol(p2norm[0, :], p2norm[1, :])

    # Compute XY normalised, assuming p1 is the origin
    x_norm, y_norm = pol2cart(r, theta)
    xy_norm = np.array([x_norm, y_norm])

    # Compute transformed XY
    xy = p1 + xy_norm

    return xy


def normalise_re(r_i, e_i, d_i, s_i, spaths, G):
    edge_len = G['Edges']['Weight'][int(e_i)]
    path = spaths[(s_i, d_i)]
    idx = np.where(path == e_i)[0]

    if len(idx) > 0:
        # If idx is empty, it means that the edge does not exist on the given
        # path to a destination. Therefore this is an invalid particle, for
        # which nothing can be done. It's likelihood will be set to zero
        # during the weight update.
        idx = idx[0]
        while r_i > edge_len or r_i < 0:
            if r_i > edge_len:
                if len(path) > idx+1:
                    # If particle has NOT reached the end of the path
                    r_i = r_i - edge_len
                    idx = idx + 1
                    e_i = path[idx]
                    edge_len = G['Edges']['Weight'][int(e_i)]
                    if len(path) == idx + 1:
                        # If particle has reached the end of the path
                        if r_i > edge_len:
                            # Cap r_i to edge_length
                            r_i = edge_len
                        break
                else:
                    # If particle has reached the end of the path
                    if r_i > edge_len:
                        # Cap r_i to edge_length
                        r_i = edge_len
                    break
            elif r_i < 0:
                if idx > 0:
                    # If particle is within the path limits
                    idx = idx - 1
                    e_i = path[idx]
                    edge_len = G['Edges']['Weight'][int(e_i)]
                    r_i = edge_len + r_i
                else:
                    # Else if the particle position is beyond the path
                    # limits, the set its range to 0.
                    r_i = 0
                    break

    return r_i, e_i, d_i, s_i


def line_intersects_circle2(line, circle):
    p1 = line[0]
    p2 = line[1]

    c = circle[0]
    r = circle[1]

    p12 = p2-p1
    n = p12/np.linalg.norm(p12)

    p1c = c - p1
    v = np.abs(n[0] * p1c[1] - n[1] * p1c[0])
    return v <= r


def line_intersects_circle(line, circle):

    c = circle[0]
    r = circle[1]

    cc = Point(c).buffer(r)
    l = LineString(line)
    i = cc.boundary.intersection(l)

    return not i.is_empty

def line_intersects_circle3(line, circle):

    l = sympy.Line(sympy.Point(line[0]), sympy.Point(line[1]))
    c = sympy.Circle(sympy.Point(circle[0]), circle[1])

    i = c.intersection(l)

    return len(i) > 0

def circle_contains_line(line, circle):


    p1 = line[0]
    p2 = line[1]

    c = circle[0]
    r = circle[1]

    p1norm = p1 - c
    p2norm = p2 - c

    d1 = np.sqrt(sum(p1norm**2))
    d2 = np.sqrt(sum(p2norm**2))

    return (d1 < r) and (d2 < r)


def line_circle_test(line, circle):

    a = line_intersects_circle(line, circle)
    b = circle_contains_line(line, circle)

    return (a or b)


def calculate_r(line, point):
    p1 = np.array(line[0])
    p2 = np.array(line[1])
    p3 = np.array(point)

    d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

    d13 = np.linalg.norm(p1, p3)

    r = np.sqrt(d13**2-d**2)
    return r