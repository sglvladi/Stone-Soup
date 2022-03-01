import pickle
import geopandas
import itertools

import numpy as np
import networkx as nx
from scipy.io import loadmat

from stonesoup.functions import cart2pol, pol2cart
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

from shapely.geometry import LineString
from shapely.geometry import Point
# import sympy


class CustomDiGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super(CustomDiGraph, self).__init__(*args, *kwargs)
        self.dict = graph_to_dict(self)
        self.gdf = dict2gdf(self.dict)
        self._rtree = None
        self.short_paths_n = dict()
        self.short_paths_e = dict()

    def edges_by_idx(self, idx):
        if not isinstance(idx, list):
            idx = list([idx])
        edges = [e for i, e in enumerate(self.edges) if i in idx]
        return edges

    def weights_by_idx(self, idx):
        edges = self.edges_by_idx(idx)
        weights = [self.adj[e[0]][e[1]]['weight'] for e in edges]
        return weights

    def as_dict(self):
        return self.dict

    @property
    def rtree(self):
        if self._rtree is None:
            self._rtree = self.gdf.sindex
        return self._rtree

    @classmethod
    def from_dict(cls, S):
        G = dict_to_graph(S)
        G.dict = S
        G.gdf = dict2gdf(G.dict)
        # G._rtree = G.gdf.sindex
        return G

    @classmethod
    def fix(cls, G):
        if not hasattr(G, 'short_paths_n'):
            G.short_paths_n = dict()
        if not hasattr(G, 'short_paths_e'):
            G.short_paths_e = dict()
        return G

    def shortest_path(self, sources=None, targets=None, path_type='both', cache_filepath=None):
        try:
            short_paths_n, short_paths_e = self.short_paths_n[(sources, targets)], \
                                           self.short_paths_e[(sources, targets)]
            if path_type == 'node':
                return short_paths_n
            elif path_type == 'edge':
                return short_paths_e
            else:
                return short_paths_n, short_paths_e
        except KeyError:
            pass
        short_paths_n, short_paths_e = shortest_path(self, sources, targets, cache_filepath)
        self.short_paths_n.update(short_paths_n)
        self.short_paths_e.update(short_paths_e)
        if path_type == 'node':
            return short_paths_n
        elif path_type == 'edge':
            return short_paths_e
        else:
            return short_paths_n, short_paths_e


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


def graph_to_dict(G):

    weights = nx.get_edge_attributes(G, 'weight')
    S = dict()
    S['Edges'] = dict()
    S['Edges']['EndNodes'] = []
    S['Edges']['Weight'] = []
    for edge in G.edges:
        S['Edges']['EndNodes'].append([edge[0], edge[1]])
        S['Edges']['Weight'].append(weights[edge])

    pos = nx.get_node_attributes(G, 'pos')
    S['Nodes'] = dict()
    S['Nodes']['Longitude'] = []
    S['Nodes']['Latitude'] = []
    for node in G.nodes:
        S['Nodes']['Longitude'].append(pos[node][0])
        S['Nodes']['Latitude'].append(pos[node][1])

    S['Edges']['Weight'] = np.array(S['Edges']['Weight'])
    S['Edges']['EndNodes'] = np.array(S['Edges']['EndNodes'])
    S['Nodes']['Longitude'] = np.array(S['Nodes']['Longitude'])
    S['Nodes']['Latitude'] = np.array(S['Nodes']['Latitude'])
    return S


def nxdigraph_to_customdigraph(G):
    adj = nx.adjacency_matrix(G).todense()
    G2 = nx.from_numpy_matrix(adj, create_using=CustomDiGraph)

    pos = nx.get_node_attributes(G, 'pos')
    weight = nx.get_edge_attributes(G, 'edge')
    for key, value in pos.items():
        pos[key] = {'pos': value}

    nx.set_node_attributes(G2, pos)
    nx.set_edge_attributes(G2, name='weight', values=weight)

    return G2


def dict2gdf(S):
    """Convert dictionary into a GeoDataFrame object"""

    d = {'col1': [], 'geometry': []}  # Initialise a new dictionary
    last_index = S['Edges']['EndNodes'].shape[0]  # Count the road segments

    for i in list(range(0, last_index)):
        d['col1'].append('Edge ' + str(i))
        node1, node2 = S['Edges']['EndNodes'][i]
        lat1 = S['Nodes']['Latitude'][node1]
        lon1 = S['Nodes']['Longitude'][node1]
        lat2 = S['Nodes']['Latitude'][node2]
        lon2 = S['Nodes']['Longitude'][node2]
        segment = LineString([(lon1, lat1), (lon2, lat2)])
        d['geometry'].append(segment)

    gdf = geopandas.GeoDataFrame(d)  # GeoDataFrame from dict
    return gdf


def shortest_path(G, sources=None, targets=None, cache_filepath=None):
    edges_index = {edge: i for i, edge in enumerate(G.edges)}

    short_paths_n = dict()
    short_paths_e = dict()

    multi_source = isinstance(sources, list)
    multi_target = isinstance(targets, list)

    if cache_filepath is None:
        if sources is None or targets is None:
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
                    edge_path = [edges_index[edge] for edge in path_edges]
                except (nx.NetworkXNoPath):
                    node_path = []
                    edge_path = []
                short_paths_n[(s, t)], short_paths_e[(s, t)] = (node_path, edge_path)
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

    S = G.as_dict()

    r = np.atleast_1d(r)
    e = np.atleast_1d(e).astype(int)

    endnodes = S['Edges']['EndNodes'][e, :]

    # Get endnode coordinates
    p1 = np.array([S['Nodes']['Longitude'][endnodes[:,0]], S['Nodes']['Latitude'][endnodes[:,0]]])
    p2 = np.array([S['Nodes']['Longitude'][endnodes[:,1]], S['Nodes']['Latitude'][endnodes[:,1]]])

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


def normalise_re(r_i, e_i, d_i, s_i, G):

    S = G.as_dict()
    spaths = G.short_paths_e
    edge_len = S['Edges']['Weight'][int(e_i)]
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
                    edge_len = S['Edges']['Weight'][int(e_i)]
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
                    edge_len = S['Edges']['Weight'][int(e_i)]
                    r_i = edge_len + r_i
                else:
                    # Else if the particle position is beyond the path
                    # limits, the set its range to 0.
                    r_i = 0
                    break
    else:
        if r_i > edge_len:
            r_i = edge_len
        elif r_i < 0:
            r_i = 0

    return r_i, e_i, d_i, s_i


def line_intersects_circle(line, circle):

    c = circle[0]
    r = circle[1]

    cc = Point(c).buffer(r)
    l = LineString(line)
    i = cc.boundary.intersection(l)

    return not i.is_empty


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

    return line_intersects_circle(line, circle) or circle_contains_line(line, circle)


def calculate_r(line, point):
    p1 = np.array(line[0])
    p2 = np.array(line[1])
    p3 = np.array(point)

    # Compute orthogonal projection of p3 on line defined by p1 and p2
    a = np.array([[-p3[0]*(p2[0]-p1[0]) - p3[1]*(p2[1]-p1[1])],
                  [-p1[1]*(p2[0]-p1[0]) + p1[0]*(p2[1]-p1[1])]])
    b = np.array([[p2[0] - p1[0], p2[1]-p1[1]],
                  [p1[1] - p2[1], p2[0] - p1[0]]])
    projected_point = -np.linalg.lstsq(b, a, rcond=-1)[0].ravel()

    # Perform checking to see where the projected point lies in relation to p1 and p2
    if point_is_on_line_segment(line, projected_point):
        # projected point is between p1 and p2: r is distance between p1 and projected point
        return np.linalg.norm(p1-projected_point)
    elif point_is_on_line_segment((projected_point, p2), p1):
        # projected point is "left" of p1: r is 0
        return 0
    elif point_is_on_line_segment((p1, projected_point), p2):
        # projected point is "right" of p2: r is edge length
        return np.linalg.norm(p2-p1)
    else:
        # raise an error if non of the above hold: There is a bug somewhere!
        raise AssertionError


def point_is_on_line_segment(line, point):
    pt1 = line[0]
    pt2 = line[1]
    pt3 = point

    x1, x2, x3 = pt1[0], pt2[0], pt3[0]

    y1, y2, y3 = pt1[1], pt2[1], pt3[1]

    if np.isclose(x2, x1):
        # If line is parallel to x-axis, the slope is undefined
        on_and_between = np.isclose(x3, x2) and (min(y1, y2) <= y3 <= max(y1, y2))
    else:
        slope = (y2 - y1) / (x2 - x1)
        pt3_on = np.isclose((y3 - y1), slope * (x3 - x1))

        pt3_between = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))

        on_and_between = pt3_on and pt3_between
    return on_and_between
