import networkx as nx
import numpy as np
from datetime import datetime,timedelta

from stonesoup.custom.graph import shortest_path
from stonesoup.functions import pol2cart, cart2pol
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.groundtruth import GroundTruthState


def simulate(G: nx.DiGraph, source: int, destination: int, speed: float, track_id: int = None,
             timestamp_init: datetime = datetime.now(), interval: timedelta = timedelta(seconds=1)):
    """ Simulate a moving target along the network

    Parameters
    ----------
    G: :class:`nx.DiGraph`
        The road network
    source: :class:`int`
        The source node index
    destination: :class:`int`
        The destination node index
    speed: :class:`float`
        The speed of the target (assumed to be constant)
    timestamp_init: :class:`datetime.datetime`
        The initial timestamp
    interval: :class:`datetime.timedelta`
        The interval between ground-truth reports

    Returns
    -------
    :class:`~.GroundTruthPath`
        The ground truth path
    :class:`list`
        The list of node idxs in the route from source to destination
    :class:`list`
        The list of edge idxs in the route from source to destination

    """
    # Compute shortest path to destination
    gnd_route_n_tmp, gnd_route_e_tmp = shortest_path(G, source, destination)
    gnd_route_n = gnd_route_n_tmp[(source, destination)]
    gnd_route_e = gnd_route_e_tmp[(source, destination)]
    path_len = len(gnd_route_n)

    # Get the node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Initialize the Ground-Truth path
    sv = StateVector(np.array(pos[gnd_route_n[0]]))
    timestamp = timestamp_init
    state = GroundTruthState(sv, timestamp=timestamp)
    gnd_path = GroundTruthPath([state], id=track_id)

    r = 0  # Stores the distance travelled (range) along a given edge
    overflow = False  # Indicates when the range has overflown to a new edge

    # Iterate over the nodes in the route
    for k in range(1, path_len):
        # Index and position of last visited node
        node_km1 = gnd_route_n[k - 1]
        pos_km1 = np.array(pos[node_km1])
        # Index and position of next node along the edge
        node_k = gnd_route_n[k]
        pos_k = np.array(pos[node_k])
        # Compute distance (max range) and angle between the two nodes
        dpos = pos_k - pos_km1
        r_max, a = cart2pol(dpos[0], dpos[1])

        # Iterate until the next node has been reached
        reached = False
        while not reached:
            # Only add to the range if not overflown
            if not overflow:
                r += speed*interval.total_seconds()

            # If r falls within the max range of the edge
            # then we need to report the new gnd position
            # and continue until we have reached the next node
            if r <= r_max:
                overflow = False    # Reset the overflow flag
                # Compute and store the new gnd position
                x, y = pol2cart(r,a)
                x_i = pos_km1 + np.array([x, y])
                sv = StateVector(np.array(x_i))
                timestamp += interval
                state = GroundTruthState(sv, timestamp=timestamp)
                gnd_path.append(state)
                # If r == r_max it means we have reached the next node
                if r == r_max:
                    reached = True  # Signal that next node is reached
                    r = 0           # Reset r
            # Else if r is greater than the edge length, then
            # skip to the next edge, without reporting the gnd
            # position, unless we have reached the destination
            elif r > r_max:
                r -= r_max          # Update r to reflect the cross-over
                overflow = True     # Set the overflow flag
                reached = True      # Signal that we have reached the next node
                # If k == path_len-1 it means we have reached the destination
                # meaning that we should report the new position
                if k == path_len-1:
                    x, y = pol2cart(r_max, a)
                    x_i = pos_km1 + np.array([x, y])
                    sv = StateVector(np.array(x_i))
                    timestamp += interval
                    state = GroundTruthState(sv, timestamp=timestamp)
                    gnd_path.append(state)

    return gnd_path, gnd_route_n, gnd_route_e


def simulate_gnd(G, num_tracks, num_destinations, speed):
    t_sources = []
    t_destinations = []
    t_colors = []
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    for i in range(num_tracks):
        t_len = 0
        print(i)
        while t_len < 20 or t_len > 100:
            t_source = int(np.random.uniform(0, num_nodes))
            t_dest = int(np.random.uniform(0, num_nodes))
            path_n_tmp, path_e_tmp = shortest_path(G, t_source, t_dest)
            path_n = path_n_tmp[(t_source, t_dest)]
            path_e = path_e_tmp[(t_source, t_dest)]
            t_len = len(path_e)
        t_sources.append(t_source)
        t_destinations.append(t_dest)
        t_colors.append(np.random.rand(3, ))

    # Simulate ground-truth
    gnd_paths = set()
    gnd_routes_n = dict()
    gnd_routes_e = dict()
    timestamp_init = datetime.now()
    for i in range(num_tracks):
        dt = timedelta(seconds=np.random.randint(0, 50))
        gnd_path, gnd_route_n, gnd_route_e = simulate(G, t_sources[i], t_destinations[i], speed,
                                                      timestamp_init=timestamp_init+dt, track_id=i)
        gnd_paths.add(gnd_path)
        gnd_routes_e[gnd_path] = gnd_route_e
        gnd_routes_n[gnd_path] = gnd_route_n

    feed = t_destinations
    feed_tmp = set([i for i in range(num_nodes)]) - set(feed)
    destinations = feed + list(
        np.random.choice(list(feed_tmp), (num_destinations - len(feed),), False))

    return gnd_paths, gnd_routes_n, gnd_routes_e, t_sources, t_destinations, t_colors, destinations

def simulate_detections(gnd_paths, measurement_model, P_D, lambda_FA, VBOUNDS):
    # Simulate detections
    scans = []
    scans_dict = dict()
    for gnd_path in gnd_paths:
        for i, gnd_state in enumerate(gnd_path):
            gnd_sv = gnd_state.state_vector
            det_sv = gnd_sv + measurement_model.rvs()
            timestamp = gnd_state.timestamp
            metadata = {"gnd_id": gnd_path.id}
            detection = Detection(state_vector=det_sv, timestamp=timestamp, metadata=metadata)
            if i == 0 or np.random.rand() <= P_D:
                if timestamp in scans_dict:
                    scans_dict[timestamp] |= set([detection])
                else:
                    scans_dict[timestamp] = set([detection])

    for timestamp in sorted(scans_dict.keys()):
        detections = scans_dict[timestamp]
        # Add clutter
        num_clutter = np.random.poisson(lambda_FA)
        x = np.random.uniform(VBOUNDS[0][0], VBOUNDS[0][1], num_clutter)
        y = np.random.uniform(VBOUNDS[1][0], VBOUNDS[1][1], num_clutter)
        for x_pos, y_pos in zip(x, y):
            detection = Detection(state_vector=StateVector([x_pos, y_pos]), timestamp=timestamp)
            detections.add(detection)

        scans.append((timestamp, detections))

    return scans

def compute_short_paths(G, sources, destinations):

    num_sources = len(sources)
    num_destinations = len(destinations)

    # Pre-compute short_paths
    short_paths_n = dict()
    short_paths_e = dict()
    for source in sources:
        for destination in destinations:
            print("Computing path: ({},{})".format(source, destination))
            short_paths_n[(source, destination)], short_paths_e[(source, destination)] = \
                shortest_path(G, source, destination)

    return short_paths_e, short_paths_n

def compute_all_short_paths(G, sources):

    # Pre-compute short_paths
    short_paths_n = dict()
    short_paths_e = dict()

    edges = [e for e in G.edges]
    edges_index = {edge: edges.index(edge) for edge in edges}

    for source in sources:
        # print("Computing paths for node {}".format(source))
        paths = nx.shortest_path(G, source, weight='weight')
        for destination, node_path in sorted(paths.items()):
            print("Computing path: ({},{})".format(source, destination))
            path_edges = zip(node_path, node_path[1:])
            edge_path = [edges_index[edge] for edge in path_edges]
            short_paths_n[(source, destination)], short_paths_e[(source, destination)] = \
                (node_path, edge_path)

    return short_paths_e, short_paths_n