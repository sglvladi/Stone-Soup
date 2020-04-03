import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from stonesoup.custom.graph import load_graph_dict, dict_to_graph, shortest_path, get_xy_from_range_edge
from stonesoup.custom.plotting import plot_network
from stonesoup.custom.simulation import simulate, simulate2

path = r'C:\Users\sglvladi\OneDrive\Workspace\PostDoc\CADMURI\MATLAB\simple\data\minn_2.mat'
S = load_graph_dict(path)
G = dict_to_graph(S)

xy = get_xy_from_range_edge(0, 5, S)
node_path, edge_path = shortest_path(G, 519, 115)

# pos = nx.get_node_attributes(G, 'pos')
# gnd_path = np.empty((0,2))
# for node in node_path:
#     x_i = np.array(pos[node])
#     gnd_path = np.append(gnd_path, [x_i], axis=0)

gnd_path, gnd_route_n, gnd_route_e = simulate2(G, 519, 115, 0.001)

# Plot the network
data = np.array([state.state_vector for state in gnd_path])
fig, ax = plt.subplots()
plot_network(G, ax)
plt.plot(data[:,0], data[:,1], 'r.-')
plt.show()

a=2

[1289, 1309, 1316, 1341, 1347, 1344, 1342, 1335, 1338, 1349, 1333, 1365, 1395, 1404, 1436, 1445, 1447, 1492, 1482, 1478, 1475, 1480, 1457, 1449, 1441, 1438, 1431, 1372, 1371, 1374, 1362, 1356, 1326, 1324, 1328, 1310, 1227, 1248, 1263, 1264, 1254, 1236, 1218, 1216, 1213, 1203, 1200, 1197, 1195, 1191, 1186, 1181, 1175, 1172, 1163, 1146, 1144, 1137, 1133, 1124, 1127, 1129, 1118, 1115, 1113, 1096, 1088, 1079, 1077, 1075, 1072, 1059, 1029, 996, 896, 875, 840, 672, 603, 495, 449, 446, 351, 242]