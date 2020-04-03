# example.py - An example script that reads in and plots the WAMI road network
#
# IMPORTANT:
# ---------
# Before executing the script, make sure to install all dependencies by
# running the following command:
#
#   pip install pandas xlrd networkx matplotlib

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Open and read the network .xls file
filename = "./WAMI_Road_Network.xls"
dfs = pd.read_excel(filename, sheet_name=None)

# Create empty graph object
G = nx.Graph()

# Add nodes to graph
for index, row in dfs['Nodes'].iterrows() :
    G.add_node(row['ID'], pos=(row['Longitude'], row['Latitude']))

# Add edges to graph
for index, row in dfs['Edges'].iterrows() :
    edge = (row['EndNodes_1'], row['EndNodes_2'])
    G.add_edge(*edge)

# Get node positions
pos = nx.get_node_attributes(G,'pos')

# Plot the network
fig, ax = plt.subplots()
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
nx.draw(G, pos, ax=ax, node_size=0.1, width=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('OSTEWG WAMI Challenge Road Network')
plt.draw()
plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()
# plt.savefig('WAMI_Road_Network.png', dpi=600)