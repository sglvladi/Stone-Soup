import networkx as nx
import matplotlib.pyplot as plt


def plot_network(G, ax, node_size=0.1, width=0.5, with_labels=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx(G, pos, arrows=False, ax=ax, node_size=node_size, width=width, with_labels=with_labels)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)