import networkx
import networkx as nx

import inetwork
import matplotlib.pyplot as plt


def draw_network(network: inetwork.INetwork):
    c = network.get_capacities()
    f = network.get_flow()

    g = dict()
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] > 0:
                g[(i, j)] = f"{f[i][j]}/{c[i][j]}"

    g_f = dict()
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] > 0:
                g_f[(i, j)] = f[i][j]

    edges = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] > 0:
                edges.append([i, j])

    GG = networkx.DiGraph()
    GG.add_nodes_from([i for i in range(len(c))])
    GG.add_edges_from(edges)

    pos = networkx.circular_layout(GG)
    plt.figure()
    networkx.draw(
        GG, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color="pink",
        labels={node: node for node in GG.nodes()}
    )

    nx.draw_networkx_edge_labels(
        GG, pos, edge_labels=g, font_color="red"
    )
    plt.show()