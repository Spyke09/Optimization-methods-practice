import matplotlib.pyplot as plt
import networkx
import networkx as nx

import network_graph


def draw_network(network: network_graph.SimpleNetwork, order) -> None:
    c = network.get_capacities()
    f = network.get_flow()

    g = dict()
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] > 0:
                g[(i, j)] = f"{f[i][j]}/{c[i][j]}"

    edges = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] > 0:
                edges.append([i, j])

    graph = networkx.DiGraph()
    graph.add_nodes_from([i for i in range(len(c))])
    graph.add_edges_from(edges)

    for node in range(network.size()):
        graph.nodes[node]["layer"] = order[node]

    pos = networkx.multipartite_layout(graph, subset_key="layer")
    plt.figure()
    networkx.draw(
        graph, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color="pink",
        labels={node: node for node in graph.nodes()}
    )

    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=g, font_color="red"
    )
    plt.show()
