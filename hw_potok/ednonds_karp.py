import typing as tp

import algo
import network_graph
from network_graph import ResidualNetwork, EdgeId, EdgeType


class MaximumFlowFinder:
    @staticmethod
    def find(network: network_graph.SimpleNetwork) -> None:
        r_network = network_graph.ResidualNetwork()
        r_network.setup(network)

        while True:
            bfs_edges, bfs_types = algo.BFS.find_minimal_way(r_network)

            if not bfs_edges:
                break

            inc = MaximumFlowFinder.__find_min_flow_in_increasing_way(r_network, bfs_edges, bfs_types)

            for edge_id, edge_type in zip(bfs_edges, bfs_types) :
                if edge_type == EdgeType.INVERTED:
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    network.set_edge_flow(r_edge_id, old_flow - inc)
                else:
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + inc)

            print(network)

    @staticmethod
    def __find_min_flow_in_increasing_way(network: ResidualNetwork, edges: tp.List[EdgeId], edge_types: tp.List[EdgeType]):
        return min(network.get_edge_flow(edges[i], edge_types[i]) for i in range(len(edges)))
