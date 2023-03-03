import typing as tp

import algo
import network_graph
from network_graph import ResidualNetwork, SimpleNetwork, NodeId, EdgeId, EdgeType


class MaximumFlowFinder:
    @staticmethod
    def find(network: network_graph.SimpleNetwork) -> None:
        r_network = network_graph.ResidualNetwork()
        r_network.setup(network)

        while True:
            bfs = algo.BFS.find_minimal_way(r_network)

            if not bfs:
                break
            bfs_edges = MaximumFlowFinder.__get_min_way_edges(network, bfs)

            inc = MaximumFlowFinder.__find_min_flow_in_increasing_way(r_network, bfs_edges)
            if inc == 0:
                break

            for edge_id in bfs_edges:
                old_flow = network.get_edge_flow(edge_id)
                network.set_edge_flow(edge_id, old_flow + inc)

    @staticmethod
    def __get_min_way_edges(network: SimpleNetwork, bfs: tp.List[NodeId]) -> tp.List[EdgeId]:
        edges = []
        for i in range(len(bfs) - 1):
            a = (bfs[i], bfs[i + 1], EdgeType.NORMAL)
            if network.edge_exist_q(a):
                edges.append(a)
        return edges

    @staticmethod
    def __find_min_flow_in_increasing_way(network: ResidualNetwork, edges: tp.List[EdgeId]):
        return min(network.get_edge_flow(i) for i in edges)
