from abc import ABC, abstractmethod

import algo
import network_graph
from network_graph import ResidualGraph, EdgeType


class IMaximumFlowFinder(ABC):

    @abstractmethod
    def find(self, network: network_graph.SimpleNetwork):
        raise NotImplementedError


class EdmondsKarp(IMaximumFlowFinder):

    def find(self, network: network_graph.SimpleNetwork) -> None:
        r_network = network_graph.ResidualGraph(network)

        while True:
            bfs = algo.BFS.bfs_for_edmonds_karp(r_network)

            if not bfs:
                break

            inc = self.__find_min_flow_in_increasing_way(r_network, bfs)

            for edge_id, edge_type in bfs:
                if edge_type == EdgeType.INVERTED:
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    network.set_edge_flow(r_edge_id, old_flow - inc)
                else:
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + inc)

    @staticmethod
    def __find_min_flow_in_increasing_way(network: ResidualGraph, edges):
        return min(network.get_edge_capacity(edges[i][0], edges[i][1]) for i in range(len(edges)))


class Dinica(IMaximumFlowFinder):
    def find(self, network: network_graph.SimpleNetwork) -> None:
        while True:
            l_network = network_graph.LayeredGraph(network_graph.ResidualGraph(network))
            l_network.init_block_way()
            new_f = l_network.get_flow()
            if all([i == 0 for i in new_f.values()]):
                break
            for (edge_id, edge_type), flow_ in new_f.items():
                if edge_type == EdgeType.INVERTED:
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    network.set_edge_flow(r_edge_id, old_flow - flow_)
                else:
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + flow_)

            assert network.check_conservation_law()
