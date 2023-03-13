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

    def __find_min_flow_in_increasing_way(self, network: ResidualGraph, edges):
        return min(network.get_edge_capacity(edges[i][0], edges[i][1]) for i in range(len(edges)))


class Dinica(IMaximumFlowFinder):
    def find(self, network: network_graph.SimpleNetwork) -> None:
        pass

