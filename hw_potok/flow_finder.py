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


class GoldbergT(IMaximumFlowFinder):
    def find(self, network: network_graph.SimpleNetwork):
        r_network = network_graph.ResidualGraph(network)
        source = network.get_source()
        sink = network.get_sink()
        psi = {i: 0 for i in range(source, sink + 1)}
        psi[source] = network.size()

        for i in network.get_node_fan_out(source):
            network.set_edge_flow((source, i), network.get_edge_capacity((source, i)))

        while True:
            a_nodes = self.get_active_node(network, psi)
            if not a_nodes:
                break

            v = a_nodes
            a_edges = self.get_acceptable_edges(v, r_network, psi)
            if not a_edges:
                self.re_rate(v, r_network, psi)

            for edge_id, edge_type in a_edges:
                f = min(network.get_excess_flow(edge_id[0]), r_network.get_edge_capacity(edge_id, edge_type))
                if edge_type == EdgeType.INVERTED:
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    gamma = min(f, old_flow)
                    network.set_edge_flow(r_edge_id, old_flow - gamma)
                else:
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + f)

        assert network.check_conservation_law()

    def re_rate(self, v, r_network: network_graph.ResidualGraph, psi):
         psi[v] = min(psi[i] for i in r_network.get_node_fan_out(v)) + 1

    def get_acceptable_edges(self, v, r_network: network_graph.ResidualGraph, psi):
        res = []
        for j in range(r_network.size()):
            for t in (EdgeType.NORMAL, EdgeType.INVERTED):
                if r_network.edge_exist_q((v, j), t) and psi[v] == psi[j] + 1:
                    res.append(((v, j), t))

        return res

    def get_active_node(self, network: network_graph.SimpleNetwork, psi):
        max_psi = -1
        res = None
        for i in range(network.size()):
            if network.get_excess_flow(i) > 0 and i != network.get_sink():
                if max_psi < psi[i]:
                    max_psi = psi[i]
                    res = i
        return res
