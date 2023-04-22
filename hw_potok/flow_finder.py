import logging
from abc import ABC, abstractmethod

import algo
import network_graph
from network_graph import ResidualGraph, EdgeType


class IMaximumFlowFinder(ABC):
    @abstractmethod
    def find(self, network: network_graph.SimpleNetwork):
        raise NotImplementedError


class EdmondsKarp(IMaximumFlowFinder):
    def __init__(self):
        self.logger = logging.getLogger("EdmondsKarp")

    def find(self, network: network_graph.SimpleNetwork) -> None:
        self.logger.info("Start finding max-flow")
        r_network = network_graph.ResidualGraph(network)

        while True:
            bfs = algo.BFS.bfs_for_edmonds_karp(r_network)
            self.logger.debug(f"Next way: {[(i[0], i[1].name) for i in bfs]}")

            if not bfs:
                break

            inc = self.__find_min_flow_in_increasing_way(r_network, bfs)
            self.logger.debug(f"The value of increase: {inc}")

            for edge_id, edge_type in bfs:
                if edge_type == EdgeType.INVERTED:
                    self.logger.debug(f"Decrease along the edge {edge_id} by {inc}.")
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    network.set_edge_flow(r_edge_id, old_flow - inc)
                else:
                    self.logger.debug(f"Increase along the edge {edge_id} by {inc}.")
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + inc)

        self.logger.info("End finding max-flow")
        self.logger.info(f"Answer is: {network.get_network_flow()}")

    @staticmethod
    def __find_min_flow_in_increasing_way(network: ResidualGraph, edges):
        return min(network.get_edge_capacity(edges[i][0], edges[i][1]) for i in range(len(edges)))


class Dinica(IMaximumFlowFinder):
    def __init__(self):
        self.logger = logging.getLogger("Dinica")

    def find(self, network: network_graph.SimpleNetwork) -> None:
        self.logger.info("Start finding max-flow")
        while True:
            l_network = network_graph.LayeredGraph(network_graph.ResidualGraph(network))
            l_network.init_block_way()
            new_f = l_network.get_flow()
            self.logger.debug(f"Next block way: {[((i[0], i[1].name), j) for i, j in new_f.items() if j > 0]}")
            if all([i == 0 for i in new_f.values()]):
                break
            for (edge_id, edge_type), flow_ in new_f.items():
                if flow_ == 0:
                    continue
                if edge_type == EdgeType.INVERTED:
                    self.logger.debug(f"Decrease along the edge {edge_id} by {flow_}.")
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    network.set_edge_flow(r_edge_id, old_flow - flow_)
                else:
                    self.logger.debug(f"Increase along the edge {edge_id} by {flow_}.")
                    old_flow = network.get_edge_flow(edge_id)
                    network.set_edge_flow(edge_id, old_flow + flow_)
            assert network.check_conservation_law()

        self.logger.info("End finding max-flow")
        self.logger.info(f"Answer is: {network.get_network_flow()}")


class GoldbergT(IMaximumFlowFinder):
    def __init__(self):
        self.logger = logging.getLogger("GoldbergT")

    def find(self, network: network_graph.SimpleNetwork):
        self.logger.info("Start finding max-flow")

        number_of_steps = 0
        r_network = network_graph.ResidualGraph(network)
        source = network.get_source()
        sink = network.get_sink()
        psi = {i: 0 for i in range(source, sink + 1)}
        psi[source] = network.size()
        self.logger.debug(f"Rate of node source is {network.size()}. For the rest, it's zero.")

        for i in network.get_node_fan_out(source):
            network.set_edge_flow((source, i), network.get_edge_capacity((source, i)))
            self.logger.debug(f"Set edge {(source, i)} flow = {network.get_edge_flow((source, i))}")

        while True:
            v = self.get_active_node(network, psi)
            self.logger.debug(f"Next active node: {v}")
            if not v:
                break

            a_edges = self.get_acceptable_edges(v, r_network, psi)
            if not a_edges:
                old_rate = psi[v]
                number_of_steps += 1
                self.re_rate(v, r_network, psi)
                self.logger.debug(f"Re-rate node {v}. Old rate: {old_rate}. New rate: {psi[v]}")
                a_edges = self.get_acceptable_edges(v, r_network, psi)

            self.logger.debug(f"Next active edges: {[(i[0], i[1].name) for i in a_edges]}")
            for edge_id, edge_type in a_edges:
                r_e = r_network.get_edge_capacity(edge_id, edge_type)
                if r_e == 0:
                    break
                f = min(network.get_excess_flow(edge_id[0]), r_e)
                if edge_type == EdgeType.INVERTED:
                    r_edge_id = network_graph.reverse_edge(edge_id)
                    old_flow = network.get_edge_flow(r_edge_id)
                    gamma = min(f, old_flow)
                    if gamma != 0:
                        self.logger.debug(f"Decrease along the edge {edge_id} by {gamma}.")
                    network.set_edge_flow(r_edge_id, old_flow - gamma)
                else:
                    old_flow = network.get_edge_flow(edge_id)
                    if f != 0:
                        self.logger.debug(f"Increase along the edge {edge_id} by {f}.")
                    network.set_edge_flow(edge_id, old_flow + f)

        self.logger.info("End finding max-flow")
        self.logger.info(f"Number of steps: {number_of_steps}")
        self.logger.info(f"Answer is: {network.get_network_flow()}")
        assert network.check_conservation_law()

    @staticmethod
    def re_rate(v, r_network: network_graph.ResidualGraph, psi):
        psi[v] = min(psi[i] for i in r_network.get_node_fan_out(v)) + 1

    @staticmethod
    def get_acceptable_edges(v, r_network: network_graph.ResidualGraph, psi):
        res = []
        for j in range(r_network.size()):
            for t in (EdgeType.NORMAL, EdgeType.INVERTED):
                if r_network.edge_exist_q((v, j), t) and psi[v] == psi[j] + 1:
                    res.append(((v, j), t))

        return list(sorted(res, key=(lambda x: x[1].value)))

    @staticmethod
    def get_active_node(network: network_graph.SimpleNetwork, psi):
        max_psi = -1
        res = None
        for i in range(network.size()):
            if network.get_excess_flow(i) > 0 and i != network.get_sink():
                if max_psi <= psi[i]:
                    max_psi = psi[i]
                    res = i
        return res


class MinCostFlow(IMaximumFlowFinder):
    def __init__(self):
        self.logger = logging.getLogger("MinCostFlow")

    def find(self, network: network_graph.SimpleNetwork, flow_finder: IMaximumFlowFinder = GoldbergT()):
        self.logger.info("Start finding min-cost max-flow")
        flow_finder.find(network)
        r_network = ResidualGraph(network)
        while True:
            cycle = algo.NegativeCycleFinder.find(r_network)
            self.logger.debug(f"Next negative cycle: {[(i[0], i[1].name) for i in cycle]}")
            if not cycle:
                break

            gamma = None
            for edge_id, edge_type in cycle:
                gamma = min(gamma, r_network.get_edge_capacity(edge_id, edge_type))

            for edge_id, edge_type in cycle:
                if edge_type == EdgeType.NORMAL:
                    self.logger.debug(f"Increase along the edge {edge_id} by {gamma}.")
                    network.set_edge_flow(edge_id, network.get_edge_flow(edge_id) + gamma)

                if edge_type == EdgeType.INVERTED:
                    self.logger.debug(f"Decrease along the edge {edge_id} by {gamma}.")
                    r_e_id = network_graph.reverse_edge(edge_id)
                    network.set_edge_flow(r_e_id, network.get_edge_flow(r_e_id) - gamma)
