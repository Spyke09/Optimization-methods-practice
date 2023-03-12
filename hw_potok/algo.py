import typing as tp
from collections import deque

import network_graph
from network_graph import EdgeType


class BFS:
    @staticmethod
    def bfs_for_edmonds_karp(
            network: network_graph.ResidualNetwork
    ) -> tp.Tuple[tp.List[network_graph.EdgeId], tp.List[EdgeType]]:
        start = network.get_source()
        finish = network.get_sink()
        fifo = deque([start])
        dist = {start: None}

        while fifo:
            current_node_id = fifo.popleft()

            out = network.get_node_fan_out(current_node_id)
            for node_id in out:
                if node_id not in dist:
                    dist[node_id] = current_node_id
                    fifo.append(node_id)

        if finish not in dist:
            return [], []
        min_way = [finish]
        while True:
            if min_way[-1] == start:
                break
            min_way.append(dist[min_way[-1]])

        min_way = list(reversed(min_way))
        edges = []
        for i in range(len(min_way) - 1):
            a = (min_way[i], min_way[i + 1])
            edges.append(a)

        types = [EdgeType.INVERTED if network.edge_exist_q(i, EdgeType.INVERTED) else EdgeType.NORMAL for i in edges]
        return edges, types

    @staticmethod
    def bfs_for_dinica(
            network: network_graph.SimpleNetwork
    ) -> tp.List[network_graph.EdgeId]:
        start = network.get_source()
        finish = network.get_sink()
        fifo = deque([start])
        dist = {start: 0}

        while fifo:
            current_node_id = fifo.popleft()

            out = network.get_node_fan_out(current_node_id)
            for node_id in out:
                edge = (current_node_id, node_id)
                temp = network.get_edge_capacity(edge) - network.get_edge_flow(edge)
                if node_id not in dist and temp > 0:
                    dist[node_id] = dist[current_node_id] + 1
                    fifo.append(node_id)

        if finish not in dist:
            return [], []
        min_way = [finish]
        while True:
            if min_way[-1] == start:
                break
            min_way.append(dist[min_way[-1]])

        min_way = list(reversed(min_way))
        edges = []
        for i in range(len(min_way) - 1):
            a = (min_way[i], min_way[i + 1])
            edges.append(a)

        return edges
