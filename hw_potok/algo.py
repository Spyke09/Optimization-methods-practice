import typing as tp
from collections import deque

import network_graph
from network_graph import EdgeType


class BFS:
    @staticmethod
    def find_minimal_way(
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
