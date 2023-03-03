import typing as tp
from collections import deque

import inetwork
import network_graph


class BFS:
    @staticmethod
    def find_minimal_way(network: inetwork.INetwork) -> tp.List[network_graph.NodeId]:
        start = network.get_source()
        finish = network.get_sink()
        fifo = deque([start])
        dist = {start: [0, None]}

        while fifo:
            current_node_id = fifo.popleft()

            for node_id in network.get_node_fan_out(current_node_id):
                if node_id not in dist or dist[node_id][0] > dist[current_node_id][0] + 1:
                    dist[node_id] = [dist[current_node_id][0] + 1, current_node_id]
                    fifo.append(node_id)

        if finish not in dist:
            return []
        min_way = [finish]
        while True:
            if min_way[-1] == start:
                break
            min_way.append(dist[min_way[-1]][1])
        return list(reversed(min_way))
