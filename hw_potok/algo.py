import typing as tp
from collections import deque

import inetwork
from inetwork import EdgeType, NodeId, EdgeId, Edge
from numeric_tools import Infinity, MinusInfinity


class BFS:
    @staticmethod
    def perform_dfs(
            network: inetwork.INetwork
    ) -> tp.Dict[NodeId, tp.Tuple[NodeId, int]]:
        start = network.get_source()
        fifo = deque([start])
        dist: tp.Dict[NodeId, tp.Tuple[NodeId, int]] = {start: (None, 0)}

        while fifo:
            current_node_id = fifo.popleft()

            out = network.get_node_fan_out(current_node_id)
            for node_id in out:
                if node_id not in dist:
                    dist[node_id] = (current_node_id, dist[current_node_id][1] + 1)
                    fifo.append(node_id)

        return dist

    @staticmethod
    def bfs_for_edmonds_karp(network: inetwork.INetwork) -> tp.List[tp.Tuple[EdgeId, EdgeType]]:
        start = network.get_source()
        finish = network.get_sink()
        dist = BFS.perform_dfs(network)

        if finish not in dist:
            return []

        min_way = [finish]
        while True:
            if min_way[-1] == start:
                break
            min_way.append(dist[min_way[-1]][0])

        min_way = list(reversed(min_way))
        edges = []
        for i in range(len(min_way) - 1):
            a = (min_way[i], min_way[i + 1])
            if network.edge_exist_q(a, EdgeType.INVERTED):
                edges.append((a, EdgeType.INVERTED))
            else:
                edges.append((a, EdgeType.NORMAL))

        return edges

    @staticmethod
    def bfs_for_dinica(network: inetwork.INetwork) -> tp.List[tp.Tuple[EdgeId, EdgeType]]:
        finish = network.get_sink()
        dist = BFS.perform_dfs(network)

        res = []
        sink_count = 0
        for i in range(finish + 1):
            for j in range(finish + 1):
                if i in dist and j in dist:
                    if network.edge_exist_q((i, j), EdgeType.NORMAL) and dist[j][1] == dist[i][1] + 1:
                        res.append(((i, j), EdgeType.NORMAL))
                        if j == network.get_sink():
                            sink_count += 1

                    if network.edge_exist_q((i, j), EdgeType.INVERTED) and dist[j][1] == dist[i][1] + 1:
                        res.append(((i, j), EdgeType.INVERTED))
                        if j == network.get_sink():
                            sink_count += 1

        return res


class Topsort:
    @staticmethod
    def dfs_topsort(out: tp.Dict[NodeId, tp.List[Edge]], source: NodeId) -> tp.List[NodeId]:
        used = set()
        ordering = []

        def dfs(v):
            used.add(v)
            for (i, j), k in (out[v] if v in out else []):
                if j not in used:
                    dfs(j)
            ordering.append(v)

        dfs(source)
        return list(reversed(ordering))


GCostValue = tp.Union[inetwork.CostValue, Infinity]


class NegativeCycleFinder:
    @staticmethod
    def __get_f_and_kpath(network: inetwork.INetwork) \
            -> tp.Tuple[
                    tp.List[tp.List[GCostValue]],
                    tp.List[tp.List[tp.Optional[Edge]]]
                ]:
        n = network.size()
        f: tp.List[tp.List[GCostValue]] = [[Infinity() for _ in range(n)] for _ in range(n + 1)]
        for i in range(n):
            f[0][i] = 0
        path: tp.List[tp.List[tp.Optional[Edge]]] = [[None for _ in range(n)] for _ in range(n + 1)]
        for k in range(len(f) - 1):
            for v in range(n):
                for u in range(n):
                    for e_t in (EdgeType.NORMAL, EdgeType.INVERTED):
                        if network.edge_exist_q((u, v), e_t):
                            c = network.get_cost((u, v), e_t)
                            if f[k + 1][v] > f[k][u] + c:
                                f[k + 1][v] = f[k][u] + c
                                path[k + 1][v] = (u, v), e_t

        return f, path

    @staticmethod
    def __get_x_z(f: tp.List[tp.List[GCostValue]]) -> tp.Optional[NodeId]:
        n = len(f) - 1
        x_z = None
        min_v = Infinity()
        for x in range(n):
            max_v = MinusInfinity()
            for k in range(n):
                if not isinstance(f[k][x], Infinity):
                    cur_v = (f[n][x] - f[k][x]) / (n - k)
                    if cur_v > max_v:
                        max_v = cur_v
            if max_v < min_v:
                min_v = max_v
                x_z = x
        return x_z

    @staticmethod
    def find(network: inetwork.INetwork) -> tp.List[Edge]:
        n = network.size()
        f, path = NegativeCycleFinder.__get_f_and_kpath(network)

        if all([isinstance(i, Infinity) for i in f[n]]):
            return []

        x_z = NegativeCycleFinder.__get_x_z(f)
        res = []
        x_c = x_z
        for i in range(n - 1, -1, -1):
            res.append(path[i][x_c])
            if res[-1][0][0] == x_z:
                break
            x_c = res[-1][0][0]
        res = list(reversed(res))

        return res
