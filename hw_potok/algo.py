import typing as tp
from collections import deque

import inetwork
from inetwork import EdgeType, NodeId, EdgeId, Edge


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
        start = network.get_source()
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


class NegativeCycleFinder:
    @staticmethod
    def find(network: inetwork.INetwork):
        n = network.size()
        f: tp.List[tp.List[tp.Optional[inetwork.CostValue]]] = [[None for _ in range(n)] for _ in range(n + 1)]
        for i in range(n):
            f[0][i] = 0
        path = [None for _ in range(n)]
        for k in range(len(f) - 1):
            for v in range(n):
                r = None
                for u in range(n):
                    for e_t in (EdgeType.NORMAL, EdgeType.INVERTED):
                        if not network.edge_exist_q((u, v), e_t):
                            continue
                        c = network.get_cost((u, v), e_t)
                        if not f[k + 1][v] is None:
                            if f[k + 1][v] > f[k][u] + c:
                                f[k + 1][v] = f[k][u] + c
                                if k == n - 1:
                                    path[v] = (u, v), e_t
                        else:
                            if not f[k][u] is None:
                                f[k + 1][v] = f[k][u] + c
                                if k == n - 1:
                                    path[v] = (u, v), e_t
                            else:
                                f[k + 1][v] = f[k][u]

        print(path)
        if not any(f[n]):
            return None

        x_z = None
        min_v = None
        for x in range(n):
            max_v = None
            for k in range(n):
                if not f[k][x] is None:
                    cur_v = (f[n][x] - f[k][x]) / (n - k)
                    if (not max_v) or cur_v > max_v:
                        max_v = cur_v
            if (not min_v) or max_v < min_v:
                min_v = max_v
                x_z = x

        res = []
        x_c = x_z
        for i in range(n + 1):
            res.append(path[x_c])
            if path[x_c][0][0] == x_z:
                break
            x_c = path[x_c][0][0]
        res = list(reversed(res))
        print(res)

        # print(path)