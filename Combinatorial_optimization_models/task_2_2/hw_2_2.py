import itertools
import logging

import coptpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

COPT_LOG_LEVEL = 0


class CompleteGraphGen:
    @staticmethod
    def generate(vertex_number):
        a = np.random.uniform(-10, 10, (vertex_number, vertex_number))
        a = a + a.T
        for i in range(vertex_number):
            a[i, i] = 0
        return a + a.T


class CliquePartitioningProblem:
    def __init__(self, c_graph: np.array):
        if c_graph.shape[0] != c_graph.shape[1]:
            raise ValueError("Dimensions of the graph matrix should be equal.")

        self._graph = c_graph

        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("CliquePartitioningProblem")
        self._model = self._envr.createModel(name="CliquePartitioningProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, COPT_LOG_LEVEL)

        n_range = range(c_graph.shape[0])
        self._range_2 = list(filter(lambda i: i[0] < i[1], coptpy.tuplelist(itertools.product(n_range, n_range))))
        self._range_3 = \
            list(filter(lambda i: i[0] < i[1] < i[2], coptpy.tuplelist(itertools.product(n_range, n_range, n_range))))

        self._create_model(c_graph)

    def _create_model(self, c_graph):
        x = self._model.addVars(
            self._range_2,
            vtype=coptpy.COPT.BINARY,
            nameprefix="x"
        )

        self._model.setObjective(sum(c_graph[i, j] * x[i, j] for i, j in self._range_2), coptpy.COPT.MAXIMIZE)

        self._model.addConstrs(
            x[i, j] + x[j, k] - x[i, k] <= 1
            for i, j, k in self._range_3
        )

        self._model.addConstrs(
            x[i, j] - x[j, k] + x[i, k] <= 1
            for i, j, k in self._range_3
        )

        self._model.addConstrs(
            -x[i, j] + x[j, k] + x[i, k] <= 1
            for i, j, k in self._range_3
        )

    @staticmethod
    def _find_component(x):
        n = x.shape[0]

        def dfs(v_, checked_, res_):
            if not checked_[v_]:
                checked_[v_] = 1
                res_.append(v_)
                for i_ in range(n):
                    if x[v_, i_] and not checked_[i_]:
                        dfs(i_, checked_, res_)

        components = dict()
        checked = [False for _ in range(n)]
        for i in range(n):
            if not checked[i]:
                res = list()
                dfs(i, checked, res)
                components[len(components.keys())] = res

        res = [0 for _ in range(n)]
        for i, j in components.items():
            for v in j:
                res[v] = i
        return res

    def solve(self):
        """
        Если задача имеет решение (а она всегда имеет решение), то
        выдает список где индекс - номер вершины, значение - номер ее клики.
        """

        self._model.solve()

        if self._model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Problem is infeasible.")
        else:
            n = self._graph.shape[0]
            x = np.full((n, n), False)
            for i, j in self._range_2:
                x[i, j] = bool(self._model.getVarByName(f"x({i},{j})").getInfo("value"))
                x[j, i] = x[i, j]
            return self._find_component(x)


class Visualizer:
    @staticmethod
    def visualize(g, color, highlight_weights=False):
        n = g.shape[0]

        graph = nx.complete_graph(n)

        pos = nx.shell_layout(graph)
        plt.figure()
        nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color=color,
            labels={node: node for node in graph.nodes()}
        )

        if highlight_weights:
            w = dict()
            for i in range(n):
                for j in range(n):
                    if i < j:
                        w[i, j] = str(round(g[i, j], 2))

            nx.draw_networkx_edge_labels(
                graph, pos, edge_labels=w, font_color="red"
            )

        plt.show()


def test():
    g = CompleteGraphGen.generate(8)
    p = CliquePartitioningProblem(g)
    x = p.solve()
    print(f"Graph: {g}")
    print(f"Components: {x}")
    Visualizer.visualize(g, x, highlight_weights=False)
