import logging
from dataclasses import dataclass

import coptpy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


@dataclass
class GTSPTWInstance:
    """
    num_clusters - сколько кластеров
    num_nodes - сколько вершин
    points - точки вида [[x1,y1],[x2,y2],...]
    clusters - лист, где индекс - кластер, а элемент - множество вершин кластера
    node_to_cluster - каждой вершине мопоставляется номер кластера
    """
    num_clusters: int
    num_nodes: int
    points: np.array
    cluster_to_node: list
    node_to_cluster: np.array
    tw: np.array
    t: np.array

    def cost(self, u, v):
        """
        Стоимости считаем как евклидово расстояние
        """
        return np.sqrt(np.sum((self.points[u] - self.points[v]) ** 2))

    def same_clusters_q(self, u, v):
        return self.node_to_cluster[u] == self.node_to_cluster[v]

    def edge_exist_q(self, u, v):
        l1 = self.node_to_cluster[u]
        l2 = self.node_to_cluster[v]
        return (self.tw[u, 0] + self.t[u, v] <= self.tw[v, 1]) and (l1 != l2)

    @property
    def nodes(self):
        return range(self.num_nodes)

    @property
    def clusters(self):
        return range(self.num_clusters)

    @property
    def edges(self):
        for i in self.nodes:
            for j in self.nodes:
                if self.edge_exist_q(i, j):
                    yield i, j

    def descendants(self, i):
        for j in self.nodes:
            if self.edge_exist_q(i, j):
                yield j

    def ancestors(self, j):
        for i in self.nodes:
            if self.edge_exist_q(i, j):
                yield i


def generate_GTSPTW_instance(num_clusters, num_nodes):
    """
    Генерация случайного экземпляра задачи GTSP
    Метод кластеризации - KMeans
    """
    assert num_nodes >= num_clusters
    points = np.random.uniform(0, 100, (num_nodes, 2))

    clustering = KMeans(n_clusters=num_clusters, n_init='auto').fit(points)
    labels = clustering.labels_
    clu = [set() for _ in range(num_clusters)]

    for i in range(num_nodes):
        clu[labels[i]].add(i)

    tw = np.random.uniform(0, 100, (num_nodes, 2))
    for i in range(num_nodes):
        tw[i] = tw[i].min(), tw[i].max()
    t = np.random.uniform(0, 5, (num_nodes, num_nodes))

    return GTSPTWInstance(
        num_clusters,
        num_nodes,
        points,
        clu,
        clustering.labels_,
        tw,
        t
    )


class GTSPTWSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._model: coptpy.Model = self._envr.createModel(name="GTSPTWProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, 0)

        self._logger = logging.getLogger("GTSPTWSolver")

        self.x = None

    def solve(self, inst: GTSPTWInstance):
        """
        Возвращает (номер `зала`, номер `порядкового фильма` в зале, `время`)
        для `порядкового фильма` который должен быть поставлен в `зале` в это `время`
        """
        self._create_model(inst)
        self._model.solve()

        if self._model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Problem is infeasible.")
            return None
        else:
            x = list()
            for u, v in inst.edges:
                if bool(self._model.getVarByName(f"x({u},{v})").getInfo("value")):
                    x.append((u, v))

            return self._transform_solution_to_way(x)

    @staticmethod
    def _transform_solution_to_way(solution):
        d = {i: j for i, j in solution}
        s = [0]
        while True:
            s.append(d[s[-1]])
            if s[-1] == 0:
                break
        return s

    def _create_model(self, inst: GTSPTWInstance):
        self.x = x = self._model.addVars(coptpy.tuplelist(inst.edges), vtype=coptpy.COPT.BINARY, nameprefix="x")
        self.y = y = self._model.addVars(inst.nodes, vtype=coptpy.COPT.BINARY, nameprefix="y")
        self.tau = tau = self._model.addVars(inst.clusters, vtype=coptpy.COPT.CONTINUOUS, nameprefix="tau", lb=0)

        clusters = inst.cluster_to_node

        # целевая функция
        self._model.setObjective(
            sum(inst.cost(u, v) * x[u, v] for u, v in inst.edges),
            coptpy.COPT.MAXIMIZE
        )

        # 2
        self._model.addConstrs(
            sum(y[i] for i in clusters[k]) == 1
            for k in inst.clusters
        )

        # 3
        self._model.addConstrs(
            sum(x[i, j] for j in inst.descendants(i)) == y[i]
            for i in inst.nodes
        )

        # 4
        self._model.addConstrs(
            sum(x[i, j] for i in inst.ancestors(j)) == y[j]
            for j in inst.nodes
        )

        # 11.left
        self._model.addConstrs(
            sum(inst.tw[i, 0] * y[i] for i in clusters[k]) <= tau[k]
            for k in inst.clusters
        )

        # 11.right
        self._model.addConstrs(
            sum(inst.tw[i, 1] * y[i] for i in clusters[k]) >= tau[k]
            for k in inst.clusters
        )

        # 12
        self._model.addConstrs(
            tau[h] -
            tau[k] +
            sum(
                inst.t[i, j] * x[i, j]
                for i in clusters[h] for j in clusters[k] if inst.edge_exist_q(i, j)
            )
            <=
            sum(inst.tw[i, 1] * y[i] for i in clusters[h]) -
            sum(inst.tw[j, 0] * y[j] for j in clusters[k]) -
            sum(
                (inst.tw[i, 1] - inst.tw[j][0]) * x[i, j]
                for i in clusters[h] for j in clusters[k] if inst.edge_exist_q(i, j)
            )
            for k in inst.clusters
            for h in inst.clusters
            if k != 0
        )

        # 13
        self._model.addConstrs(
            tau[k] + sum(inst.t[i, 0] * x[i, 0] for i in clusters[k] if inst.edge_exist_q(i, 0)) <= inst.tw[0, 1]
            for k in inst.clusters
        )


if __name__ == "__main__":
    def test1():
        for i in range(100):
            inst = generate_GTSPTW_instance(10, 15)
            solver = GTSPTWSolver()
            solution = solver.solve(inst)
            if solution is not None:
                trace = inst.points[solution]
                plt.scatter(inst.points[:, 0], inst.points[:, 1], c=inst.node_to_cluster, s=100)
                plt.plot(trace[:, 0], trace[:, 1])
                plt.show()
                break



    test1()