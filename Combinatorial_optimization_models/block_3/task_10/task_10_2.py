import logging
from dataclasses import dataclass

import coptpy
import numpy as np
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
    clusters: list
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

    @property
    def nodes(self):
        return range(self.num_nodes)

    def edges(self):
        for l1 in range(self.num_clusters):
            for l2 in range(self.num_clusters):
                if l1 == l2:
                    continue
                for i in self.clusters[l1]:
                    for j in self.clusters[l2]:
                        if self.tw[i] + self.t[i, j] <= self.tw[j]:
                            yield i, j


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
    t = np.random.uniform(0, 5, num_nodes)

    return GTSPTWInstance(
        num_clusters,
        num_nodes,
        points,
        clu,
        clustering.labels_,
        tw
    )


class GTSPTWSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._model: coptpy.Model = self._envr.createModel(name="CehProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, 0)

        self._logger = logging.getLogger("CehSolver")

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
            self.x = list()
            # for p, s, t in inst.x_iter():
            #     if bool(self._model.getVarByName(f"x({p},{s},{t})").getInfo("value")):
            #         self.x.append((p, s, t))

            return self.x

    def _create_model(self, inst: GTSPTWInstance):
        self.x = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.INTEGER, nameprefix="x")
        self.y = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.BINARY, nameprefix="y")

        # целевая функция
        self._model.setObjective(
            0,
            coptpy.COPT.MAXIMIZE
        )

        self._model.addConstrs(

        )
