import heapq
import itertools
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


@dataclass
class GTSPInstance:
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

    def edges(self, layers):
        """
        Набор ребра зависит от перестановки (layers)
        """
        for l1, l2 in zip(layers, layers[1:] + layers[0]):
            for i in self.clusters[l1]:
                for j in self.clusters[l2]:
                    yield i, j

    def neighbors(self, layers: list, v):
        """
        Потомки данной вершины. Как и ребра, зависят от перестановки
        """
        clus_v = self.node_to_cluster[v]
        pos_clus_v = layers.index(clus_v)
        pos_clus_u = (pos_clus_v + 1) % len(layers)
        clus_u = layers[pos_clus_u]
        for u in self.clusters[clus_u]:
            yield u


def generate_GTSP_instance(num_clusters, num_nodes):
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

    return GTSPInstance(
        num_clusters,
        num_nodes,
        points,
        clu,
        clustering.labels_
    )


class GTSPSolver:
    def find_min_way(self, instance: GTSPInstance, layers):
        """
        Запуск дейкстры из каждой вершины нулевого слоя.
        Выбирается наилучший путь из всех получаемых путей.
        """
        best_obj = None
        best_way = None
        for start in instance.clusters[layers[0]]:
            dist = [float('infinity') for _ in instance.nodes]
            prev = [None for _ in instance.nodes]
            dist[start] = 0

            priority_queue = [(0, start)]

            while priority_queue:
                current_distance, cur_node = heapq.heappop(priority_queue)

                if current_distance > dist[cur_node]:
                    continue

                for neighbor in instance.neighbors(layers, cur_node):
                    distance = current_distance + instance.cost(cur_node, neighbor)

                    if distance < dist[neighbor] or neighbor == start and dist[neighbor] == 0:
                        dist[neighbor] = distance
                        prev[neighbor] = cur_node
                        heapq.heappush(priority_queue, (distance, neighbor))

            min_dist = min(dist[i] for i in instance.clusters[layers[0]])
            if best_obj is None or min_dist < best_obj:
                best_obj = min_dist
                v = [u for u, d in enumerate(dist) if d == min_dist][0]
                way = [v]
                while prev[way[-1]] != v:
                    way.append(prev[way[-1]])
                way.append(prev[way[-1]])
                best_way = way[::-1]

        return best_obj, best_way

    @staticmethod
    def get_permutations(n):
        """
        Здесь не совсем классический перестановки.
        Для n = 3 генерируются не (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
        а (0, 1, 2), (0, 2, 1)
        Т. е. без циклических повторений.
        """
        for i in itertools.permutations(range(n - 1)):
            yield n - 1, *i

    def solve(self, inst):
        """
        Сам алгоритм из статьи.
        """

        # Ищу кластер, в котором меньше всего элементов, чтобы делать его первым во всех перестановках
        min_cluster_size = min(len(i) for i in inst.clusters)
        min_cluster = [i for i in range(inst.num_clusters) if len(inst.clusters[i]) == min_cluster_size][0]

        best_obj = None
        best_way = None
        for perm in self.get_permutations(inst.num_clusters):
            # перестраиваю цепочку так, чтобы первым был наименьший кластер
            idx = perm.index(min_cluster)
            perm = perm[idx:] + perm[:idx]

            # Обновление лучшей стоимости и лучшего пути.
            obj, way = self.find_min_way(inst, perm)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_way = way

        return best_way


def test_1():
    inst = generate_GTSP_instance(4, 10)
    solver = GTSPSolver()
    solution = solver.solve(inst)
    trace = inst.points[solution]
    plt.scatter(inst.points[:, 0], inst.points[:, 1], c=inst.node_to_cluster, s=100)
    plt.plot(trace[:, 0], trace[:, 1])
    plt.show()


test_1()
