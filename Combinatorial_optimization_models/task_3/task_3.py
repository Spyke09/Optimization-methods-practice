import numpy as np
from abc import ABC, abstractmethod
import Combinatorial_optimization_models.task_2_2.hw_2_2 as hw_2


class AbstractSolution(ABC):
    def __init__(self):
        self._graph = None
        self._n = -1

    def obj_value(self, sol):
        val = 0
        for i in range(self._n):
            for j in range(self._n):
                if sol[i] == sol[j]:
                    val += self._graph[i, j]
        return val

    @abstractmethod
    def solve(self):
        raise NotImplementedError()


class BaseSolution(AbstractSolution):
    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        self._n = graph.shape[0]

    def solve(self):
        vertex_to_clique_id = [i for i in range(self._n)]
        clique_id_to_vertexes = {i: {i} for i in range(self._n)}

        for cur_cli_id in range(self._n):
            for next_vertex in range(self._n):
                if next_vertex not in clique_id_to_vertexes[cur_cli_id]:
                    next_cli_id = vertex_to_clique_id[next_vertex]
                    delta = 0
                    for i in clique_id_to_vertexes[next_cli_id]:
                        delta -= self._graph[i, next_vertex]
                    for i in clique_id_to_vertexes[cur_cli_id]:
                        delta += self._graph[i, next_vertex]

                    if delta > 0:
                        clique_id_to_vertexes[next_cli_id].remove(next_vertex)
                        clique_id_to_vertexes[cur_cli_id].add(next_vertex)

                        vertex_to_clique_id[next_vertex] = cur_cli_id

        return vertex_to_clique_id


def test1():
    graph = hw_2.CompleteGraphGen.generate(10)

    base = BaseSolution(graph)
    solution = base.solve()

    hw2_solver = hw_2.CliquePartitioningProblem(graph)
    true_solution = hw2_solver.solve()

    print(base.obj_value(solution))
    print(solution)


test1()
