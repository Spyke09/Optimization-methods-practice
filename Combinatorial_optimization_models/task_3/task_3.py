from abc import ABC, abstractmethod

import numpy as np
import time
import Combinatorial_optimization_models.task_2_2.hw_2_2 as hw_2


class AbstractSolver(ABC):
    def __init__(self, graph):
        self._graph = graph
        self._n = graph.shape[0]

    def obj_value(self, sol):
        val = 0
        for i in range(self._n):
            for j in range(self._n):
                if i < j and sol[i] == sol[j]:
                    val += self._graph[i, j]
        return val

    @abstractmethod
    def solve(self):
        raise NotImplementedError()


class BaseSolver(AbstractSolver):
    def __init__(self, graph):
        super(BaseSolver, self).__init__(graph)

    @staticmethod
    def renumerate(solution):
        mapping = dict()
        c = 0
        for i in solution:
            if i not in mapping:
                mapping[i] = c
                c += 1

        return [mapping[i] for i in solution]

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

        return self.renumerate(vertex_to_clique_id)


class LocalSearch(AbstractSolver):
    def __init__(self, graph, base_solution):
        super(LocalSearch, self).__init__(graph)
        self._vertex_to_clique_id = base_solution.copy()
        self._clique_id_to_vertexes = dict()

        for vertex, clique in enumerate(base_solution):
            if clique not in self._clique_id_to_vertexes:
                self._clique_id_to_vertexes[clique] = set()
            self._clique_id_to_vertexes[clique].add(vertex)

    @property
    def _clique_id_set(self):
        return self._clique_id_to_vertexes.keys()

    # функции для мутирования
    def _move(self, vertex, clique_id):
        clique_id_v = self._vertex_to_clique_id[vertex]
        if clique_id_v == clique_id:
            return

        self._vertex_to_clique_id[vertex] = clique_id
        self._clique_id_to_vertexes[clique_id_v].remove(vertex)
        self._clique_id_to_vertexes[clique_id].add(vertex)

    def _separate(self, vertex):
        clique_id = self._vertex_to_clique_id[vertex]
        self._clique_id_to_vertexes[clique_id].remove(vertex)

        clique_id_v = max(self._vertex_to_clique_id) + 1
        self._vertex_to_clique_id[vertex] = clique_id_v
        self._clique_id_to_vertexes[clique_id_v] = {vertex}
        return None

    def _swap(self, vertex_1, vertex_2):
        clique_id_1 = self._vertex_to_clique_id[vertex_1]
        clique_id_2 = self._vertex_to_clique_id[vertex_2]
        if clique_id_2 == clique_id_1:
            return

        self._move(vertex_1, clique_id_2)
        self._move(vertex_2, clique_id_1)

    # функции для расчета полезности мутирования без (до) самого мутирования
    def _delta_move(self, vertex, clique_id):
        clique_id_v = self._vertex_to_clique_id[vertex]
        if clique_id_v == clique_id:
            return 0

        delta1 = 0
        for i in self._clique_id_to_vertexes[clique_id_v]:
            delta1 -= self._graph[vertex, i]

        delta2 = 0
        for i in self._clique_id_to_vertexes[clique_id]:
            delta2 += self._graph[vertex, i]
        return delta1 + delta2

    def _delta_separate(self, vertex):
        delta = 0
        clique_id = self._vertex_to_clique_id[vertex]
        for i in self._clique_id_to_vertexes[clique_id]:
            delta -= self._graph[vertex, i]
        return delta

    def _delta_swap(self, vertex_1, vertex_2):
        delta = 0
        if self._vertex_to_clique_id[vertex_2] == self._vertex_to_clique_id[vertex_1]:
            return delta

        delta1 = self._delta_move(vertex_1, self._vertex_to_clique_id[vertex_2])
        delta2 = self._delta_move(vertex_2, self._vertex_to_clique_id[vertex_1])
        return delta1 + delta2 - 2 * self._graph[vertex_1, vertex_2]

    def _strategy_1(self):
        for vertex in range(self._n):
            for clique in self._clique_id_set:
                yield self._delta_move(vertex, clique), self._move, vertex, clique

        for vertex in range(self._n):
            yield self._delta_separate(vertex), self._separate, vertex

    def _strategy_2(self):
        for i in self._strategy_1():
            yield i

        for i in range(self._n):
            for j in range(self._n):
                if i < j:
                    yield self._delta_swap(i, j), self._swap, i, j

    @staticmethod
    def _step_stop_first(strategy):
        for delta, action, *args in strategy():
            if delta > 0:
                action(*args)
                return delta
        return None

    @staticmethod
    def _step_greed(strategy):
        best_delta = 0
        next_action = None
        next_args = None
        for delta, action, *args in strategy():
            if delta > best_delta:
                best_delta = delta
                next_action = action
                next_args = args

        if best_delta > 0:
            next_action(*next_args)
            return best_delta
        else:
            return None

    def solve(self, step_number=10, strategy_id=0, step_id=0):
        strategy = [self._strategy_1, self._strategy_2]
        step = [self._step_greed, self._step_stop_first]

        for i in range(step_number):
            res = step[step_id](strategy[strategy_id])
            if res is None:
                break

        return self._vertex_to_clique_id


def test1():
    graph = hw_2.CompleteGraphGen.generate(10)

    base = BaseSolver(graph)
    solution = base.solve()

    hw2_solver = hw_2.CliquePartitioningProblem(graph)
    true_solution = hw2_solver.solve()

    print(f"obj: {base.obj_value(solution)}")
    print(f"true obj: {base.obj_value(true_solution)}")
    print(f"true_solution: {true_solution}")
    print(f"solution:      {solution}")


def test2():
    graph = hw_2.CompleteGraphGen.generate(10)

    base = BaseSolver(graph)
    solution_1 = base.solve()

    ls = LocalSearch(graph, solution_1)
    solution_2 = ls.solve(1000, 0, 1)

    hw2_solver = hw_2.CliquePartitioningProblem(graph)
    true_solution = hw2_solver.solve()

    print(f"obj: {base.obj_value(solution_1)}")
    print(f"obj: {base.obj_value(solution_2)}")
    print(f"true_solution: {base.obj_value(true_solution)}")


def main_test():
    # (0, 0) - ((a,b)  , greed     ),
    # (0, 1) - ((a,b)  , first_stop),
    # (1, 0) - ((a,b,c), greed     ),
    # (1, 1) - ((a,b,c), first_stop),

    # нужно (0, 0) VS (1, 0)    и    (1, 0) VS (1, 1)
    def temp_def(a1, a2):
        dtimes = np.array([0.0, 0.0])
        ds = np.array([0.0, 0.0])
        times = np.array([0.0, 0.0])
        n = 50
        for test_number in range(n):
            graph = hw_2.CompleteGraphGen.generate(70)
            base = BaseSolver(graph)
            solution_1 = base.solve()
            ls = LocalSearch(graph, solution_1)

            st = time.time()
            solution_2 = ls.solve(1000, *a1)
            p = time.time()
            ls = LocalSearch(graph, solution_1)
            p = time.time()

            solution_3 = ls.solve(1000, *a2)
            fn = time.time()
            dtimes += (p - st, fn - p)
            s1 = base.obj_value(solution_1)
            ds += (base.obj_value(solution_2) - s1, base.obj_value(solution_3) - s1)
        dtimes /= n
        print(f"{a1} VS {a2} time: {dtimes}")
        print(f"{a1} VS {a2} obj: {ds}")

    temp_def((0, 0), (1, 0))
    temp_def((1, 0), (1, 1))
    # time - затраченное время, obj - насколько учлучшилась функция после LS
    # (0, 0) VS (1, 0) time: [0.09311929 0.82128852]
    # (0, 0) VS (1, 0) obj: [15404.55328308 15594.92377154]
    # Видно, что в среднем (0, 0) работает быстрее, причем улучшает в среднем примерно так же как (1, 0)
    # Итого, берем ((a,b), greed)

    # (1, 0) VS (1, 1) time: [0.8310325  0.11410839]
    # (1, 0) VS (1, 1) obj: [15174.54810291 15753.84239695]
    # Здесь вариант (1, 1) работает чуть быстее, причем улучшает в среднем чуть лучше, чем (1, 0)
    # Итого, ((a,b,c), first_stop) чуть лучше


main_test()
