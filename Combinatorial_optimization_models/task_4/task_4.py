import abc
import random

import numpy as np
import matplotlib.pyplot as plt
import Combinatorial_optimization_models.task_2_2.hw_2_2 as hw_2
import Combinatorial_optimization_models.task_3.task_3 as hw_3


class SimulatedAnnealing(hw_3.AbstractSolver):
    def __init__(self, graph, base_solution, temp_func, t_min=1., t_max=10000., seed=42):
        super(SimulatedAnnealing, self).__init__(graph)
        self._vertex_to_clique_id = base_solution.copy()
        self._clique_id_to_vertexes = dict()
        self._d_mut = [hw_3.Mutation.delta_move, hw_3.Mutation.delta_separate, hw_3.Mutation.delta_swap]
        self._mut = [hw_3.Mutation.move, hw_3.Mutation.separate, hw_3.Mutation.swap]
        self._t_max = t_max
        self._t_min = t_min
        self._temp_func = temp_func

        self.obj_mem = []

        random.seed(seed)
        np.random.seed(seed)

        for vertex, clique in enumerate(base_solution):
            if clique not in self._clique_id_to_vertexes:
                self._clique_id_to_vertexes[clique] = set()
            self._clique_id_to_vertexes[clique].add(vertex)

    @staticmethod
    def _get_mutation_id():
        return random.randint(0, 2)

    def _get_delta(self, mut_id, s_current, ctv):
        first_args = [
            (random.randint(0, self._n - 1), random.choice(list(ctv.keys())), ),
            (random.randint(0, self._n - 1), ),
            (random.randint(0, self._n - 1), random.randint(0, self._n - 1), )
        ][mut_id]
        rest_args = s_current, ctv, self._graph
        return self._d_mut[mut_id](*first_args, *rest_args), first_args

    def _mutate(self, mut_id, s_current, ctv, first_args):
        rest_args = s_current, ctv
        self._mut[mut_id](*first_args, *rest_args)
        return rest_args

    @staticmethod
    def _get_copy_of_solution(vertex_to_clique_id, clique_id_to_vertexes):
        return vertex_to_clique_id.copy(), {i: j.copy() for i, j in clique_id_to_vertexes.items()}

    def solve(self):

        t_current = self._t_max
        s_best = self._vertex_to_clique_id, self._clique_id_to_vertexes
        s_best_obj = self.obj_value(s_best[0])

        s_current = self._get_copy_of_solution(*s_best)
        s_cur_obj = s_best_obj

        it = 1
        while t_current > self._t_min:
            mut_id = self._get_mutation_id()
            delta_e, args = self._get_delta(mut_id, *s_current)

            if delta_e > 0 or (delta_e < 0 and random.uniform(0, 1) < np.exp(delta_e / t_current)):
                s_current = self._mutate(mut_id, *s_current, args)
                s_cur_obj += delta_e

            t_current = self._temp_func(it)
            self.obj_mem.append(s_cur_obj)
            if s_cur_obj > s_best_obj:
                s_best = s_current

            it += 1

        return s_best[0]


class AbstractSATemp:
    def __init__(self, alpha, init_temp):
        self._alpha = alpha
        self._init_temp = init_temp

    @abc.abstractmethod
    def __call__(self, it):
        raise NotImplementedError


class LinearTemp(AbstractSATemp):
    def __init__(self, alpha, init_temp):
        super(LinearTemp, self).__init__(alpha, init_temp)

    def __call__(self, it):
        return self._init_temp / (1 + it * self._alpha)


class QudraticTemp(AbstractSATemp):
    def __init__(self, alpha, init_temp):
        super(QudraticTemp, self).__init__(alpha, init_temp)

    def __call__(self, it):
        return self._init_temp / (1 + (it ** 2) * self._alpha)


class GeomTemp(AbstractSATemp):
    def __init__(self, alpha, init_temp):
        super(GeomTemp, self).__init__(alpha, init_temp)
        self._cur_temp = init_temp

    def __call__(self, it):
        self._cur_temp *= self._alpha
        return self._cur_temp

class LogTemp(AbstractSATemp):
    def __init__(self, alpha, init_temp):
        super(LogTemp, self).__init__(alpha, init_temp)

    def __call__(self, it):
        return self._init_temp / (1 + np.log(it) * self._alpha)


def test(max_t, min_t, temp_f, show_q=True):
    graph = hw_2.CompleteGraphGen.generate(70)
    base = hw_3.BaseSolver(graph)
    solution_1 = base.solve()
    print(f"База: {base.obj_value(solution_1)}")

    sa = SimulatedAnnealing(graph, solution_1, temp_f, min_t, max_t)
    solution_2 = sa.solve()

    if show_q:
        plt.plot(sa.obj_mem)
        plt.show()
    print(f"Улучшение: {base.obj_value(solution_2)}")


if __name__ == "__main__":
    test(10, 1, LinearTemp(0.0001, 10))
    test(10, 0.01, QudraticTemp(0.000001, 10))
    test(10, 0.01, GeomTemp(0.9999, 10))
    test(1, 0.01, LogTemp(9.9, 1))
