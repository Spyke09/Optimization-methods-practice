import abc
import random

import matplotlib.pyplot as plt
import numpy as np

import Combinatorial_optimization_models.task_2_2.hw_2_2 as hw_2
import Combinatorial_optimization_models.task_3.task_3 as hw_3


class SimulatedAnnealing(hw_3.AbstractSolver):
    def __init__(self, temp_func, t_min=1., t_max=10000., seed=42):
        self._t_max = t_max
        self._t_min = t_min
        self._temp_func = temp_func

        self.obj_mem = []

        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _get_mutation_id():
        return random.randint(0, 2)

    @staticmethod
    def _get_random_vertex_id(instance: hw_3.Instance):
        return random.randint(0, instance.vertex_number - 1)

    @staticmethod
    def _get_random_clique_id(instance: hw_3.Instance):
        clique = list(instance.clique_id_iter())
        return random.choice(clique)

    def _get_delta(self, instance, mut_id):
        rvf = self._get_random_vertex_id
        rcf = self._get_random_clique_id
        args = [
            (rvf(instance), rcf(instance)),
            (rvf(instance),),
            (rvf(instance), rvf(instance))
        ][mut_id]

        d_mut = [
            instance.delta_move,
            instance.delta_separate,
            instance.delta_swap
        ]
        return d_mut[mut_id](*args), args

    def _mutate(self, instance, mut_id, args):
        mut = [
            instance.move,
            instance.separate,
            instance.swap
        ]
        mut[mut_id](*args)

    def solve(self, instance):
        t_current = self._t_max
        s_best = instance.copy()
        s_best_obj = instance.obj_value()

        s_current = instance.copy()
        s_cur_obj = s_best_obj

        it = 1
        while t_current > self._t_min:
            mut_id = self._get_mutation_id()
            delta_e, args = self._get_delta(s_current, mut_id)

            if delta_e > 0 or (delta_e < 0 and random.uniform(0, 1) < np.exp(delta_e / t_current)):
                self._mutate(s_current, mut_id,  args)
                s_cur_obj += delta_e

            t_current = self._temp_func(it)
            self.obj_mem.append(s_cur_obj)
            if s_cur_obj > s_best_obj:
                s_best = s_current.copy()

            it += 1

        return s_best


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


def test(max_t, min_t, temp_f, show_q=False):
    print(temp_f)
    graph = hw_2.CompleteGraphGen.generate(70)
    instance = hw_3.Instance(graph)

    base = hw_3.BaseSolver()
    solution = base.solve(instance)
    print(f"База: {solution.obj_value()}")

    sa = SimulatedAnnealing(temp_f, min_t, max_t)
    solution_2 = sa.solve(solution)

    if show_q:
        plt.plot(sa.obj_mem)
        plt.show()
    print(f"Улучшение SA: {solution_2.obj_value()}")

    ls = hw_3.LocalSearch(1000, 1, 1)
    solution_3 = ls.solve(solution)

    print(f"Улучшение LS: {solution_3.obj_value()}\n")


if __name__ == "__main__":
    test(10, 1, LinearTemp(0.0001, 10))
    test(10, 0.01, QudraticTemp(0.000001, 10))
    test(10, 0.01, GeomTemp(0.9999, 10))
    test(1, 0.01, LogTemp(9.9, 1))

    # <__main__.LinearTemp object at 0x7f6921121ab0>
    # База: 1642.530046205885
    # Улучшение SA: 2025.4653792079635
    # Улучшение LS: 2085.060150780179
    #
    # <__main__.QudraticTemp object at 0x7f691cb25a20>
    # База: 1589.2322502379031
    # Улучшение SA: 2008.8253483753867
    # Улучшение LS: 1925.0064615462124
    #
    # <__main__.GeomTemp object at 0x7f691cb25a20>
    # База: 1589.2322502379031
    # Улучшение SA: 1855.5354931346335
    # Улучшение LS: 1925.0064615462124
    #
    # <__main__.LogTemp object at 0x7f6913b64be0>
    # База: 1589.2322502379031
    # Улучшение SA: 1883.1816024348796
    # Улучшение LS: 1925.0064615462124
    #
    #
    # Process finished with exit code 0
