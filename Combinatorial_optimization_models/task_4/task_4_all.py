import abc
import random
import typing as tp
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

CliqueId = int
VertexId = int


class CompleteGraphGen:
    @staticmethod
    def generate(vertex_number):
        a = np.random.uniform(-10, 10, (vertex_number, vertex_number))
        a = a + a.T
        for i in range(vertex_number):
            a[i, i] = 0
        return a


class Instance:
    """
    Класс, хранящий экзепляр и текущее решение задачи о кликах.
    """
    def __init__(
            self,
            weight: np.array,
            vertex_id_to_clique_id: tp.Optional[tp.List[CliqueId]] = None,
            clique_id_to_vertexes: tp.Optional[tp.Dict[CliqueId, tp.Set[VertexId]]] = None,
            base_solution_type=0
    ):
        """
        :param weight: двумерный np-массив - симметричная матрица с нулевой диагональю
        :param vertex_id_to_clique_id: отображение id вершины в id клики
        :param clique_id_to_vertexes: отображение id клики в множество id вершин из этой клики
        :param base_solution_type: 0 - начальное решение "каждая вершина - клика"
                                   1 - начальное решение "клика - все вершины"
        """
        self._weight = weight
        self._n = weight.shape[0]
        self._vertex_id_to_clique_id: tp.Optional[tp.List[CliqueId]] = None
        self._clique_id_to_vertexes: tp.Optional[tp.Dict[CliqueId, tp.Set[VertexId]]] = None

        if vertex_id_to_clique_id is not None:
            self._vertex_id_to_clique_id = vertex_id_to_clique_id.copy()
            if clique_id_to_vertexes is not None:
                self._clique_id_to_vertexes = {i: j.copy() for i, j in clique_id_to_vertexes.items()}
            else:
                self._clique_id_to_vertexes = dict()
                for vertex, clique in enumerate(self._vertex_id_to_clique_id):
                    if clique not in self._clique_id_to_vertexes:
                        self._clique_id_to_vertexes[clique] = {vertex}
                    else:
                        self._clique_id_to_vertexes[clique].add(vertex)
        else:
            if base_solution_type == 0:
                self._vertex_id_to_clique_id = [i for i in range(self._n)]
                self._clique_id_to_vertexes = {i: {i} for i in range(self._n)}
            if base_solution_type == 1:
                self._vertex_id_to_clique_id = [0 for _ in range(self._n)]
                self._clique_id_to_vertexes = {0: set(range(self._n))}

    def _mex(self):
        """
        O(#клик)
        :return: кандидат на id для новой клики
        """
        m = 0
        while m in self._clique_id_to_vertexes and len(self._clique_id_to_vertexes[m]) > 0:
            m += 1
        return m

    def copy(self):
        return Instance(
            self._weight,
            self._vertex_id_to_clique_id.copy()
        )

    @property
    def vertex_number(self):
        return self._n

    def in_one_clique_q(self, *args):
        """
        Проверка в одной ли клике две вершины (ребро)
        O(1)
        :param args: 2 вершины или ребро (кортеж двух вершин)
        :return: True если вершины в одной клике иначе False
        """
        if len(args) == 1:
            v1, v2 = args[0]
        elif len(args) == 2:
            v1, v2 = args
        else:
            raise ValueError
        return self._vertex_id_to_clique_id[v1] == self._vertex_id_to_clique_id[v2]

    def vertex_in_clique_q(self, vertex_id, clique_id):
        """
         Проверка находится ли вершина в данной клике
         O(1)
        """
        return self._vertex_id_to_clique_id[vertex_id] == clique_id

    def weight(self, *args):
        if len(args) == 1:
            v1, v2 = args[0]
        elif len(args) == 2:
            v1, v2 = args
        else:
            raise ValueError
        return self._weight[v1, v2]

    def edge_iter(self):
        for v1 in range(self.vertex_number):
            for v2 in range(self.vertex_number):
                if v1 < v2:
                    yield v1, v2

    def clique_id_iter(self):
        for i in self._clique_id_to_vertexes:
            yield i

    def clique_iter(self):
        for i, j in self._clique_id_to_vertexes.items():
            yield i

    def obj_value(self):
        """
        Расчитать целевую функцию
        :return: значение целевой функции для данного решения
        O(n^2)
        """
        val = 0
        for edge in self.edge_iter():
            if self.in_one_clique_q(edge):
                val += self.weight(edge)
        return val

    def renumerate(self):
        """
        Перенумеровать id клик в соответсвенно {0, 1, ..., #клик}
        """
        mapping = dict()
        c = 0
        for i in self._vertex_id_to_clique_id:
            if i not in mapping:
                mapping[i] = c
                c += 1

        self._vertex_id_to_clique_id = [mapping[i] for i in self._vertex_id_to_clique_id]
        self._clique_id_to_vertexes = {mapping[i]: j for i, j in self._clique_id_to_vertexes.items() if len(j) > 0}

    def get_clique_id(self, vertex_id):
        """
        Геттер для self._vertex_id_to_clique_id
        """
        return self._vertex_id_to_clique_id[vertex_id]

    def get_vertexes(self, clique_id):
        """
        Итератор!!! для получения элементов множества вершин из данной клики
        """
        for i in self._clique_id_to_vertexes[clique_id]:
            yield i

    def move(self, vertex_id, clique_id):
        """
        Перемещение вершины в данную клику
        O(1)
        """
        old_cli_id = self._vertex_id_to_clique_id[vertex_id]
        self._clique_id_to_vertexes[old_cli_id].remove(vertex_id)

        self._clique_id_to_vertexes[clique_id].add(vertex_id)

        self._vertex_id_to_clique_id[vertex_id] = clique_id

    def separate(self, vertex):
        """
        Изолирование вершины
        O(1) + O(#клик)
        """
        clique_id = self._vertex_id_to_clique_id[vertex]
        if len(self._clique_id_to_vertexes[clique_id]) == 1:
            return

        self._clique_id_to_vertexes[clique_id].remove(vertex)
        clique_id_v = self._mex()
        self._vertex_id_to_clique_id[vertex] = clique_id_v
        self._clique_id_to_vertexes[clique_id_v] = {vertex}

    def swap(self, vertex_1, vertex_2):
        """
        Поменять вершины кликами.
        O(1)
        """
        clique_id_1 = self._vertex_id_to_clique_id[vertex_1]
        clique_id_2 = self._vertex_id_to_clique_id[vertex_2]
        if clique_id_2 == clique_id_1:
            return

        self.move(vertex_1, clique_id_2)
        self.move(vertex_2, clique_id_1)

    def delta_move(self, vertex, clique_id):
        """
        Предварительный рассчет изменения целевой функции при перемещении вершины в данную клику.
        O(n) в худшем случае (вместо вызова obj_val с O(n^2))
        """
        clique_id_v = self._vertex_id_to_clique_id[vertex]
        if clique_id_v == clique_id:
            return 0

        delta1 = 0
        for i in self._clique_id_to_vertexes[clique_id_v]:
            delta1 -= self._weight[vertex, i]

        delta2 = 0
        for i in self._clique_id_to_vertexes[clique_id]:
            delta2 += self._weight[vertex, i]

        return delta1 + delta2

    def delta_separate(self, vertex):
        """
        Предварительный рассчет изменения целевой функции при изолировании вершины.
        O(n) в худшем случае (вместо вызова obj_val с O(n^2))
        """
        delta = 0
        clique_id = self._vertex_id_to_clique_id[vertex]
        for i in self._clique_id_to_vertexes[clique_id]:
            delta -= self._weight[vertex, i]

        return delta

    def delta_swap(self, vertex_1, vertex_2):
        """
        Предварительный рассчет изменения целевой функции при swap-e двух вершин.
        O(n) в худшем случае (вместо вызова obj_val с O(n^2))
        """
        delta = 0
        if self._vertex_id_to_clique_id[vertex_2] == self._vertex_id_to_clique_id[vertex_1]:
            return delta

        delta1 = self.delta_move(vertex_1, self._vertex_id_to_clique_id[vertex_2])
        delta2 = self.delta_move(vertex_2, self._vertex_id_to_clique_id[vertex_1])

        return delta1 + delta2 - 2 * self._weight[vertex_1, vertex_2]

    def smart_move(self, vertex_id, clique_id):
        if self.delta_move(vertex_id, clique_id) > 0:
            self.move(vertex_id, clique_id)

    def smart_separate(self, vertex_id):
        if self.delta_separate(vertex_id) > 0:
            self.separate(vertex_id)

    def smart_swap(self, vertex_1, vertex_2):
        if self.delta_swap(vertex_1, vertex_2) > 0:
            self.swap(vertex_1, vertex_2)


class AbstractSolver(ABC):
    @abstractmethod
    def solve(self, instance: Instance):
        raise NotImplementedError()


class BaseSolver(AbstractSolver):
    def solve(self, instance):
        """
        Базовое решение для Instance.
        O(n ^ 3)
        """
        instance = instance.copy()
        n = instance.vertex_number

        for cur_cli_id in instance.clique_id_iter():  # O(n)
            for next_vertex in range(n):  # O(n)
                instance.smart_move(next_vertex, cur_cli_id)  # O(n)

        return instance


class LocalSearch(AbstractSolver):
    """
    Локальный поиск.
    Стратегии:
        1) Перемещение, изолирование - _strategy_1
        2) Перемещение, изолирование и swap - _strategy_2
    Виды "шагов":
        1) Жадный шаг - перебираем все возможные мутации из данного решения и выбираем наилучшее - _step_greed
        2) Быстрый шаг - перебираем все возможные мутации пока не улучшим целевую функцию - _step_stop_first
    """
    def __init__(self, step_number=10, strategy_id=0, step_id=0):
        """
        :param step_number: кол-во шагов
        :param strategy_id: 0 - _strategy_1, 1 - _strategy_2
        :param step_id: 0 - _step_greed, 1 - _step_stop_first
        """
        self._step_number = step_number
        self._strategy_id = strategy_id
        self._step_id = step_id

    @staticmethod
    def _strategy_1(instance):
        for vertex in range(instance.vertex_number):
            for clique in instance.clique_id_iter():
                yield instance.delta_move(vertex, clique), instance.move, vertex, clique

        for vertex in range(instance.vertex_number):
            yield instance.delta_separate(vertex), instance.separate, vertex

    @staticmethod
    def _strategy_2(instance):
        for i in LocalSearch._strategy_1(instance):
            yield i

        for v1, v2 in instance.edge_iter():
            yield instance.delta_swap(v1, v2), instance.swap, v1, v2

    @staticmethod
    def _step_stop_first(instance: Instance, strategy):
        for delta, action, *args in strategy(instance):
            if delta > 0:
                action(*args)
                return delta
        return None

    @staticmethod
    def _step_greed(instance: Instance, strategy):
        best_delta = 0
        next_action = None
        next_args = None
        for delta, action, *args in strategy(instance):
            if delta > best_delta:
                best_delta = delta
                next_action = action
                next_args = args

        if best_delta > 0:
            next_action(*next_args)
            return best_delta
        else:
            return None

    def solve(self, instance):
        instance = instance.copy()

        strategy = [self._strategy_1, self._strategy_2][self._strategy_id]
        step = [self._step_greed, self._step_stop_first][self._step_id]

        for i in range(self._step_number):
            res = step(instance, strategy)
            if res is None:
                break

        return instance


class SimulatedAnnealing(AbstractSolver):
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
    def _get_random_vertex_id(instance: Instance):
        return random.randint(0, instance.vertex_number - 1)

    @staticmethod
    def _get_random_clique_id(instance: Instance):
        clique = list(instance.clique_id_iter())
        return random.choice(clique)

    def _get_delta(self, instance, mut_id):
        """
        Получение изменения целевой функции при мутации
        O(n) (вместо O(n^2) при прямом подсчете)
        """
        rvf1 = self._get_random_vertex_id(instance)
        rvf2 = self._get_random_vertex_id(instance)
        rcf = self._get_random_clique_id(instance)
        args = [(rvf1, rcf), (rvf1,), (rvf1, rvf2)][mut_id]

        d_mut = [instance.delta_move, instance.delta_separate, instance.delta_swap]
        return d_mut[mut_id](*args), args

    @staticmethod
    def _mutate(instance, mut_id, args):
        [instance.move, instance.separate, instance.swap][mut_id](*args)

    def solve(self, instance):
        s_best = instance.copy()
        s_best_obj = instance.obj_value()

        s_current = instance.copy()
        s_cur_obj = s_best_obj

        it = 1
        t_current = self._t_max
        self.obj_mem.clear()
        while t_current > self._t_min:
            mut_id = self._get_mutation_id()
            delta_e, args = self._get_delta(s_current, mut_id)

            if delta_e > 0 or (delta_e < 0 and random.uniform(0, 1) < np.exp(delta_e / t_current)):
                self._mutate(s_current, mut_id,  args)
                s_cur_obj += delta_e

            self.obj_mem.append(s_cur_obj)  # запись изменений целевой функции для графика
            if s_cur_obj > s_best_obj:
                s_best = s_current.copy()

            t_current = self._temp_func(it)
            it += 1

        return s_best


# различные функции температуры
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


if __name__ == "__main__":
    def test(max_t, min_t, temp_f, show_q=True):
        print(temp_f)
        graph = CompleteGraphGen.generate(100)
        instance = Instance(graph)

        base = BaseSolver()
        solution = base.solve(instance)
        print(f"База: {solution.obj_value()}")

        sa = SimulatedAnnealing(temp_f, min_t, max_t)
        solution_2 = sa.solve(solution)

        if show_q:
            plt.plot(sa.obj_mem)
            plt.show()
        print(f"Улучшение SA: {solution_2.obj_value()}")

        ls = LocalSearch(1000, 1, 1)
        solution_3 = ls.solve(solution)

        print(f"Улучшение LS: {solution_3.obj_value()}\n")

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
