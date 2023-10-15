import itertools
import logging
import random

import coptpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import typing as tp
from collections import defaultdict

from dataclasses import dataclass

COPT_LOG_LEVEL = 0
ContainerId = int
PlatformId = int
SubsetId = int


@dataclass
class CPInstance:
    subsets: tp.List[tp.Set[ContainerId]]  # список всех расстановок
    f_subsets: tp.Dict[PlatformId, tp.List[int]]  # для каждой платформы id-ы допустимых расстановок контейнеров
    containers: tp.Set[ContainerId]  # список контейнеров
    container_to_priority: tp.Dict[ContainerId, int]  # приоритеты для контейнеров
    container_to_batch: tp.Dict[ContainerId, int]  # номер партии для каждого контейнера
    priority_to_containers: tp.Dict[int, tp.Set[ContainerId]]  # приоритет в мн-во таких контейнеров
    batch_to_containers: tp.Dict[int, tp.Set[ContainerId]]  # партия в мн-во таких контейнеров
    platform_to_priority: tp.Dict[PlatformId, int]  # приоритет для каждой платформы

    def r(self, sc, b):
        return all(self.container_to_batch[i] == b for i in self.subsets[sc])

    def fi(self, b):
        return len(self.batch_to_containers[b])

    def psi(self, o):
        return len(self.priority_to_containers[o])

    def h(self, sc, o):
        return sum(self.container_to_priority[i] == o for i in self.subsets[sc])

    @property
    def number_of_subsets(self):
        return len(self.subsets)

    @property
    def number_of_containers(self):
        return len(self.containers)

    @property
    def platform_ids(self):
        return self.f_subsets.keys()

    @property
    def batches(self):
        return sorted(self.batch_to_containers.keys())

    @property
    def priorities(self):
        return sorted(self.priority_to_containers.keys())

    @property
    def priorities_without_max(self):
        max_ = max(self.priority_to_containers.keys())
        return sorted(i for i in self.priority_to_containers.keys() if i != max_)

    @property
    def subsets_ids(self):
        return range(self.number_of_subsets)


class CPSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._model: coptpy.Model = self._envr.createModel(name="CliquePartitioningProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, COPT_LOG_LEVEL)

        self._logger = logging.getLogger("CPSolver")

        self.x = None

    def solve(self, inst: CPInstance, alpha=0.5):
        self._create_model(inst, alpha)
        self._model.solve()

        if self._model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Problem is infeasible.")
        else:
            self.x = dict()
            for p, sc in itertools.product(inst.platform_ids, inst.subsets_ids):
                self.x[p, sc] = bool(self._model.getVarByName(f"x({p},{sc})").getInfo("value"))

        return self.x

    def _create_model(self, inst: CPInstance, alpha=0.5):
        x = self._model.addVars(coptpy.tuplelist(
            itertools.product(inst.platform_ids, inst.subsets_ids)
        ), vtype=coptpy.COPT.BINARY, nameprefix="x")

        y = self._model.addVars(inst.batches, vtype=coptpy.COPT.BINARY, nameprefix="y")

        y1 = self._model.addVars(inst.priorities_without_max, vtype=coptpy.COPT.BINARY, nameprefix="y1")

        # На каждой платформе с нулевым приоритетом должен быть размещен как минимум один контейнер.
        self._model.addConstrs(
            sum(x[p, sc] for sc in inst.subsets_ids) == 1
            for p in inst.platform_ids if inst.platform_to_priority[p] == 0
        )

        # Для меньших приоритетов платформ:
        self._model.addConstrs(
            sum(x[p, sc] for sc in inst.subsets_ids) <= 1
            for p in inst.platform_ids if inst.platform_to_priority[p] > 0
        )

        # Каждый контейнер может разместиться не более, чем на 1ой платформе:
        self._model.addConstrs(
            sum(
                x[p, sc]
                for p in inst.platform_ids
                for sc in inst.f_subsets[p] if c in inst.subsets[sc]
            ) <= 1
            for c in inst.containers
        )

        # Каждая партия может быть полностью расставлена на платформах, либо никто из партии не ставится на платформы:
        self._model.addConstrs(
            sum(
                x[p, sc]
                for p in inst.platform_ids
                for sc in inst.f_subsets[p] if inst.r(sc, b) == 1
            ) == inst.fi(b) * y[b]
            for b in inst.batches
        )

        # Контейнеры размещаются на платформах по своим приоритетам:
        # Если не все контейнеры высшего приоритета расставлены на платформах,
        # то запрещается расставлять контейнеры меньшего приоритета:
        self._model.addConstrs(
            sum(
                inst.h(sc, o) * x[p, sc]
                for p in inst.platform_ids
                for sc in inst.f_subsets[p] if any(inst.container_to_priority[c] == o for c in inst.subsets[sc])

            ) >= inst.psi(o) * y1[o]
            for o in inst.priorities_without_max
        )

        self._model.addConstrs(
            sum(
                inst.h(sc, o2) * x[p, sc]
                for p in inst.platform_ids
                for sc in inst.f_subsets[p] if any(inst.container_to_priority[c] == o2 for c in inst.subsets[sc])

            ) <= inst.psi(o2) * y1[o1]
            for o1 in inst.priorities_without_max
            for o2 in inst.priorities_without_max if o2 == o1 + 1
        )

        self._model.setObjective(
            alpha * sum(
                x[p, sc] * len(inst.subsets[sc])
                for p in inst.platform_ids
                for sc in inst.f_subsets[p]
            ) -
            (1 - alpha) * sum(
                x[p, sc]
                for p in inst.platform_ids if inst.platform_to_priority[p] > 0
                for sc in inst.f_subsets[p]
            ),
            coptpy.COPT.MAXIMIZE
        )

    def total_container(self, inst):
        s = list()
        for p in inst.platform_ids:
            for sc in inst.f_subsets[p]:
                if self.x[p, sc]:
                    for c in inst.subsets[sc]:
                        s.append(c)
        return s

    def containers_by_batch(self, inst: CPInstance):
        s = defaultdict(set)
        for p in inst.platform_ids:
            for sc in inst.f_subsets[p]:
                if self.x[p, sc]:
                    for c in inst.subsets[sc]:
                        s[inst.container_to_batch[c]].add(c)

        return s

    def containers_by_priority(self, inst: CPInstance):
        s = defaultdict(set)
        for p in inst.platform_ids:
            for sc in inst.f_subsets[p]:
                if self.x[p, sc]:
                    for c in inst.subsets[sc]:
                        s[inst.container_to_priority[c]].add(c)

        return s

    def platforms_by_priority(self, inst: CPInstance):
        s = defaultdict(set)
        for p in inst.platform_ids:
            for sc in inst.f_subsets[p]:
                if self.x[p, sc]:
                    s[inst.platform_to_priority[p]].add(p)
        return s

    def platforms_by_load(self, inst: CPInstance):
        s = defaultdict(int)
        for p in inst.platform_ids:
            for sc in inst.f_subsets[p]:
                if self.x[p, sc]:
                    s[len(inst.subsets[sc])] += 1
        return s


class RandomPCInstanceGen:
    @staticmethod
    def powerset(iterable, r):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, r + 1))

    @staticmethod
    def gen(
            number_of_containers,
            number_of_platforms,
            number_of_preferences,
            number_of_batches,
            share=0.05
    ):
        f_subsets = dict()
        powerset = list(RandomPCInstanceGen.powerset(range(number_of_containers), 3))
        unique_id = set()
        for i in range(number_of_platforms):
            f_subsets[i] = random.sample(range(len(powerset)), random.randint(0, int(share * len(powerset))))
            unique_id.update(f_subsets[i])

        subsets = [set(powerset[i]) for i in unique_id]
        m = {j: i for i, j in enumerate(unique_id)}
        for i in range(number_of_platforms):
            f_subsets[i] = [m[j] for j in f_subsets[i]]

        containers = set(range(number_of_containers))
        container_to_priority = {i: random.randint(0, number_of_preferences - 1) for i in containers}

        container_to_batch = {i: random.randint(0, number_of_batches - 1) for i in containers}

        priority_to_containers = defaultdict(set)
        for c, p in container_to_priority.items():
            priority_to_containers[p].add(c)

        batch_to_containers = defaultdict(set)
        for c, b in container_to_batch.items():
            batch_to_containers[b].add(c)

        platform_to_priority = {i: random.randint(0, 1) for i in range(number_of_platforms)}

        return CPInstance(
            subsets,
            f_subsets,
            containers,
            container_to_priority,
            container_to_batch,
            priority_to_containers,
            batch_to_containers,
            platform_to_priority
        )


def test():
    inst = RandomPCInstanceGen.gen(20, 3, 2, 3)
    print(inst)


    # solver = CPSolver()
    # solver.solve(inst)
    # print("total_container: ", solver.total_container(inst))
    # print("containers_by_batch: ", solver.containers_by_batch(inst))
    # print("containers_by_priority: ", solver.containers_by_priority(inst))
    # print("platforms_by_priority: ", solver.platforms_by_priority(inst))
    # print("platforms_by_load: ", solver.platforms_by_load(inst))


test()


# CPInstance(
#   subsets=[
#       {0}, {5}, {0, 4}, {0, 5}, {0, 6}, {1, 2}, {1, 3}, {1, 5}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {3, 4}, {4, 5},
#       {4, 6}, {0, 1, 4}, {0, 1, 6}, {0, 3, 5}, {0, 4, 5}, {0, 5, 6}, {1, 2, 3}, {1, 2, 5}, {1, 3, 6}, {1, 4, 5},
#       {1, 4, 6}, {2, 3, 6}, {4, 5, 6}
#   ],
#   f_subsets={
#       0: [19, 20, 3, 8, 4, 26, 10, 16, 24, 17, 9, 15, 14, 2, 18, 1],
#       1: [13, 7, 16, 26, 5],
#       2: [14, 26, 1, 22, 25, 20, 11, 8, 0, 23, 21, 17, 12, 6]
#   },
#   containers={0, 1, 2, 3, 4, 5, 6},
#   container_to_priority={0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1},
#   container_to_batch={0: 1, 1: 0, 2: 2, 3: 1, 4: 1, 5: 0, 6: 1},
#   priority_to_containers={0: {0, 1, 2, 4}, 1: {3, 5, 6}},
#   batch_to_containers={1: {0, 3, 4, 6}, 0: {1, 5}, 2: {2}},
#   platform_to_priority={0: 0, 1: 1, 2: 1}
# )
#
# total_container:  [0, 4, 5, 1, 2, 3]
# containers_by_batch:  {1: {0, 3, 4}, 0: {1, 5}, 2: {2}}
# containers_by_priority:  {0: {0, 1, 2, 4}, 1: {3, 5}}
# platforms_by_priority:  {0: {0}, 1: {2}}
# platforms_by_load:  {3: 2}
#
# как видно, здесь не отправлен 6-й контейнер из 1-й партии, но отправлен 2-й из 2-й,
# это потому что не получилось нагенерить данные так, чтобы любая расстановка была такой, что все контейнеры в
# ней были из одной партии
#