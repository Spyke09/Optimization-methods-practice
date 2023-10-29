import dataclasses
import logging
from dataclasses import dataclass

import coptpy
import numpy as np


@dataclass
class CehInstance:
    """
    Класс для хранения данных об экземпляре
    """
    n_products: int
    n_machines: int
    n_time: int
    m: np.array
    tech: np.array
    price: np.array
    d: np.array
    tau: np.array
    tc: np.array
    first: np.array = dataclasses.field(init=False)
    next: np.array = dataclasses.field(init=False)
    t_a: np.array = dataclasses.field(init=False)
    t_b: np.array = dataclasses.field(init=False)

    def big_t(self, p, s):
        lb = int(self.t_a[p, s] + 1)
        ub = int(max(self.tau[p] - self.tc[p, s] - self.t_b[p, s], lb))
        return np.arange(lb, ub)

    def big_u(self, s, t):
        res = list()
        for p in range(self.n_products):
            for t_ in range(self.n_time):
                if t_ <= t <= t_ + self.tc[p, s]:
                    res.append((p, t_))
        return res

    def __post_init__(self):
        self.first = np.full(self.n_products, -1)
        self.next = np.full((self.n_products, self.n_machines), -1)
        for p in range(self.n_products):
            a = [i[0] for i in sorted(filter((lambda x: x[1] > 0), enumerate(self.tech[p])), key=lambda x: x[1])]
            for i in range(len(a) - 1):
                self.next[p, a[i]] = a[i + 1]
            self.first[p] = a[0]

        self.t_a = np.full((self.n_products, self.n_machines), 0.0)
        self.t_b = np.full((self.n_products, self.n_machines), 0.0)

        for p in range(self.n_products):
            cur = self.first[p]
            prev_value = 0
            while cur != -1:
                self.t_a[p, cur] = prev_value
                prev_value += self.tc[p, cur]
                cur = self.next[p, cur]

            for m in range(self.n_machines):
                self.t_b[p, m] = prev_value - self.t_a[p, m] - self.tc[p, m]

    def x_iter(self):
        for p in range(self.n_products):
            for s in range(self.n_machines):
                for t in range(self.n_time):
                    yield p, s, t


class CehSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._model: coptpy.Model = self._envr.createModel(name="CehProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, 0)

        self._logger = logging.getLogger("CehSolver")

        self.x = None

    def solve(self, inst: CehInstance):
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
            for p, s, t in inst.x_iter():
                if bool(self._model.getVarByName(f"x({p},{s},{t})").getInfo("value")):
                    self.x.append((p, s, t))

            return self.x

    def _create_model(self, inst):
        self.x = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.INTEGER, nameprefix="x")
        self.y = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.BINARY, nameprefix="y")
        # self.h = self._model.addVars(range(inst.n_products), vtype=coptpy.COPT.BINARY, nameprefix="h")

        # целевая функция
        self._model.setObjective(
            sum(inst.price[p] * sum(self.x[p, inst.first[p], t] for t in inst.big_t(p, inst.first[p]))
                for p in range(inst.n_products)) -

            sum((t + 1) * self.x[p, inst.first[p], t]
                for p in range(inst.n_products)
                for t in range(inst.n_time)),
            coptpy.COPT.MAXIMIZE
        )

        self._model.addConstrs(
            self.x[p, s, t] <= inst.m[p, s, t] * self.y[p, s, t]
            for p in range(inst.n_products)
            for s in range(inst.n_machines)
            for t in inst.big_t(p, s)
        )

        self._model.addConstrs(
            self.x[p, s1, t] == self.x[p, inst.next[p, s1], t + inst.tc[p, s1]]
            for p in range(inst.n_products)
            for s1 in range(inst.n_machines)
            for t in inst.big_t(p, s1) if inst.next[p, s1] != -1
        )

        self._model.addConstrs(
            sum(self.y[p, s, t_] for p, t_ in inst.big_u(s, t)) <= 1
            for s in range(inst.n_machines)
            for t in range(
                min(inst.big_t(p, s)[0] for p in range(inst.n_products)),
                max(inst.big_t(p, s)[-1] for p in range(inst.n_products)) + 1
            )
        )

        self._model.addConstrs(self.x[p, s, t] >= 0 for p, s, t in inst.x_iter())

        self._model.addConstrs(
            sum(self.x[p, inst.first[p], t] for t in inst.big_t(p, inst.first[p])) <= inst.d[p]
            for p in range(inst.n_products)
        )


if __name__ == "__main__":
    logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S',
                        level=logging.DEBUG)

    def test1():
        n_p, n_m, n_t = 2, 5, 12
        temp = [50, 50, 50, 50, 100]
        inst = CehInstance(
            n_p, n_m, n_t,
            np.array([[[temp[m] for _ in range(n_t)] for m in range(n_m)] for _ in range(n_p)]),
            np.array([[1, 0, 2, 3, 4], [0, 1, 2, 3, 4]]),
            np.array([150, 150]),
            np.array([100, 100]),
            np.array([13, 13]),
            np.array([[2, 0, 2, 1, 1], [0, 2, 0, 1, 1]])
        )

        solver = CehSolver()
        x = solver.solve(inst)
        print(x)


    def test2():
        n_p, n_m, n_t = 2, 5, 8
        temp = [50, 50, 50, 50, 100]
        inst = CehInstance(
            n_p, n_m, n_t,
            np.array([[[temp[m] for _ in range(n_t)] for m in range(n_m)] for _ in range(n_p)]),
            np.array([[1, 0, 3, 4, 2], [0, 1, 0, 2, 3]]),
            np.array([150, 120]),
            np.array([100, 30]),
            np.array([9, 9]),
            np.array([[2, 0, 2, 1, 1], [0, 2, 0, 1, 1]])
        )

        solver = CehSolver()
        x = solver.solve(inst)
        print(x)


    def test3():
        n_p, n_m, n_t = 4, 7, 24
        temp = [50, 50, 50, 50, 50, 50, 100]
        inst = CehInstance(
            n_p, n_m, n_t,
            np.array([[[temp[m] for _ in range(n_t)] for m in range(n_m)] for _ in range(n_p)]),
            np.array([[1, 0, 3, 4, 2, 5, 0], [0, 1, 2, 4, 3, 5, 0], [0, 0, 0, 2, 3, 0, 1], [0, 5, 4, 2, 1, 3, 0]]),
            np.array([150, 150, 80, 120]),
            np.array([100, 100, 30, 80]),
            np.array([25, 25, 25, 25]),
            np.array([[2, 0, 2, 1, 1, 1, 0], [0, 1, 1, 1, 1, 3, 0], [0, 0, 0, 1, 1, 0, 2], [0, 1, 2, 1, 1, 1, 0]])
        )

        solver = CehSolver()
        x = solver.solve(inst)
        print(x)

    test1()
    test2()
    test3()
