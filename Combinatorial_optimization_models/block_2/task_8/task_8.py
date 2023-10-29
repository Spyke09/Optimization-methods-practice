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
        lb = self.t_a[p, s] + 1
        ub = max(self.tau[p] - self.tc[p, s] - self.t_b[p, s], lb)
        return np.arange(lb, ub)

    def big_u(self, s, t):
        res = list()
        for p in range(self.n_products):
            for t_ in range(self.n_time):
                if t_ <= t <= t_ + self.tc[p, s]:
                    res.append((p, s, t_))

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
        x = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.BINARY, nameprefix="x")
        y = self._model.addVars(coptpy.tuplelist(inst.x_iter()), vtype=coptpy.COPT.BINARY, nameprefix="y")
        # h = self._model.addVars(range(inst.n_products), vtype=coptpy.COPT.BINARY, nameprefix="h")

        # целевая функция
        self._model.setObjective(
            sum(inst.price[p] * sum(x[p, inst.first[p], t] for t in inst.big_t(p, inst.first(p)))
                for p in range(inst.n_products)) -

            sum((t + 1) * x[p, inst.first[p], t]
                for p in range(inst.n_products)
                for t in range(inst.n_time)),
            coptpy.COPT.MAXIMIZE
        )

        self._model.addConstrs(for p in range(inst.n_products) for s in range())


if __name__ == "__main__":
    logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S',
                        level=logging.DEBUG)


    def test():
        n_p, n_m, n_t = 2, 5, 8
        inst = CehInstance(
            n_p, n_m, n_t,
            np.array([[50, 50, 50, 50, 100] for _ in range(n_t)]),
            np.array([[1, 0, 3, 4, 2], [0, 1, 0, 2, 3]]),
            np.array([150, 120]),
            np.array([100, 30]),
            np.array([9, 9]),
            np.array([[2, 0, 2, 1, 1], [0, 2, 0, 1, 1]])
        )
        print(inst)

        for p in range(inst.n_products):
            for s in range(inst.n_machines):
                print(p, s, inst.big_t(p, s))
        # solver = CehSolver()
        # x = solver.solve(inst)
        # print(x)


    test()
