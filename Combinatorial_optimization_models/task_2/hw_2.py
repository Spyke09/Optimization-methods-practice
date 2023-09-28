import dataclasses
import itertools
import logging

import coptpy
import numpy as np

COPT_LOG_LEVEL = 0


@dataclasses.dataclass
class AssignmentsProblem:
    _n: int  # N = {1, ..., _n}
    _m: int  # M = {0, ..., _m+1}
    _ad_time: np.array  # (a_i, d_i) for each i from N
    _w: np.array  # w_k_l for each i, j from M\{0}
    _f: np.array  # f_i_j for each i, j from N+{0}
    _alpha: float = 1.

    _envr: coptpy.Envr = dataclasses.field(init=False)
    _logger: logging.Logger = dataclasses.field(init=False)
    _model: coptpy.Model = dataclasses.field(init=False)

    def __post_init__(self):
        if not (0 <= self._alpha <= 1):
            raise ValueError("Param alpha should lie in [0, 1].")

        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("AssignmentsProblem")
        self._model = self._envr.createModel(name="AssignmentsProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, COPT_LOG_LEVEL)

        self._create_model()

    def _create_model(self):
        n, m = self._n, self._m
        ad = self._ad_time
        m_range = range(1, m + 2)  # sequence 1...m+1
        n_range = range(1, n + 1)  # sequence 1...n

        # adding x и y to model
        # using itertools.product for cartesian product of lists
        x = self._model.addVars(
            coptpy.tuplelist(itertools.product(n_range, m_range)),
            vtype=coptpy.COPT.BINARY,
            nameprefix="x"
        )

        y = self._model.addVars(
            coptpy.tuplelist(filter(
                lambda idx: idx[0] < idx[2],
                itertools.product(n_range, m_range, n_range, m_range)
            )),
            vtype=coptpy.COPT.BINARY,
            nameprefix="y"
        )

        # objective = alpha * objective1 + (1 - alpha) * objective2
        self._model.setObjective(
            sum(x[i, m + 1] for i in n_range) * self._alpha +
            (sum(self._f[i, j] * self._w[k, l] * y[i, k, j, l]
                 for i in n_range
                 for j in range(i + 1, n + 1)
                 for k in m_range
                 for l in m_range) +
             sum((self._f[0, i] * self._w[0, k] - self._f[i, 0] * self._w[k, 0]) * x[i, k]
                 for i in n_range
                 for k in m_range)) * (1 - self._alpha),
            coptpy.COPT.MINIMIZE
        )

        # flight assignment to exactly one gate
        self._model.addConstrs(
            sum(x[i, k] for k in m_range) == 1 for i in n_range
        )

        # ban overlapping flights within the gate
        for i in n_range:
            for j in n_range:
                if i > j and (
                        (ad[j - 1][1] >= ad[i - 1][1] > ad[j - 1][0]) or
                        (ad[i - 1][1] > ad[j - 1][0] >= ad[i - 1][0]) or
                        (ad[i - 1][1] >= ad[j - 1][1] > ad[i - 1][0]) or
                        (ad[j - 1][1] > ad[i - 1][0] >= ad[j - 1][0])):
                    self._model.addConstrs(
                        x[i, k] + x[j, k] <= 1
                        for k in m_range
                    )

    def solve(self):
        self._model.solve()

        if self._model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Problem is infeasible.")
        else:
            x = dict()
            for i in range(1, self._n + 1):
                for j in range(1, self._m + 2):
                    x[(i, j)] = self._model.getVarByName(f"x({i},{j})").getInfo("value")
            return x


def test1():
    ap = AssignmentsProblem(
        10,
        3,
        np.array([[76, 78], [42, 47], [42, 44],
                  [42, 44], [49, 53], [67, 72],
                  [26, 34], [48, 49], [27, 27], [76, 80]]),
        np.array([[0., 19., 19., 60., 45.], [19., 0., 6., 41., 30.], [19., 6., 0., 41., 36.],
                  [60., 41., 41., 0., 15.], [45., 30., 36., 15., 0.]]),
        np.array([[0, 15, 60, 68, 46, 28, 3, 59, 42, 25, 94],
                  [15, 0, 43, 1, 67, 78, 20, 20, 32, 30, 41],
                  [60, 43, 0, 63, 64, 7, 26, 43, 47, 13, 58],
                  [68, 1, 63, 0, 20, 72, 7, 36, 15, 28, 41],
                  [46, 67, 64, 20, 0, 87, 76, 72, 100, 27, 32],
                  [28, 78, 7, 72, 87, 0, 37, 49, 83, 56, 55],
                  [3, 20, 26, 7, 76, 37, 0, 39, 25, 16, 2],
                  [59, 20, 43, 36, 72, 49, 39, 0, 47, 59, 39],
                  [42, 32, 47, 15, 100, 83, 25, 47, 0, 80, 43],
                  [25, 30, 13, 28, 27, 56, 16, 59, 80, 0, 58],
                  [94, 41, 58, 41, 32, 55, 2, 39, 43, 58, 0]]),
        0.5
    )

    x = [f"{j[0][0]} -> {j[0][1]}" for j in filter(lambda i: i[1] == 1, ap.solve().items())]
    print(f"x = {x}")
    # x = ['1 -> 1', '2 -> 2', '3 -> 1', '4 -> 3', '5 -> 1', '6 -> 1', '7 -> 1', '8 -> 1', '9 -> 2', '10 -> 2']


def test2():
    ap = AssignmentsProblem(
        3,
        2,
        np.array([[2, 10], [9, 10], [9, 10]]),
        np.array([[0., 19., 19., 60.], [19., 0., 6., 41.], [19., 6., 0., 41.],
                  [60., 41., 41., 0.]]),
        np.array([[0, 15, 60, 68],
                  [15, 0, 43, 1],
                  [60, 43, 0, 63],
                  [68, 1, 63, 0]]),
        0.5
    )

    x = [f"{j[0][0]} -> {j[0][1]}" for j in filter(lambda i: i[1] == 1, ap.solve().items())]
    print(f"x = {x}")


test1()
test2()
