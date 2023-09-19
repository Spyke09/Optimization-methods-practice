import coptpy
import typing as tp
import numpy as np
import dataclasses

import logging
import itertools


COPT_LOG_LEVEL = 0


@dataclasses.dataclass
class AssignmentsProblem:
    _n: int           # N = {1, ..., N}
    _m: int           # M = {0, ..., M+1}
    _ad_time: np.array  # (a_i, d_i) for each i from N
    _w: np.array        # w_k_l for each i, j from M\{0}
    _f: np.array        # f_i_j for each i, j from N+{0}

    _envr: coptpy.Envr = dataclasses.field(init=False)
    _logger: logging.Logger = dataclasses.field(init=False)
    _model: coptpy.Model = dataclasses.field(init=False)

    def __post_init__(self):
        self._envr = coptpy.Envr()
        self._logger = logging.getLogger("UniqueSolutionSolver")
        self._model = self._envr.createModel(name="AssignmentsProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, COPT_LOG_LEVEL)

    def create_model(self, op_cr=0):
        n, m = self._n, self._m
        ad = self._ad_time
        m_range = range(1, m + 2)
        n_range = range(1, n + 1)

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

        if op_cr == 0:
            self._model.setObjective(
                sum(x[i, m + 1] for i in n_range),
                coptpy.COPT.MINIMIZE
            )
        else:
            self._model.setObjective(
                sum(self._f[i, j] * self._w[k, l] * y[i, k, j, l]
                    for i in n_range
                    for j in range(i + 1, n + 1)
                    for k in m_range
                    for l in m_range) +
                sum((self._f[0, i] * self._w[0, k] - self._f[i, 0] * self._w[k, 0]) * x[i, k]
                    for i in n_range
                    for k in m_range),
                coptpy.COPT.MINIMIZE
            )

        self._model.addConstrs(
            sum(x[i, k] for k in m_range) for i in n_range
        )

        for i in n_range:
            for j in n_range:
                if i < j and ((ad[j][1] > ad[i][1] > ad[j][0]) or (ad[i][1] > ad[j][1] > ad[i][0])):
                    self._model.addConstr(
                        x[i, k] * x[j, k] <= 1
                        for k in m_range
                    )
