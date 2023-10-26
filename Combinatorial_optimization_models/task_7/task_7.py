import itertools
import logging
from dataclasses import dataclass

import coptpy
import numpy as np
import typing as tp


@dataclass
class CinemaInstance:
    number_of_movies: int
    number_of_screens: int
    hor_plan: int
    td_m: np.array
    p_m_t: np.array
    tech_s: np.array
    cap_s: np.array
    price_s: np.array
    pi_s: tp.List[np.array]
    t_s: tp.List[np.array]
    delta_t_1: np.array


class CinemaInstanceGen:
    @staticmethod
    def _generate_delta_t_1(hor_plan):
        n = 10
        a = [
            [i, min(i + np.random.randint(1, 3), hor_plan)]
            for i in np.random.randint(1, hor_plan + 1, n)
        ]
        a = list(sorted(a, key=lambda x: x[0]))
        delta_t_1 = list()

        i = 0
        while i < n:
            j = i
            cur_j = a[i][1]
            while j < n and a[j][0] == a[i][0]:
                cur_j = max(cur_j, a[j][1])
                j += 1
            if delta_t_1 and delta_t_1[-1][1] >= a[i][0]:
                delta_t_1[-1][1] = cur_j
            elif a[i][0] != cur_j:
                delta_t_1.append([a[i][0], cur_j])
            i = j

        return delta_t_1

    @staticmethod
    def generate(number_of_movies, number_of_screens, hor_plan):
        td_m = np.random.randint(1, 6, number_of_movies)
        p_m_t = np.random.randint(1, 6, (number_of_movies, hor_plan))

        tech_s = np.full(number_of_screens, 1)
        cap_s = np.random.randint(50, 251, number_of_screens)
        price_s = np.random.randint(300, 1001, number_of_screens)
        pi_s = [
            np.random.choice(list(range(number_of_movies)), np.random.randint(6, 10))
            for _ in range(number_of_screens)
        ]
        t_s = [
            np.arange(np.random.randint(1, 5), hor_plan + np.random.randint(-5, -1))
            for _ in range(number_of_screens)
        ]
        delta_t_1 = np.array(CinemaInstanceGen._generate_delta_t_1(hor_plan))

        return CinemaInstance(
            number_of_movies,
            number_of_screens,
            hor_plan,
            td_m,
            p_m_t,
            tech_s,
            cap_s,
            price_s,
            pi_s,
            t_s,
            delta_t_1
        )


class CinemaSolver:
    def __init__(self):
        self._envr = coptpy.Envr()
        self._model: coptpy.Model = self._envr.createModel(name="CinemaProblem")
        self._model.setParam(coptpy.COPT.Param.Logging, 0)

        self._logger = logging.getLogger("CinemaSolver")

        self.x = None

    def solve(self, inst: CinemaInstance):
        self._create_model(inst)
        self._model.solve()

        if self._model.status != coptpy.COPT.OPTIMAL:
            self._logger.info("Problem is infeasible.")
            return None
        else:
            self.x = list()
            for s in range(inst.number_of_screens):
                for i in range(len(inst.pi_s[s])):
                    for t in self._t_s(inst, s, i):
                        if bool(self._model.getVarByName(f"x({s},{i},{t})").getInfo("value")):
                            self.x.append((s, i, t))

            return self.x

    def _create_x(self, inst):
        temp = list()
        for s in range(inst.number_of_screens):
            for i in range(len(inst.pi_s[s])):
                for t in self._t_s(inst, s, i):
                    temp.append((s, i, t))
        return self._model.addVars(coptpy.tuplelist(temp), vtype=coptpy.COPT.BINARY, nameprefix="x")

    def _create_y(self, inst):
        temp = list()
        for s in range(inst.number_of_screens):
            for i in range(len(inst.pi_s[s])):
                temp.append((s, i))
        return self._model.addVars(coptpy.tuplelist(temp), vtype=coptpy.COPT.BINARY, nameprefix="y")

    @staticmethod
    def _t_s(inst, s, i):
        return np.array([j for j in inst.t_s[s] if j + inst.tech_s[s] + inst.td_m[inst.pi_s[s][i]] <= 24])

    def _create_model(self, inst):
        screens = range(inst.number_of_screens)

        x = self._create_x(inst)
        self.x = x

        y = self._create_y(inst)

        self._model.setObjective(
            sum(inst.price_s[s] * min(inst.cap_s[s], inst.p_m_t[inst.pi_s[s][i], t]) * x[s, i, t]
                for s in screens
                for i in range(len(inst.pi_s[s]))
                for t in self._t_s(inst, s, i)),
            coptpy.COPT.MAXIMIZE
        )

        self._model.addConstrs(
            sum(x[s, i, t] for t in self._t_s(inst, s, i)) + y[s, i] == 1
            for s in screens
            for i in range(len(inst.pi_s[s]))
        )

        self._model.addConstrs(
            sum((t + inst.td_m[inst.pi_s[s][i]] + inst.tech_s[s]) * x[s, i, t] for t in self._t_s(inst, s, i)) <=
            sum(t * x[s, j, t] for t in self._t_s(inst, s, j)) + 100 * inst.hor_plan * y[s, j]
            for s in screens
            for i in range(len(inst.pi_s[s]) - 1)
            for j in range(i + 1, len(inst.pi_s[s]))
        )

        for delta in inst.delta_t_1:
            temp = [
                    x[s, i, t]
                    for s in screens
                    for i in range(len(inst.pi_s[s]))
                    for t in self._t_s(inst, s, i) if delta[0] <= t <= delta[1]

            ]
            if temp:
                self._model.addConstr(
                     sum(temp) <= 1
                )


if __name__ == "__main__":
    logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S',
                        level=logging.DEBUG)


    def test():
        inst = CinemaInstanceGen.generate(5, 2, 24)
        print(inst)
        solver = CinemaSolver()
        x = solver.solve(inst)
        print(x)


    test()

    # CinemaInstance(
    #   number_of_movies=5,
    #   number_of_screens=2,
    #   hor_plan=24,
    #   td_m=array([3, 4, 5, 4, 5]),
    #   p_m_t=array([
    #       [3, 4, 2, 4, 2, 3, 2, 3, 3, 2, 1, 2, 1, 5, 1, 3, 1, 3, 4, 1, 3, 4, 3, 4],
    #       [3, 5, 2, 3, 5, 2, 4, 1, 5, 4, 3, 2, 3, 4, 1, 2, 3, 5, 2, 1, 1, 3, 5, 2],
    #       [2, 4, 2, 5, 3, 5, 2, 5, 4, 3, 3, 1, 4, 2, 1, 4, 2, 5, 3, 4, 4, 5, 3, 4],
    #       [5, 4, 5, 5, 3, 1, 3, 1, 1, 5, 3, 4, 3, 4, 5, 5, 1, 3, 1, 4, 3, 2, 3, 5],
    #       [1, 5, 3, 5, 3, 4, 2, 5, 3, 5, 1, 3, 5, 1, 4, 3, 1, 1, 2, 3, 3, 4, 3, 3]
    #   ]),
    #   tech_s=array([1, 1]),
    #   cap_s=array([ 60, 227]),
    #   price_s=array([319, 359]),
    #   pi_s=[
    #       array([4, 1, 2, 0, 1, 3]),
    #       array([1, 4, 0, 3, 1, 3, 0, 3])
    #   ],
    #   t_s=[
    #       array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    #       array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    #   ],
    #   delta_t_1=array([[ 3,  4], [ 8, 14], [17, 22]])
    # )
    #
    # Outputs:
    #   [(0, 0, 3), (0, 1, 9), (0, 5, 15), (1, 0, 2), (1, 1, 7), (1, 3, 15), (1, 6, 20)]
