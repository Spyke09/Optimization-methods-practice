import logging

from hw_potok.src import flow_finder
from hw_potok.src import network_graph
from hw_potok.src import algo

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.DEBUG)


logger = logging.getLogger("TestMinCostFlowFinder")


def test_1():
    cap = {
        (0, 1): 15,
        (0, 4): 10,
        (1, 2): 15,
        (1, 5): 5,
        (2, 3): 15,
        (3, 6): 10,
        (3, 7): 10,
        (4, 5): 8,
        (4, 6): 10,
        (5, 3): 4,
        (5, 6): 8,
        (6, 7): 15
    }

    cost = {
        (0, 1): 0,
        (0, 4): 0,
        (1, 2): 5,
        (1, 5): -15,
        (2, 3): 5,
        (3, 6): 5,
        (3, 7): 0,
        (4, 5): 0,
        (4, 6): 6,
        (5, 3): 3,
        (5, 6): 5,
        (6, 7): 0
    }

    my_mew_network = network_graph.SimpleNetwork(cap, 0, 7, cost)
    finder = flow_finder.MinCostFlow()
    for i in (flow_finder.EdmondsKarp(), flow_finder.Dinica(), flow_finder.GoldbergT()):
        my_mew_network.clear()
        finder.find(my_mew_network, i)
        assert my_mew_network.get_total_cost() == 107
        # visualization.draw_network(my_mew_network, [0, 1, 2, 4, 1, 3, 4, 5])


def test_2():
    cap = {
        (0, 1): 1,
        (1, 2): 1,
        (2, 3): 1
    }

    cost = {
        (0, 1): -5,
        (1, 2): 5,
        (2, 3): 0
    }

    my_mew_network = network_graph.SimpleNetwork(cap, 0, 2, cost)
    algo.NegativeCycleFinder.find(my_mew_network)


test_1()
# test_2()
