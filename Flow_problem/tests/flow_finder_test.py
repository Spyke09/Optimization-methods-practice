import logging

from hw_potok.src import flow_finder
from hw_potok.src import network_graph
from hw_potok.src import visualization

logging.basicConfig(format='[%(name)s]: %(message)s', datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.DEBUG)


logger = logging.getLogger("TestFlowFinder")


def test(instance, st, fn, order, viz_q=False):
    my_mew_network = network_graph.SimpleNetwork(instance, st, fn)

    def test_(finder, net):
        finder.find(net)
        assert net.check_conservation_law()
        answer = net.get_network_flow()
        net.clear()
        return answer

    answer1 = test_(flow_finder.EdmondsKarp(), my_mew_network)
    answer2 = test_(flow_finder.Dinica(), my_mew_network)
    answer3 = test_(flow_finder.GoldbergT(), my_mew_network)
    assert answer1 == answer3 and answer3 == answer2

    if viz_q:
        flow_finder.GoldbergT().find(my_mew_network)
        visualization.draw_network(my_mew_network, order)


def test1():
    logger.info("Test 1")
    capacities = {
        (0, 1): 5,
        (0, 2): 2,
        (1, 3): 3,
        (2, 3): 8
    }
    test(capacities, 0, 3, [0, 1, 1, 2])


def test2():
    logger.info("Test 2")
    capacities = {
        (0, 1): 10,
        (0, 2): 5,
        (1, 3): 5,
        (2, 3): 10,
        (1, 2): 15,
    }
    test(capacities, 0, 3, [0, 1, 1, 3])


def test3():
    logger.info("Test 3")
    capacities = {
        (0, 1): 3,
        (0, 2): 3,
        (1, 3): 3,
        (2, 4): 2,
        (3, 5): 2,
        (4, 5): 3,
        (1, 2): 2,
        (3, 4): 4,
    }
    test(capacities, 0, 5, [0, 1, 1, 2, 2, 3])


def test4():
    logger.info("Test 4")
    capacities = {
        (0, 1): 10,
        (0, 2): 10,
        (1, 2): 1,
        (1, 3): 8,
        (1, 5): 6,
        (2, 0): 0,
        (2, 1): 1,
        (2, 4): 12,
        (2, 5): 4,
        (3, 1): 0,
        (3, 5): 3,
        (3, 6): 7,
        (4, 2): 0,
        (4, 5): 2,
        (4, 6): 8,
        (5, 1): 0,
        (5, 2): 4,
        (5, 3): 3,
        (5, 4): 2,
        (5, 6): 2,
        (6, 3): 0,
        (6, 4): 0,
        (6, 5): 0,
    }
    test(capacities, 0, 6, [0, 1, 1, 2, 2, 2, 4])


def test5():
    logger.info("Test 5")
    capacities = {
        (0, 1): 4,
        (0, 2): 5,
        (1, 2): 6,
        (1, 3): 2,
        (1, 4): 5,
        (2, 3): 3,
        (2, 4): 2,
        (3, 4): 3,
        (3, 5): 3,
        (4, 5): 3,
    }
    test(capacities, 0, 5, [0, 1, 1, 2, 2, 3])


def test6():
    logger.info("Test 5")
    capacities = {
        (0, 3): 1,
        (0, 1): 2,
        (1, 2): 2,
        (2, 4): 2,
        (4, 7): 1,
        (3, 4): 1,
        (4, 3): 1,
        (3, 5): 2,
        (5, 6): 2,
        (6, 7): 2,
    }
    test(capacities, 0, 7, [0, 1, 2, 1, 2, 2, 4, 5])


test1()
test2()
test3()
test4()
test5()
test6()
