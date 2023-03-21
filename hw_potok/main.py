import flow_finder
import network_graph
import visualization


def test(instance, st, fn, order, viz_q=False):
    my_mew_network = network_graph.SimpleNetwork(instance, st, fn)
    my_mew_network2 = network_graph.SimpleNetwork(instance, st, fn)
    ed_k = flow_finder.EdmondsKarp()
    dinica = flow_finder.Dinica()

    ed_k.find(my_mew_network)
    dinica.find(my_mew_network2)

    print(f"Answer is: {my_mew_network.get_network_flow()}")
    print("Network graph:\n", str(my_mew_network), "\n", sep="")
    print("Check conservation law: ", my_mew_network.check_conservation_law())
    assert my_mew_network.check_conservation_law()
    assert my_mew_network2.check_conservation_law()
    assert my_mew_network.get_network_flow() == my_mew_network2.get_network_flow()
    if viz_q:
        visualization.draw_network(my_mew_network, order)


def test1():
    print("Test 1")
    capacities = {
        (0, 1): 5,
        (0, 2): 2,
        (1, 3): 3,
        (2, 3): 8
    }
    test(capacities, 0, 3, [0, 1, 1, 2])


def test2():
    print("Test 2")
    capacities = {
        (0, 1): 10,
        (0, 2): 5,
        (1, 3): 5,
        (2, 3): 10,
        (1, 2): 15,
    }
    test(capacities, 0, 3, [0, 1, 1, 3])


def test3():
    print("Test 3")
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
    print("Test 4")
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
    print("Test 5")
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
    print("Test 5")
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


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
