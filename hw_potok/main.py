import main_algo
import network_graph


def test1():
    my_mew_network = network_graph.SimpleNetwork()
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
    ]
    capacities = {
        (0, 1): 5,
        (0, 2): 2,
        (1, 3): 3,
        (2, 3): 8
    }
    my_mew_network.setup(edges, capacities, 0, 3)

    main_algo.MaximumFlowFinder().find(my_mew_network)

    print(f"Answer is: {my_mew_network.get_network_flow()}")
    print("Network graph:\n", str(my_mew_network), "\n", sep="")


def test2():
    my_mew_network = network_graph.SimpleNetwork()
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (1, 2),
    ]
    capacities = {
        (0, 1): 10,
        (0, 2): 5,
        (1, 3): 5,
        (2, 3): 10,
        (1, 2): 15,
    }
    my_mew_network.setup(edges, capacities, 0, 3)

    main_algo.MaximumFlowFinder().find(my_mew_network)

    print(f"Answer is: {my_mew_network.get_network_flow()}")
    print("Network graph:\n", str(my_mew_network), "\n", sep="")


def test3():
    my_mew_network = network_graph.SimpleNetwork()
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        (1, 2),
        (3, 4),
    ]
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
    my_mew_network.setup(edges, capacities, 0, 5)

    main_algo.MaximumFlowFinder().find(my_mew_network)

    print(f"Answer is: {my_mew_network.get_network_flow()}")
    print("Network graph:\n", str(my_mew_network), "\n", sep="")


if __name__ == "__main__":
    test1()
    test2()
    test3()
