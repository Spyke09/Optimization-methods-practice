import typing as tp
from enum import Enum

import inetwork

NodeId = int
FlowValue = float
EdgeId = tp.Tuple[NodeId, NodeId]


class EdgeType(Enum):
    NORMAL = 0,
    INVERTED = 1


def reverse_edge(edge_id: EdgeId) -> EdgeId:
    return edge_id[1], edge_id[0]


class SimpleNetwork(inetwork.INetwork):
    def __init__(self):
        self.__fan_in_nodes: tp.List[tp.List[NodeId]] = list()
        self.__fan_out_nodes: tp.List[tp.List[NodeId]] = list()
        self.__capacities: tp.List[tp.List[FlowValue]] = list()
        self.__flow: tp.List[tp.List[FlowValue]] = list()
        self.__source: NodeId = -1
        self.__sink: NodeId = -1
        self.__nodes_number = 0
        self.__edges_number = 0

    def __repr__(self):
        res = ""
        for i in range(self.__nodes_number):
            for j in range(self.__nodes_number):
                if self.__capacities[i][j] > 0:
                    res += f"{(i, j)}: {self.__flow[i][j]}/{self.__capacities[i][j]} "
                else:
                    res += len(f"{(i, j)}: {self.__flow[i][j]}/{self.__capacities[i][j]}") * "_" + " "
            res += "\n"

        return res

    def get_flow(self):
        return self.__flow

    def edge_exist_q(self, edge_id: EdgeId, edge_type: EdgeType = EdgeType.NORMAL) -> bool:
        return self.__capacities[edge_id[0]][edge_id[1]] > 0

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_in_nodes[node_id]

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_out_nodes[node_id]

    def get_edge_capacity(self, edge_id: EdgeId) -> FlowValue:
        return self.__capacities[edge_id[0]][edge_id[1]]

    def get_edge_flow(self, edge_id) -> FlowValue:
        return self.__flow[edge_id[0]][edge_id[1]]

    def set_edge_flow(self, edge_id: EdgeId, value: FlowValue) -> None:
        self.__flow[edge_id[0]][edge_id[1]] = value

    def check_conservation_law(self) -> bool:
        for i in range(self.__nodes_number):
            if i == self.__sink or i == self.__source:
                continue
            if self.get_excess_flow(i) != 0:
                return False
        return True

    def get_sink(self) -> NodeId:
        return self.__sink

    def get_source(self) -> NodeId:
        return self.__source

    def get_excess_flow(self, node_id: NodeId) -> FlowValue:
        sum_in = 0
        for j in self.get_node_fan_in(node_id):
            sum_in += self.__flow[j][node_id]

        sum_out = 0
        for j in self.get_node_fan_out(node_id):
            sum_out += self.__flow[node_id][j]

        return sum_in - sum_out

    def get_network_flow(self) -> FlowValue:
        return -self.get_excess_flow(self.__source)

    def setup(
            self,
            capacities: tp.Dict[tp.Tuple[int, int], FlowValue],
            source: NodeId,
            sink: NodeId
    ) -> None:
        n = max([max(i) for i in capacities.keys()]) + 1
        self.__source = source
        self.__sink = sink
        self.__edges_number = n
        self.__capacities = [[0.0 for _ in range(n)] for _ in range(n)]
        self.__flow = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if (i, j) in capacities:
                    self.__capacities[i][j] = capacities[(i, j)]
                self.__flow[i][j] = 0

        node_set = set()
        for i, j in capacities.keys():
            node_set.add(i)
            node_set.add(j)
        self.__nodes_number = len(node_set)

        self.__fan_in_nodes = [[] for _ in range(self.__nodes_number)]
        self.__fan_out_nodes = [[] for _ in range(self.__nodes_number)]

        for i, j in capacities.keys():
            self.__fan_in_nodes[j].append(i)
            self.__fan_out_nodes[i].append(j)

    def size(self):
        return self.__nodes_number

    def get_capacities(self):
        return self.__capacities


class ResidualNetwork(inetwork.INetwork):
    def __init__(self):
        self.__network: tp.Optional[inetwork.INetwork] = None

    def get_capacities(self):
        return self.__network.get_capacities()

    def get_flow(self):
        return self.__network.get_flow()

    def size(self):
        return self.__network.size()

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        a1 = [i for i in range(self.size()) if self.edge_exist_q((i, node_id), EdgeType.NORMAL)]
        a2 = [i for i in range(self.size()) if self.edge_exist_q((i, node_id), EdgeType.INVERTED)]
        return a1 + a2

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        a1 = [i for i in range(self.size()) if self.edge_exist_q((node_id, i), EdgeType.NORMAL)]
        a2 = [i for i in range(self.size()) if self.edge_exist_q((node_id, i), EdgeType.INVERTED)]
        return a1 + a2

    def get_edge_capacity(self, edge_id: EdgeId) -> FlowValue:
        return self.__network.get_edge_capacity(edge_id)

    def get_edge_flow(self, edge_id: EdgeId, edge_type: EdgeType) -> FlowValue:
        if edge_type == EdgeType.INVERTED:
            return self.__network.get_edge_flow(reverse_edge(edge_id))
        else:
            return self.__network.get_edge_capacity(edge_id) - self.__network.get_edge_flow(edge_id)

    def get_source(self) -> NodeId:
        return self.__network.get_source()

    def get_sink(self) -> NodeId:
        return self.__network.get_sink()

    def edge_exist_q(self, edge_id: EdgeId, edge_type: EdgeType) -> bool:
        return self.get_edge_flow(edge_id, edge_type) > 0

    def setup(self, network: inetwork.INetwork):
        self.__network = network
