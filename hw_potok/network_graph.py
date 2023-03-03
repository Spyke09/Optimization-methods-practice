import typing as tp
from enum import Enum

import inetwork


class EdgeType(Enum):
    NORMAL = 0
    INVERSE = 1


NodeId = int
FlowValue = float
EdgeId = tp.Tuple[NodeId, NodeId, EdgeType]


def inverse_edge(edge_id: EdgeId) -> EdgeId:
    new_type = EdgeType.INVERSE if edge_id[2] == EdgeType.NORMAL else EdgeType.NORMAL
    return edge_id[0], edge_id[1], new_type


def true_inverse_edge(edge_id: EdgeId) -> EdgeId:
    new_type = EdgeType.INVERSE if edge_id[2] == EdgeType.NORMAL else EdgeType.NORMAL
    return edge_id[1], edge_id[0], new_type


def make_norm_edge(edge_id: EdgeId) -> EdgeId:
    return edge_id[0], edge_id[1], EdgeType.NORMAL


def make_inverse_edge(edge_id: EdgeId) -> EdgeId:
    return edge_id[1], edge_id[0], EdgeType.INVERSE


class SimpleNetwork(inetwork.INetwork):
    def __init__(self):
        self.__fan_in_nodes: tp.List[tp.List[NodeId]] = list()
        self.__fan_out_nodes: tp.List[tp.List[NodeId]] = list()
        self.__capacities: tp.Dict[EdgeId, FlowValue] = dict()
        self.__flow: tp.Dict[EdgeId, FlowValue] = dict()
        self.__source: NodeId = -1
        self.__sink: NodeId = -1
        self.__nodes_number = 0
        self.__edges_number = 0

    def __repr__(self) -> str:
        s = ""
        for i, j, k in self.__capacities:
            a = i, j, k
            s += f"({i}, {j}): {self.__flow[a]}/{self.__capacities[a]}\n"
        return s

    def edge_exist_q(self, edge_id: EdgeId) -> bool:
        return edge_id in self.__capacities

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_in_nodes[node_id]

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_out_nodes[node_id]

    def get_edge_capacity(self, edge_id: EdgeId) -> FlowValue:
        return self.__capacities[edge_id]

    def get_edge_flow(self, edge_id: EdgeId) -> FlowValue:
        return self.__flow[edge_id]

    def set_edge_flow(self, edge_id: EdgeId, value: FlowValue) -> None:
        self.__flow[edge_id] = value

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
            sum_in += self.__flow[(j, node_id, EdgeType.NORMAL)]

        sum_out = 0
        for j in self.get_node_fan_out(node_id):
            sum_out += self.__flow[(node_id, j, EdgeType.NORMAL)]

        return sum_in - sum_out

    def get_network_flow(self) -> FlowValue:
        return -self.get_excess_flow(self.__source)

    def setup(
            self,
            list_edges: tp.List[tp.Tuple[int, int]],
            capacities: tp.Dict[tp.Tuple[int, int], FlowValue],
            source: NodeId,
            sink: NodeId
    ) -> None:
        self.__capacities = {(i[0], i[1], EdgeType.NORMAL): j for i, j in capacities.items()}
        self.__source = source
        self.__sink = sink
        self.__edges_number = len(list_edges)

        node_set = set()
        for i, j in list_edges:
            node_set.add(i)
            node_set.add(j)
        self.__nodes_number = len(node_set)

        self.__fan_in_nodes = [[] for _ in range(self.__nodes_number)]
        self.__fan_out_nodes = [[] for _ in range(self.__nodes_number)]
        self.__flow = {i: 0 for i in self.__capacities}
        for i, j in list_edges:
            self.__fan_in_nodes[j].append(i)
            self.__fan_out_nodes[i].append(j)


class ResidualNetwork(inetwork.INetwork):
    def __init__(self):
        self.__network: tp.Optional[inetwork.INetwork] = None

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        inp: tp.List[NodeId] = self.__network.get_node_fan_in(node_id)
        out: tp.List[NodeId] = self.__network.get_node_fan_out(node_id)
        result_norm = [i for i in inp if self.edge_exist_q((i, node_id, EdgeType.NORMAL))]
        result_inv = [i for i in out if self.edge_exist_q((i, node_id, EdgeType.INVERSE))]
        return result_norm + result_inv

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        inp: tp.List[NodeId] = self.__network.get_node_fan_in(node_id)
        out: tp.List[NodeId] = self.__network.get_node_fan_out(node_id)
        result_norm = [i for i in out if self.edge_exist_q((node_id, i, EdgeType.NORMAL))]
        result_inv = [i for i in inp if self.edge_exist_q((node_id, i, EdgeType.INVERSE))]
        return result_norm + result_inv

    def get_edge_capacity(self, edge_id: EdgeId) -> FlowValue:
        return self.__network.get_edge_capacity(make_norm_edge(edge_id))

    def get_edge_flow(self, edge_id: EdgeId) -> FlowValue:
        if edge_id[2] == EdgeType.NORMAL:
            return self.__network.get_edge_capacity(edge_id) - self.__network.get_edge_flow(edge_id)
        if edge_id[2] == EdgeType.INVERSE:
            return self.__network.get_edge_flow(inverse_edge(edge_id))

    def get_source(self) -> NodeId:
        return self.__network.get_source()

    def get_sink(self) -> NodeId:
        return self.__network.get_sink()

    def edge_exist_q(self, edge_id: EdgeId) -> bool:
        if edge_id[2] == EdgeType.NORMAL and self.__network.edge_exist_q(edge_id):
            return self.get_edge_flow(edge_id) > 0
        elif edge_id[2] == EdgeType.INVERSE and self.__network.edge_exist_q(inverse_edge(edge_id)):
            return self.get_edge_flow(edge_id) > 0
        else:
            return False

    def setup(self, network: inetwork.INetwork):
        self.__network = network
