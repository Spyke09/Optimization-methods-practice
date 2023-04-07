import typing as tp

import algo
import inetwork
from inetwork import EdgeType, NodeId, EdgeId, FlowValue, Edge


def reverse_edge(edge_id: EdgeId) -> EdgeId:
    return edge_id[1], edge_id[0]


class SimpleNetwork(inetwork.INetwork):
    def __init__(
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

    def clear(self):
        self.__flow = [[0.0 for _ in range(self.size())] for _ in range(self.size())]

    def to_str(self) -> str:
        max_len = 0
        for i in range(self.__nodes_number):
            for j in range(self.__nodes_number):
                if self.__capacities[i][j] > 0:
                    max_len = max(max_len, len(f"{(i, j)}: {self.__flow[i][j]}/{self.__capacities[i][j]} "))

        res = ""
        for i in range(self.__nodes_number):
            for j in range(self.__nodes_number):
                if self.__capacities[i][j] > 0:
                    aboba = f"{(i, j)}: {self.__flow[i][j]}/{self.__capacities[i][j]} "
                    res += aboba + (max_len - len(aboba) + 1) * " "
                else:
                    res += max_len * "_" + " "
            res += "\n"

        return res

    def get_flow(self) -> tp.List[tp.List[FlowValue]]:
        return self.__flow

    def edge_exist_q(self, edge_id: EdgeId, edge_type: EdgeType = EdgeType.NORMAL) -> bool:
        if edge_type == EdgeType.INVERTED:
            return False
        return self.__capacities[edge_id[0]][edge_id[1]] > 0

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_in_nodes[node_id]

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        return self.__fan_out_nodes[node_id]

    def get_edge_capacity(self, edge_id: EdgeId, edge_type: EdgeType = EdgeType.NORMAL) -> FlowValue:
        if edge_type == EdgeType.INVERTED:
            return 0
        return self.__capacities[edge_id[0]][edge_id[1]]

    def get_edge_flow(self, edge_id, edge_type: EdgeType = EdgeType.NORMAL) -> FlowValue:
        if edge_type == EdgeType.INVERTED:
            return 0
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

    def size(self) -> int:
        return self.__nodes_number

    def get_capacities(self) -> tp.List[tp.List[FlowValue]]:
        return [[j for j in i] for i in self.__capacities]


class ResidualGraph(inetwork.INetwork):
    def __init__(self, network: SimpleNetwork):
        self.__network: SimpleNetwork = network

    def get_capacities(self) -> tp.List[tp.List[FlowValue]]:
        return self.__network.get_capacities()

    def size(self) -> int:
        return self.__network.size()

    def get_node_fan_in(self, node_id: NodeId) -> tp.List[NodeId]:
        a1 = [i for i in range(self.size()) if self.edge_exist_q((i, node_id), EdgeType.NORMAL)]
        a2 = [i for i in range(self.size()) if self.edge_exist_q((i, node_id), EdgeType.INVERTED)]
        return a1 + a2

    def get_node_fan_out(self, node_id: NodeId) -> tp.List[NodeId]:
        a1 = [i for i in range(self.size()) if self.edge_exist_q((node_id, i), EdgeType.NORMAL)]
        a2 = [i for i in range(self.size()) if self.edge_exist_q((node_id, i), EdgeType.INVERTED)]
        return a1 + a2

    def get_edge_capacity(self, edge_id: EdgeId, edge_type: EdgeType) -> FlowValue:
        if edge_type == EdgeType.INVERTED:
            return self.__network.get_edge_flow(reverse_edge(edge_id), EdgeType.NORMAL)
        else:
            return self.__network.get_edge_capacity(edge_id, EdgeType.NORMAL) - self.__network.get_edge_flow(edge_id, EdgeType.NORMAL)

    def get_edge_flow(self, edge_id: EdgeId, edge_type: EdgeType) -> FlowValue:
        return 0.0

    def get_source(self) -> NodeId:
        return self.__network.get_source()

    def get_sink(self) -> NodeId:
        return self.__network.get_sink()

    def edge_exist_q(self, edge_id: EdgeId, edge_type: EdgeType) -> bool:
        return self.get_edge_capacity(edge_id, edge_type) > 0


class LayeredGraph(inetwork.INetwork):
    def __init__(self, r_network: ResidualGraph):
        self.__r_network: ResidualGraph = r_network
        self.__edges: tp.Set[Edge] = set(algo.BFS.bfs_for_dinica(r_network))
        self.__out: tp.Dict[NodeId, tp.List[Edge]] = dict()
        self.__in: tp.Dict[NodeId, tp.List[Edge]] = dict()
        self.__flow: tp.Dict[Edge, FlowValue] = {i: 0 for i in self.__edges}

        for i in self.__edges:
            if i[0][0] not in self.__out:
                self.__out[i[0][0]] = [i]
            else:
                self.__out[i[0][0]].append(i)

            if i[0][1] not in self.__in:
                self.__in[i[0][1]] = [i]
            else:
                self.__in[i[0][1]].append(i)

    def get_node_fan_in(self, node) -> tp.List[Edge]:
        if node in self.__in:
            return self.__in[node]
        else:
            return []

    def get_node_fan_out(self, node) -> tp.List[Edge]:
        if node in self.__out:
            return self.__out[node]
        else:
            return []

    def get_edge_capacity(self, edge, edge_type) -> FlowValue:
        return self.__r_network.get_edge_capacity(edge, edge_type)

    def get_edge_flow(self, edge, edge_type) -> FlowValue:
        return self.__flow[(edge, edge_type)]

    def set_edge_flow(self, edge, edge_type, value) -> None:
        self.__flow[(edge, edge_type)] += value

    def edge_exist_q(self, edge, edge_type) -> bool:
        return (edge, edge_type) in self.__edges

    def size(self) -> int:
        return self.__r_network.size()

    def get_flow(self) -> tp.Dict[Edge, FlowValue]:
        return {i: j for i, j in self.__flow.items()}

    def get_capacities(self) -> tp.List[tp.List[FlowValue]]:
        return self.__r_network.get_capacities()

    def get_source(self) -> NodeId:
        return self.__r_network.get_source()

    def get_sink(self) -> NodeId:
        return self.__r_network.get_sink()

    def get_excess_flow(self, node_id: NodeId) -> FlowValue:
        sum_in = 0
        for j in self.get_node_fan_in(node_id):
            sum_in += self.__flow[j]

        sum_out = 0
        for j in self.get_node_fan_out(node_id):
            sum_out += self.__flow[j]

        return sum_in - sum_out

    def init_block_way(self) -> None:
        source = self.get_source()
        sink = self.get_sink()
        lifo = [[] for i in range(self.size())]
        frozen = [False for i in range(self.size())]
        for i in range(self.size()):
            for edge in (((source, i), EdgeType.NORMAL), ((source, i), EdgeType.INVERTED)):
                self.__flow[edge] = self.get_edge_capacity(*edge)
                lifo[i].append((edge, self.__flow[edge]))

        order = algo.Topsort.dfs_topsort(self.__out, source)
        while True:
            last_active_node = -1
            for i in order:
                if i == source or i == sink:
                    continue
                for j in self.get_node_fan_out(i):
                    if not frozen[j[0][1]]:
                        h = min(self.get_edge_capacity(*j) - self.__flow[j], self.get_excess_flow(i))
                        self.__flow[j] += h
                        lifo[j[0][1]].append((j, h))
                    if self.get_excess_flow(i) <= 0:
                        break

                if self.get_excess_flow(i) > 0:
                    last_active_node = i

            if last_active_node == -1:
                return None
            else:
                for i, h in reversed(lifo[last_active_node]):
                    self.__flow[i] -= min(h, self.get_excess_flow(last_active_node))
                    if self.get_excess_flow(last_active_node) <= 0:
                        break
                frozen[last_active_node] = True
