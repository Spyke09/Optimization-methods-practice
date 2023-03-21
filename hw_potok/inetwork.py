import typing as tp
from abc import ABC, abstractmethod
from enum import Enum


class EdgeType(Enum):
    NORMAL = 0
    INVERTED = 1


NodeId = int
FlowValue = float
EdgeId = tp.Tuple[NodeId, NodeId]
Edge = tp.Tuple[EdgeId, EdgeType]


class INetwork(ABC):
    @abstractmethod
    def get_node_fan_in(self, node):
        raise NotImplementedError

    @abstractmethod
    def get_node_fan_out(self, node):
        raise NotImplementedError

    @abstractmethod
    def get_edge_capacity(self, edge, edge_type):
        raise NotImplementedError

    @abstractmethod
    def get_edge_flow(self, edge, edge_type):
        raise NotImplementedError

    @abstractmethod
    def edge_exist_q(self, edge, edge_type) -> bool:
        raise NotImplementedError

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def get_capacities(self):
        raise NotImplementedError

    @abstractmethod
    def get_source(self):
        raise NotImplementedError

    @abstractmethod
    def get_sink(self):
        raise NotImplementedError
