from abc import ABC, abstractmethod


class INetwork(ABC):
    @abstractmethod
    def get_node_fan_in(self, node):
        raise NotImplementedError

    @abstractmethod
    def get_node_fan_out(self, node):
        raise NotImplementedError

    @abstractmethod
    def get_edge_capacity(self, edge):
        raise NotImplementedError

    @abstractmethod
    def get_edge_flow(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_source(self):
        raise NotImplementedError

    @abstractmethod
    def get_sink(self):
        raise NotImplementedError

    @abstractmethod
    def edge_exist_q(self, *args) -> bool:
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args):
        raise NotImplementedError

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def get_flow(self):
        raise NotImplementedError

    @abstractmethod
    def get_capacities(self):
        raise NotImplementedError
