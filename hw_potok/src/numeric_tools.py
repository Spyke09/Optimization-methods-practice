class Infinity:
    def __add__(self, other):
        if isinstance(other, MinusInfinity):
            raise ValueError("It is forbidden to subtract infinity from infinity")
        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            raise ValueError("It is forbidden to subtract infinity from infinity")
        return self

    def __lt__(self, other):
        if isinstance(other, type(self)) or isinstance(other, MinusInfinity):
            raise ValueError("It is forbidden to compare infinities with each other.")
        return False

    def __gt__(self, other):
        if isinstance(other, type(self)) or isinstance(other, MinusInfinity):
            raise ValueError("It is forbidden to compare infinities with each other.")
        return True

    def __repr__(self):
        return "∞"


class MinusInfinity:
    def __add__(self, other):
        if isinstance(other, Infinity):
            raise ValueError("It is forbidden to subtract infinity from infinity")
        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            raise ValueError("It is forbidden to subtract infinity from infinity")
        return self

    def __lt__(self, other):
        if isinstance(other, type(self)):
            raise ValueError("It is forbidden to compare infinities with each other.")
        return True

    def __gt__(self, other):
        if isinstance(other, type(self)):
            raise ValueError("It is forbidden to compare infinities with each other.")
        return False

    def __repr__(self):
        return "-∞"
