class Infinity:
    def __add__(self, other):
        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            raise ValueError
        return self

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True
