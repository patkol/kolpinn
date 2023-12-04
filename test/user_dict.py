import collections

class Quantity:
    def __init__(self, grid):
        self.grid = grid

class QuantityDict(collections.UserDict):
    def __init__(self, grid):
        super().__init__()
        self.grid = grid

    def __setitem__(self, label: str, quantity: Quantity):
        assert quantity.grid is self.grid, f'{quantity.grid}, {self.grid}'
        super().__setitem__(label, quantity)

a = Quantity('o')
b = Quantity('o')
c = Quantity('x')
d = Quantity('o')
e = Quantity('o')

dct = QuantityDict('o')
print(dct)
dct['a'] = a
dct['b'] = b
print(dct)
dct.update({'d': d, 'e': e})
print(dct)
dct.update({'c': c})
#dct['c'] = c
