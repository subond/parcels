from parcels.grid import GridIndex
from ctypes import Structure, c_int, POINTER, c_void_p
import numpy as np

__all__ = ['GridSet', 'GridIndexSet']


class GridSet(object):
    """GridSet class that holds the Grids on which the Fields are defined


    :param grids: Table of :class:`parcels.grid.Grid` objects
    """

    def __init__(self, grids=[]):
        self.grids = grids
        self.size = len(grids)

    def add_grid(self, grid):
        existing_grid = False
        for g in self.grids:
            sameGrid = True
            for attr in ['lon', 'lat', 'depth', 'time']:
                gattr = getattr(g, attr)
                gridattr = getattr(grid, attr)
                if gattr.shape != gridattr.shape:
                    sameGrid = False
                    break
            if not sameGrid:
                continue
            existing_grid = True
            grid.name = g.name
            break

        if not existing_grid:
            for g in self.grids:
                if g.name == grid.name:
                    grid.name = grid.name + '_b'
            self.grids.append(grid)
            self.size += 1


class GridIndexSet(object):
    """GridIndexSet class that holds the GridIndices which store the particle position indices for the different grids

    :param gridset: GridSet object
    """
    def __init__(self, id, gridset):
        self.size = gridset.size
        self.gridindices = np.empty(self.size, GridIndex)
        self._gridindices_data = np.empty(self.size, GridIndex.dtype())

        def cptr(i):
            return self._gridindices_data[i]

        for i, g in enumerate(gridset.grids):
            self.gridindices[i] = GridIndex(g, cptr=cptr(i))

    @property
    def ctypes_struct(self):
        class CGridIndexSet(Structure):
            _fields_ = [('size', c_int),
                        ('grids', POINTER(c_void_p))]
        cstruct = CGridIndexSet(self.size,
                                self._gridindices_data.ctypes.data_as(POINTER(c_void_p)))
        return cstruct
