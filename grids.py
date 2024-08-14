# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict
import copy
import textwrap
import math
import torch

from . import mathematics


class Grid:
    def __init__(
        self,
        dimensions: Dict[str, torch.Tensor],
    ):
        """
        dimensions[label] is a 1D tensor.

        Tensors on the grid:
        tensor[i, j, ...] represents the value at coordinates (i, j, k, ...)
        in the grid (coordinate ordering according to grid.dimensions).
        For dimensions the tensor does not depend on it has
        singleton dimensions.
        """

        self.dimensions = dimensions
        self.dimensions_labels = list(dimensions.keys())
        self.dtype = list(self.dimensions.values())[0].dtype
        for label in self.dimensions_labels:
            assert dimensions[label].dtype is self.dtype, (
                self.dtype,
                label,
                dimensions[label].dtype,
            )
        self.shape = torch.Size(
            [torch.numel(self.dimensions[label]) for label in self.dimensions_labels]
        )
        self.n_dim = len(self.shape)
        self.n_points = math.prod(self.shape)
        self.dim_size = dict(
            (label, torch.numel(self.dimensions[label]))
            for label in self.dimensions_labels
        )
        self.index = dict(
            (label, index) for index, label in enumerate(self.dimensions_labels)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dimensions.keys()})"

    def __getitem__(self, dimension_label: str) -> torch.Tensor:
        return self.dimensions[dimension_label]

    def get_subgrid(self, conditions_dict: dict, copy_all: bool):
        indices_dict = {}
        for label, condition in conditions_dict.items():
            passing = condition(self.dimensions[label])
            indices_dict[label] = torch.nonzero(passing).flatten()

        return Subgrid(self, indices_dict, copy_all)

    def get_subgrids(self, conditions_dicts: dict, copy_all: bool):
        subgrids = {}
        for grid_label, conditions_dict in conditions_dicts.items():
            subgrids[grid_label] = self.get_subgrid(
                conditions_dict,
                copy_all=copy_all,
            )

        return subgrids


class Subgrid(Grid):
    def __init__(self, parent_grid: Grid, indices_dict: dict, copy_all: bool):
        """
        indices_dict[label]: indices of entries of dimensions[label] to be kept
        If copy_all all dimensions will be deepcopied such that they have
        seperate gradients.
        """

        self.parent_grid = parent_grid
        self.indices_dict = dict(
            (
                label,
                (
                    indices
                    if torch.is_tensor(indices)
                    else torch.tensor(indices, dtype=torch.long)
                ),
            )
            for (label, indices) in indices_dict.items()
        )

        for label, indices in indices_dict.items():
            assert len(indices) == 0 or (
                min(indices) >= 0 and max(indices) < parent_grid.dim_size[label]
            ), f"{label} {indices} {parent_grid.dim_size}"

        dimensions = {}
        for label in parent_grid.dimensions_labels:
            dimension = parent_grid[label]
            if label in indices_dict.keys():
                dimension = dimension[self.indices_dict[label]]
            dimensions[label] = dimension

        if copy_all:
            dimensions = copy.deepcopy(dimensions)

        super().__init__(dimensions)

    def __repr__(self):
        return (
            "Subgrid(\n"
            + textwrap.indent(f"parent_grid={self.parent_grid},\n", "    ")
            + textwrap.indent(f"indices_dict={self.indices_dict}),\n", "    ")
            + ")"
        )

    def descends_from(self, grid: Grid):
        """Whether any possibly higher level parent is grid"""
        if self.parent_grid is grid:
            return True
        if not isinstance(self.parent_grid, Subgrid):
            return False
        return self.parent_grid.descends_from(grid)


class Supergrid(Grid):
    def __init__(
        self,
        child_grids: dict[str, Grid],
        dimension_name: str,
        copy_all: bool,
    ):
        """
        Concatenate the `child_grids` along `dimension_name`.
        The `indices_dict` then allows one to get the values on the
        child grids:
        `dimensions[dimension_name][self.indices_dict[child_grid_name]]`
        corresponds to
        `child_grid.dimensions[dimension_name]`.
        `subgrids` contains the `child_grids` as `Subgrid`s of `self`.
        """

        first_child = next(iter(child_grids.values()))
        dimensions = copy.copy(first_child.dimensions)
        dimensions[dimension_name] = torch.cat(
            [child_grid[dimension_name] for child_grid in child_grids.values()]
        )

        if copy_all:
            dimensions = copy.deepcopy(dimensions)

        super().__init__(dimensions)

        self.indices_dict = {}
        self.subgrids = {}
        first_index = 0
        for child_grid_name, child_grid in child_grids.items():
            for label, dimension in child_grid.dimensions.items():
                assert label == dimension_name or torch.all(
                    dimension == self.dimensions[label]
                ), f"Dimension {label} is not the same for all grids"

            dimension_size = child_grid[dimension_name].size(dim=0)
            next_first_index = first_index + dimension_size
            self.indices_dict[child_grid_name] = torch.arange(
                first_index,
                first_index + dimension_size,
            )
            first_index = next_first_index

            self.subgrids[child_grid_name] = Subgrid(
                self,
                {dimension_name: self.indices_dict[child_grid_name]},
                copy_all=False,
            )

            assert torch.all(
                self.dimensions[dimension_name][self.indices_dict[child_grid_name]]
                == child_grid.dimensions[dimension_name]
            ), child_grid_name

        assert first_index == self.dimensions[dimension_name].size(dim=0)


def get_as_subsubgrid(
    small_subgrid: Subgrid, large_subgrid: Subgrid, *, copy_all: bool
) -> Subgrid:
    """
    Return the small_subgrid as a subgrid of large_subgrid.
    Both subgrids should have the same parent_grid, and large_subgrids
    must contain all gridpoints that small_subgrid does.
    The indices in `large_subgrid` must be sorted.
    """
    assert small_subgrid.parent_grid is large_subgrid.parent_grid

    subsubgrid_indices_dict: Dict[str, list[int]] = {}

    for label in small_subgrid.indices_dict:
        parent_indices_small = list(small_subgrid.indices_dict[label])
        subsubgrid_indices = parent_indices_small

        if label in large_subgrid.indices_dict:
            parent_indices_large = list(large_subgrid.indices_dict[label])
            if parent_indices_small == parent_indices_large:
                # The current dimension is the same on both subgrids
                continue
            subsubgrid_indices = [
                # For unsorted: parent_indices_large.index(parent_index)
                mathematics.binary_search(parent_indices_large, parent_index)
                for parent_index in parent_indices_small
            ]

        subsubgrid_indices_dict[label] = subsubgrid_indices

    subsubgrid = Subgrid(large_subgrid, subsubgrid_indices_dict, copy_all=copy_all)
    return subsubgrid


def get_as_subgrid(subsubgrid: Subgrid, *, copy_all: bool) -> Subgrid:
    """
    Return subsubgrid as a direct subgrid of its grandparent.
    """
    subgrid = subsubgrid.parent_grid
    assert isinstance(subgrid, Subgrid)
    parent_grid = subgrid.parent_grid
    # Edge case: label in subgrid, but not in subsubgrid
    new_subgrid_indices_dict = copy.copy(subgrid.indices_dict)

    for label in subsubgrid.indices_dict:
        subsubgrid_indices = subsubgrid.indices_dict[label]
        # Edge case: label in subsubgrid, but not in subgrid
        new_subgrid_indices = subsubgrid_indices
        if label in subgrid.indices_dict:
            subgrid_indices = subgrid.indices_dict[label]
            new_subgrid_indices = [
                subgrid_indices[subsubgrid_index]
                for subsubgrid_index in subsubgrid_indices
            ]

        new_subgrid_indices_dict[label] = new_subgrid_indices

    new_subgrid = Subgrid(parent_grid, new_subgrid_indices_dict, copy_all=copy_all)
    return new_subgrid
