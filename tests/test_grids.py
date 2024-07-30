# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import pytest
import copy
import torch

from kolpinn import grids

import utilities


def test_Grid_init():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)

    assert grid.shape == tuple(sizes.values())
    assert grid.n_dim == len(sizes)
    assert grid.n_points == sizes["a"] * sizes["b"] * sizes["c"]
    assert grid.dim_size == sizes
    assert grid.index == {"a": 0, "b": 1, "c": 2}


def test_Grid_get_subgrids():
    dtype = torch.float32
    a_tensor = torch.tensor([0, 1, 2], dtype=dtype)
    b_tensor = torch.tensor([-7.1, -1.2, 0.2, 5.2], dtype=dtype)
    c_tensor = torch.tensor([8], dtype=dtype)
    dimensions = {"a": a_tensor, "b": b_tensor, "c": c_tensor}
    grid = grids.Grid(dimensions)

    def greater_than_one(x):
        return x > 1

    conditions_dicts = {
        "positive_b": {"b": lambda x: x > 0},
        "negative_a": {"a": lambda x: x < 0},
        "no_condition": {},
        "greater_than_one": dict((key, greater_than_one) for key in dimensions.keys()),
    }

    subgrids = grid.get_subgrids(conditions_dicts, copy_all=False)

    assert subgrids["positive_b"]["a"] is a_tensor
    assert torch.equal(subgrids["positive_b"]["b"], b_tensor[2:])
    assert subgrids["positive_b"]["c"] is c_tensor

    assert len(subgrids["negative_a"]["a"]) == 0
    assert subgrids["negative_a"]["b"] is b_tensor
    assert subgrids["negative_a"]["c"] is c_tensor

    assert subgrids["no_condition"]["a"] is a_tensor
    assert subgrids["no_condition"]["b"] is b_tensor
    assert subgrids["no_condition"]["c"] is c_tensor

    assert torch.equal(subgrids["greater_than_one"]["a"], a_tensor[2:])
    assert torch.equal(subgrids["greater_than_one"]["b"], b_tensor[-1:])
    # Not is - all entries of c_tensor pass the condition, but there
    # is a condition, so c_tensor[:] is the new dimension
    assert torch.equal(subgrids["greater_than_one"]["c"], c_tensor)


def test_Subgrid():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict = {
        "a": [2, 1],
        "c": torch.tensor([1]),
    }

    subgrid = grids.Subgrid(parent_grid, indices_dict, copy_all=False)
    subsubgrid = grids.Subgrid(subgrid, {}, copy_all=True)

    assert subgrid["a"][0] == parent_grid["a"][2]
    assert subgrid["a"][1] == parent_grid["a"][1]
    assert subgrid["b"] is parent_grid["b"]
    assert subgrid["c"].item() == parent_grid["c"][1]
    assert subgrid.descends_from(parent_grid)
    assert torch.equal(subsubgrid["a"], subgrid["a"])
    assert subsubgrid["a"] is not subgrid["a"]  # We copied
    assert subsubgrid.descends_from(subgrid)
    assert subsubgrid.descends_from(parent_grid)


def test_Subgrid_full():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)

    subgrid = grids.Subgrid(parent_grid, {}, copy_all=False)

    for key in parent_sizes.keys():
        assert subgrid[key] is parent_grid[key]
    assert subgrid.descends_from(parent_grid)


def test_Subgrid_empty():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict: dict = dict((key, []) for key in parent_sizes.keys())

    subgrid = grids.Subgrid(parent_grid, indices_dict, copy_all=True)

    for key in parent_sizes.keys():
        assert subgrid.dim_size[key] == 0
    assert subgrid.descends_from(parent_grid)


def test_Supergrid():
    child1_sizes = {"a": 3, "b": 1, "c": 2}
    child1_dimensions = utilities.get_random_dimensions(child1_sizes, seed=0)
    child2_dimensions = copy.copy(child1_dimensions)
    child2_dimensions["b"] = utilities.get_random_tensor(size=(3,), seed=1)
    child3_dimensions = copy.copy(child1_dimensions)
    child3_dimensions["b"] = utilities.get_random_tensor(size=(2,), seed=3)
    child_grids = {
        "child1": grids.Grid(child1_dimensions),
        "child2": grids.Grid(child2_dimensions),
        "child3": grids.Grid(child3_dimensions),
    }

    supergrid = grids.Supergrid(child_grids, "b", copy_all=False)

    assert supergrid["a"] is child_grids["child1"]["a"]
    assert supergrid["c"] is child_grids["child3"]["c"]
    assert len(supergrid["b"]) == 6
    assert supergrid["b"][0] == child_grids["child1"]["b"][0]
    assert torch.equal(supergrid["b"][1:4], child_grids["child2"]["b"])
    assert torch.equal(supergrid["b"][4:], child_grids["child3"]["b"])

    for key in ("a", "b", "c"):
        for i in (1, 2, 3):
            assert torch.equal(
                supergrid.subgrids[f"child{i}"][key],
                child_grids[f"child{i}"][key],
            )


def test_Supergrid_single():
    child_sizes = {"a": 3, "b": 1, "c": 2}
    child_grid = utilities.get_random_grid(child_sizes, seed=0)
    child_grids = {"child": child_grid}

    supergrid = grids.Supergrid(child_grids, "c", copy_all=False)

    assert supergrid["a"] is child_grid["a"]
    assert supergrid["b"] is child_grid["b"]
    assert torch.equal(
        supergrid["c"],
        child_grid["c"],
    )


def test_get_as_subsubgrid_additional_dim():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict1 = {
        "a": [2, 1],
        "c": torch.tensor([1]),
    }
    indices_dict2 = {
        "a": [1, 2],
    }
    subgrid1 = grids.Subgrid(parent_grid, indices_dict1, copy_all=False)
    subgrid2 = grids.Subgrid(parent_grid, indices_dict2, copy_all=False)

    subsubgrid = grids.get_as_subsubgrid(subgrid1, subgrid2, copy_all=False)

    assert subsubgrid.parent_grid is subgrid2
    for dimension in subsubgrid.dimensions:
        assert torch.equal(
            subsubgrid.dimensions[dimension], subgrid1.dimensions[dimension]
        ), dimension


def test_get_as_subsubgrid_equal_grids():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    subgrid = grids.Subgrid(parent_grid, indices_dict, copy_all=False)

    subsubgrid = grids.get_as_subsubgrid(subgrid, subgrid, copy_all=False)

    assert subsubgrid.parent_grid is subgrid
    for dimension in subsubgrid.dimensions:
        assert torch.equal(
            subsubgrid.dimensions[dimension], subgrid.dimensions[dimension]
        ), dimension
        assert dimension not in subsubgrid.indices_dict


def test_get_as_subsubgrid_same_dim():
    parent_sizes = {"a": 3, "b": 1, "c": 4}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict1 = {
        "a": [2, 1],
        "c": [1],
    }
    indices_dict2 = {
        "a": [1, 2, 0],
        "c": [0, 1, 2],
    }
    subgrid1 = grids.Subgrid(parent_grid, indices_dict1, copy_all=False)
    subgrid2 = grids.Subgrid(parent_grid, indices_dict2, copy_all=False)

    subsubgrid = grids.get_as_subsubgrid(subgrid1, subgrid2, copy_all=False)

    assert subsubgrid.parent_grid is subgrid2
    for dimension in subsubgrid.dimensions:
        assert torch.equal(
            subsubgrid.dimensions[dimension], subgrid1.dimensions[dimension]
        ), dimension


def test_get_as_subsubgrid_missing_coords():
    parent_sizes = {"a": 3, "b": 1, "c": 4}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dict1 = {
        "a": [2, 1],
        "c": [1, 2],
    }
    indices_dict2 = {
        "a": [1, 2, 0],
        "c": [0, 1],
    }
    subgrid1 = grids.Subgrid(parent_grid, indices_dict1, copy_all=False)
    subgrid2 = grids.Subgrid(parent_grid, indices_dict2, copy_all=False)

    with pytest.raises(ValueError):
        grids.get_as_subsubgrid(subgrid1, subgrid2, copy_all=False)

    with pytest.raises(ValueError):
        grids.get_as_subsubgrid(subgrid2, subgrid1, copy_all=False)


def test_get_as_subgrid_same_dim():
    parent_sizes = {"a": 3, "b": 1, "c": 4}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_indices_dict = {
        "a": [1, 2, 0],
        "c": [0, 1, 2],
    }
    subsubgrid_indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    subgrid = grids.Subgrid(parent_grid, subgrid_indices_dict, copy_all=False)
    subsubgrid = grids.Subgrid(subgrid, subsubgrid_indices_dict, copy_all=False)

    new_subgrid = grids.get_as_subgrid(subsubgrid, copy_all=False)

    assert new_subgrid.parent_grid is parent_grid
    for dimension in parent_grid.dimensions:
        assert torch.equal(
            new_subgrid.dimensions[dimension],
            subsubgrid.dimensions[dimension],
        ), dimension


def test_get_as_subgrid_dim_only_in_subgrid():
    parent_sizes = {"a": 3, "b": 1, "c": 4}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_indices_dict = {
        "a": [1, 2, 0],
        "c": [0, 1, 2],
    }
    subsubgrid_indices_dict = {
        "c": [1],
    }
    subgrid = grids.Subgrid(parent_grid, subgrid_indices_dict, copy_all=False)
    subsubgrid = grids.Subgrid(subgrid, subsubgrid_indices_dict, copy_all=False)

    new_subgrid = grids.get_as_subgrid(subsubgrid, copy_all=False)

    assert new_subgrid.parent_grid is parent_grid
    for dimension in parent_grid.dimensions:
        assert torch.equal(
            new_subgrid.dimensions[dimension],
            subsubgrid.dimensions[dimension],
        ), dimension


def test_get_as_subgrid_dim_only_in_subsubgrid():
    parent_sizes = {"a": 3, "b": 1, "c": 4}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_indices_dict = {
        "a": [1, 2, 0],
    }
    subsubgrid_indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    subgrid = grids.Subgrid(parent_grid, subgrid_indices_dict, copy_all=False)
    subsubgrid = grids.Subgrid(subgrid, subsubgrid_indices_dict, copy_all=False)

    new_subgrid = grids.get_as_subgrid(subsubgrid, copy_all=False)

    assert new_subgrid.parent_grid is parent_grid
    for dimension in parent_grid.dimensions:
        assert torch.equal(
            new_subgrid.dimensions[dimension],
            subsubgrid.dimensions[dimension],
        ), dimension
