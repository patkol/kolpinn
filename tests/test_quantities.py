# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import List, Dict
import pytest
import torch

from kolpinn import grids
from kolpinn import quantities

import utilities


def test_QuantityDict():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3, 1, 2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(1, 1, 1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(1, 1, 2), seed=2)
    tensor4 = utilities.get_random_tensor(size=(2, 1, 2), seed=3)

    q = quantities.QuantityDict(grid, {"1": tensor1, "2": tensor2})
    q["3"] = tensor3
    q.overwrite("1", tensor3)

    assert q["1"] is tensor3
    assert q["2"] is tensor2
    assert q["3"] is tensor3

    # Cannot overwrite through indexing
    with pytest.raises(AssertionError):
        q["2"] = tensor1

    # All elements must be tensors compatible with `grid`
    with pytest.raises(AssertionError):
        q["5"] = [0.3, 7.1]
    with pytest.raises(AssertionError):
        q["6"] = tensor4
    with pytest.raises(AssertionError):
        quantities.QuantityDict(grid, {"4": tensor4})


def test_QuantityDict_empty():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)

    q = quantities.QuantityDict(grid)

    assert len(q) == 0


def test_compatible():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3, 1, 2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(1, 1, 1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(1, 1, 2), seed=2)
    tensor4 = utilities.get_random_tensor(size=(2, 1, 2), seed=3)
    tensor5 = utilities.get_random_tensor(size=(3, 3, 2), seed=4)

    assert quantities.compatible(tensor1, grid)
    assert quantities.compatible(tensor2, grid)
    assert quantities.compatible(tensor3, grid)
    assert not quantities.compatible(tensor4, grid)
    assert not quantities.compatible(tensor5, grid)


def test_is_singleton_dimension():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1, 1, 2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(2, 1, 2), seed=1)

    assert quantities.is_singleton_dimension("a", tensor1, grid)
    assert quantities.is_singleton_dimension("b", tensor1, grid)
    assert not quantities.is_singleton_dimension("c", tensor1, grid)

    # Must be compatible
    with pytest.raises(AssertionError):
        quantities.is_singleton_dimension("b", tensor2, grid)


def test_might_depend_on():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(1, 1, 2), seed=0)

    assert not quantities.might_depend_on("a", tensor, grid)
    assert quantities.might_depend_on("b", tensor, grid)
    assert quantities.might_depend_on("c", tensor, grid)


def test_sum_dimension():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(3, 1, 2), seed=0)

    summed_tensor = quantities.sum_dimension("c", tensor, grid)

    assert summed_tensor.size() == (3, 1, 1)
    assert torch.allclose(summed_tensor[:, :, 0], tensor[:, :, 0] + tensor[:, :, 1])


def test_sum_dimension_singleton_grid():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(3, 1, 2), seed=0)

    summed_tensor = quantities.sum_dimension("b", tensor, grid)

    assert torch.equal(summed_tensor, tensor)


def test_sum_dimension_singleton_dimension():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(1, 1, 2), seed=0)

    summed_tensor = quantities.sum_dimension("a", tensor, grid)

    assert torch.allclose(summed_tensor, 3 * tensor)


def test_restrict():
    sizes = {"a": 3, "b": 1, "c": 2}
    subgrid_indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    grid = utilities.get_random_grid(sizes, seed=0)
    subgrid = grids.Subgrid(grid, subgrid_indices_dict, copy_all=False)
    tensor = utilities.get_random_tensor(size=(3, 1, 2), seed=0)

    restricted_tensor = quantities.restrict(tensor, subgrid)

    assert quantities.compatible(restricted_tensor, subgrid)
    assert restricted_tensor[0, 0, 0] == tensor[2, 0, 1]
    assert restricted_tensor[1, 0, 0] == tensor[1, 0, 1]


def test_restrict_trivial():
    sizes = {"a": 3, "b": 1, "c": 2}
    subgrid_indices_dict: Dict[str, list[int]] = {}
    grid = utilities.get_random_grid(sizes, seed=0)
    subgrid = grids.Subgrid(grid, subgrid_indices_dict, copy_all=False)
    tensor = utilities.get_random_tensor(size=(3, 1, 2), seed=0)

    restricted_tensor = quantities.restrict(tensor, subgrid)

    assert quantities.compatible(restricted_tensor, subgrid)
    assert restricted_tensor is tensor


def test_restrict_with_singleton():
    sizes = {"a": 3, "b": 1, "c": 2}
    subgrid_indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    grid = utilities.get_random_grid(sizes, seed=0)
    subgrid = grids.Subgrid(grid, subgrid_indices_dict, copy_all=False)
    tensor = utilities.get_random_tensor(size=(1, 1, 2), seed=0)

    restricted_tensor = quantities.restrict(tensor, subgrid)

    assert quantities.compatible(restricted_tensor, subgrid)
    assert torch.equal(restricted_tensor, tensor[:, :, [1]])


def test_restrict_all_singleton():
    sizes = {"a": 3, "b": 1, "c": 2}
    subgrid_indices_dict = {
        "a": [2, 1],
        "c": [1],
    }
    grid = utilities.get_random_grid(sizes, seed=0)
    subgrid = grids.Subgrid(grid, subgrid_indices_dict, copy_all=False)
    tensor = utilities.get_random_tensor(size=(1, 1, 1), seed=0)

    restricted_tensor = quantities.restrict(tensor, subgrid)

    assert quantities.compatible(restricted_tensor, subgrid)
    assert restricted_tensor is tensor


def test_expand_all_dims():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1, 1, 1), seed=0)
    tensor2 = utilities.get_random_tensor(size=(3, 1, 1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(3, 1, 2), seed=2)

    tensor1_expanded = quantities.expand_all_dims(tensor1, grid)
    tensor2_expanded = quantities.expand_all_dims(tensor2, grid)
    tensor3_expanded = quantities.expand_all_dims(tensor3, grid)

    assert tensor1_expanded.size() == (3, 1, 2)
    assert torch.all(tensor1_expanded == tensor1)
    assert tensor2_expanded.size() == (3, 1, 2)
    assert torch.all(tensor2_expanded == tensor2)
    assert torch.equal(tensor3_expanded, tensor3)


def test_squeeze_to():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1, 1, 1), seed=0)
    tensor2 = utilities.get_random_tensor(size=(3, 1, 1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(3, 1, 2), seed=2)

    tensor1_squeezed = quantities.squeeze_to(["b"], tensor1, grid)
    tensor1_completely_squeezed = quantities.squeeze_to([], tensor1, grid)
    tensor2_squeezed = quantities.squeeze_to(["a"], tensor2, grid)
    tensor3_squeezed = quantities.squeeze_to(["a", "c"], tensor3, grid)
    tensor3_not_squeezed = quantities.squeeze_to(["a", "b", "c"], tensor3, grid)

    assert tensor1_squeezed.size() == (1,)
    assert torch.equal(tensor1_squeezed, tensor1[0, :, 0])
    assert tensor1_completely_squeezed.size() == ()
    assert tensor2_squeezed.size() == (3,)
    assert torch.equal(tensor2_squeezed, tensor2[:, 0, 0])
    assert tensor3_squeezed.size() == (3, 2)
    assert torch.equal(tensor3_squeezed, tensor3[:, 0, :])
    assert tensor3_not_squeezed.size() == (3, 1, 2)
    assert torch.equal(tensor3_not_squeezed, tensor3)

    # The squeezed tensor must possibly depend on all dimensions
    with pytest.raises(AssertionError):
        quantities.squeeze_to(["b", "c"], tensor1, grid)


def test_squeeze_too_much():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3, 1, 1), seed=1)
    tensor2 = utilities.get_random_tensor(size=(3, 1, 2), seed=2)

    # Cannot squeeze in a dimension the tensor depends on
    with pytest.raises(AssertionError):
        quantities.squeeze_to(["b", "c"], tensor2, grid)
    with pytest.raises(AssertionError):
        quantities.squeeze_to([], tensor1, grid)


def test_unsqueeze_to():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3, 1, 1), seed=1)
    tensor2 = utilities.get_random_tensor(size=(3, 1, 2), seed=2)
    tensor3 = utilities.get_random_tensor(size=(1,), seed=3)
    tensor4 = utilities.get_random_tensor(size=(3, 2), seed=4)
    tensor5 = utilities.get_random_tensor(size=(2, 1, 3), seed=4)

    tensor2_unsqueezed = quantities.unsqueeze_to(grid, tensor2, ("a", "b", "c"))
    tensor3_unsqueezed = quantities.unsqueeze_to(grid, tensor3, ("b",))
    tensor4_unsqueezed = quantities.unsqueeze_to(grid, tensor4, ("a", "c"))

    assert torch.equal(tensor2, tensor2_unsqueezed)
    assert tensor3_unsqueezed.size() == (1, 1, 1)
    assert torch.equal(tensor3_unsqueezed[0, :, 0], tensor3)
    assert tensor4_unsqueezed.size() == (3, 1, 2)
    assert torch.equal(tensor4_unsqueezed[:, 0, :], tensor4)

    # Input must depend on all dimensions
    with pytest.raises(AssertionError):
        quantities.unsqueeze_to(grid, tensor1, ("a", "b", "c"))
    with pytest.raises(AssertionError):
        quantities.unsqueeze_to(grid, tensor3, ("c"))

    # Input needs to be ordered
    with pytest.raises(AssertionError):
        quantities.unsqueeze_to(grid, tensor5, ("c", "b", "a"))


def test_unsqueeze_to_from_constant():
    sizes = {"a": 3, "b": 1, "c": 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(), seed=1)

    tensor_unsqueezed = quantities.unsqueeze_to(grid, tensor, ())

    assert tensor_unsqueezed.size() == (1, 1, 1)
    assert tensor_unsqueezed.item() == tensor.item()


def test_combine_quantity():
    parent_sizes = {"a": 3, "b": 1, "c": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dicts: List[Dict] = [
        {"a": [2, 0], "c": [1]},
        {"a": [2], "c": [0]},
        {"a": [0], "c": [0]},
        {"a": [], "c": [0, 1]},
        {"a": [], "c": []},
        {"a": [1], "c": [0, 1]},
    ]
    subgrids = [
        grids.Subgrid(parent_grid, indices_dict, copy_all=False)
        for indices_dict in indices_dicts
    ]
    quantity = utilities.get_random_tensor(size=(3, 1, 2), seed=1)
    sub_quantities = [quantities.restrict(quantity, subgrid) for subgrid in subgrids]

    combined_quantity = quantities.combine_quantity(
        sub_quantities,
        subgrids,
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity)

    # The whole grid must be covered
    with pytest.raises(AssertionError):
        quantities.combine_quantity(
            sub_quantities[:-1],
            subgrids[:-1],
            parent_grid,
        )

    # No double coverage
    with pytest.raises(AssertionError):
        quantities.combine_quantity(
            sub_quantities + [sub_quantities[1]],
            subgrids + [subgrids[1]],
            parent_grid,
        )


def test_combine_quantity_empty_grid():
    parent_sizes = {"a": 3, "b": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grids.Subgrid(
        parent_grid,
        {"a": [0, 1, 2], "b": [0, 1]},
        copy_all=False,
    )
    subgrid_empty = grids.Subgrid(
        parent_grid,
        {"a": [], "b": []},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(3, 2), seed=1)
    quantity_empty = utilities.get_random_tensor(size=(0, 0), seed=2)

    combined_quantity = quantities.combine_quantity(
        [quantity_full, quantity_empty],
        [subgrid_full, subgrid_empty],
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity_full)


def test_combine_quantity_expands_sliced_singleton_dimension():
    parent_sizes = {"a": 3, "b": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grids.Subgrid(
        parent_grid,
        {"a": [0, 1, 2], "b": [0, 1]},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(1, 2), seed=1)

    combined_quantity = quantities.combine_quantity(
        [quantity_full],
        [subgrid_full],
        parent_grid,
    )

    assert combined_quantity.size() == (3, 2)
    assert torch.all(combined_quantity == quantity_full)


def test_combine_quantity_leaves_unsliced_singleton_dimension():
    parent_sizes = {"a": 3, "b": 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grids.Subgrid(
        parent_grid,
        {"b": [0, 1]},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(1, 2), seed=1)

    combined_quantity = quantities.combine_quantity(
        [quantity_full],
        [subgrid_full],
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity_full)
