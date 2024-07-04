# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import pytest
import copy
import torch

from kolpinn import grid_quantities

import utilities


def test_QuantityDict():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3,1,2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(1,1,1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(1,1,2), seed=2)
    tensor4 = utilities.get_random_tensor(size=(2,1,2), seed=3)

    q = grid_quantities.QuantityDict(grid, {'1': tensor1, '2': tensor2})
    q['3'] = tensor3
    q.overwrite('1', tensor3)

    assert q['1'] is tensor3
    assert q['2'] is tensor2
    assert q['3'] is tensor3

    # Cannot overwrite through indexing
    with pytest.raises(Exception) as e_info:
        q['2'] = tensor1

    # All elements must be tensors compatible with `grid`
    with pytest.raises(Exception) as e_info:
        q['5'] = [0.3, 7.1]
    with pytest.raises(Exception) as e_info:
        q['6'] = tensor4
    with pytest.raises(Exception) as e_info:
        grid_quantities.QuantityDict(grid, {'4': tensor4})

def test_QuantityDict_empty():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)

    q = grid_quantities.QuantityDict(grid)

    assert len(q) == 0

def test_compatible():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3,1,2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(1,1,1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(1,1,2), seed=2)
    tensor4 = utilities.get_random_tensor(size=(2,1,2), seed=3)
    tensor5 = utilities.get_random_tensor(size=(3,3,2), seed=4)

    assert grid_quantities.compatible(tensor1, grid)
    assert grid_quantities.compatible(tensor2, grid)
    assert grid_quantities.compatible(tensor3, grid)
    assert not grid_quantities.compatible(tensor4, grid)
    assert not grid_quantities.compatible(tensor5, grid)

def test_is_singleton_dimension():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1,1,2), seed=0)
    tensor2 = utilities.get_random_tensor(size=(2,1,2), seed=1)

    assert grid_quantities.is_singleton_dimension('a', tensor1, grid)
    assert grid_quantities.is_singleton_dimension('b', tensor1, grid)
    assert not grid_quantities.is_singleton_dimension('c', tensor1, grid)

    # Must be compatible
    with pytest.raises(Exception) as e_info:
        grid_quantities.is_singleton_dimension('b', tensor2, grid)

def test_might_depend_on():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(1,1,2), seed=0)

    assert not grid_quantities.might_depend_on('a', tensor, grid)
    assert grid_quantities.might_depend_on('b', tensor, grid)
    assert grid_quantities.might_depend_on('c', tensor, grid)

def test_sum_dimension():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(3,1,2), seed=0)

    summed_tensor = grid_quantities.sum_dimension('c', tensor, grid)

    assert summed_tensor.size() == (3,1,1)
    assert torch.allclose(summed_tensor[:,:,0], tensor[:,:,0] + tensor[:,:,1])

def test_sum_dimension_singleton_grid():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(3,1,2), seed=0)

    summed_tensor = grid_quantities.sum_dimension('b', tensor, grid)

    assert torch.equal(summed_tensor, tensor)

def test_sum_dimension_singleton_dimension():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(1,1,2), seed=0)

    summed_tensor = grid_quantities.sum_dimension('a', tensor, grid)

    assert torch.allclose(summed_tensor, 3*tensor)

def test_expand_all_dims():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1,1,1), seed=0)
    tensor2 = utilities.get_random_tensor(size=(3,1,1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(3,1,2), seed=2)

    tensor1_expanded = grid_quantities.expand_all_dims(tensor1, grid)
    tensor2_expanded = grid_quantities.expand_all_dims(tensor2, grid)
    tensor3_expanded = grid_quantities.expand_all_dims(tensor3, grid)

    assert tensor1_expanded.size() == (3,1,2)
    assert torch.all(tensor1_expanded == tensor1)
    assert tensor2_expanded.size() == (3,1,2)
    assert torch.all(tensor2_expanded == tensor2)
    assert torch.equal(tensor3_expanded, tensor3)

def test_squeeze_to():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(1,1,1), seed=0)
    tensor2 = utilities.get_random_tensor(size=(3,1,1), seed=1)
    tensor3 = utilities.get_random_tensor(size=(3,1,2), seed=2)

    tensor1_squeezed = grid_quantities.squeeze_to(['b'], tensor1, grid)
    tensor1_completely_squeezed = grid_quantities.squeeze_to([], tensor1, grid)
    tensor2_squeezed = grid_quantities.squeeze_to(['a'], tensor2, grid)
    tensor3_squeezed = grid_quantities.squeeze_to(['a', 'c'], tensor3, grid)
    tensor3_not_squeezed = grid_quantities.squeeze_to(['a', 'b', 'c'], tensor3, grid)

    assert tensor1_squeezed.size() == (1,)
    assert torch.equal(tensor1_squeezed, tensor1[0,:,0])
    assert tensor1_completely_squeezed.size() == ()
    assert tensor2_squeezed.size() == (3,)
    assert torch.equal(tensor2_squeezed, tensor2[:,0,0])
    assert tensor3_squeezed.size() == (3,2)
    assert torch.equal(tensor3_squeezed, tensor3[:,0,:])
    assert tensor3_not_squeezed.size() == (3,1,2)
    assert torch.equal(tensor3_not_squeezed, tensor3)

    # The squeezed tensor must possibly depend on all dimensions
    with pytest.raises(Exception) as e_info:
        grid_quantities.squeeze_to(['b', 'c'], tensor1, grid)

def test_squeeze_too_much():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3,1,1), seed=1)
    tensor2 = utilities.get_random_tensor(size=(3,1,2), seed=2)

    # Cannot squeeze in a dimension the tensor depends on
    with pytest.raises(Exception) as e_info:
        grid_quantities.squeeze_to(['b', 'c'], tensor2, grid)
    with pytest.raises(Exception) as e_info:
        grid_quantities.squeeze_to([], tensor1, grid)

def test_unsqueeze_to():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor1 = utilities.get_random_tensor(size=(3,1,1), seed=1)
    tensor2 = utilities.get_random_tensor(size=(3,1,2), seed=2)
    tensor3 = utilities.get_random_tensor(size=(1,), seed=3)
    tensor4 = utilities.get_random_tensor(size=(3,2), seed=4)
    tensor5 = utilities.get_random_tensor(size=(2,1,3), seed=4)

    tensor2_unsqueezed = grid_quantities.unsqueeze_to(grid, tensor2, ('a','b','c'))
    tensor3_unsqueezed = grid_quantities.unsqueeze_to(grid, tensor3, ('b',))
    tensor4_unsqueezed = grid_quantities.unsqueeze_to(grid, tensor4, ('a','c'))

    assert torch.equal(tensor2, tensor2_unsqueezed)
    assert tensor3_unsqueezed.size() == (1,1,1)
    assert torch.equal(tensor3_unsqueezed[0,:,0], tensor3)
    assert tensor4_unsqueezed.size() == (3,1,2)
    assert torch.equal(tensor4_unsqueezed[:,0,:], tensor4)

    # Input must depend on all dimensions
    with pytest.raises(Exception) as e_info:
        grid_quantities.unsqueeze_to(grid, tensor1, ('a','b','c'))
    with pytest.raises(Exception) as e_info:
        grid_quantities.unsqueeze_to(grid, tensor3, ('c'))

    # Input needs to be ordered
    with pytest.raises(Exception) as e_info:
        grid_quantities.unsqueeze_to(grid, tensor5, ('c', 'b', 'a'))

def test_unsqueeze_to_from_constant():
    sizes = {'a': 3, 'b': 1, 'c': 2}
    grid = utilities.get_random_grid(sizes, seed=0)
    tensor = utilities.get_random_tensor(size=(), seed=1)

    tensor_unsqueezed = grid_quantities.unsqueeze_to(grid, tensor, ())

    assert tensor_unsqueezed.size() == (1,1,1)
    assert tensor_unsqueezed.item() == tensor.item()

def test_combine_quantity():
    parent_sizes = {'a': 3, 'b': 1, 'c': 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    indices_dicts = [
        {'a': [2,0], 'c': [1]},
        {'a': [2], 'c': [0]},
        {'a': [0], 'c': [0]},
        {'a': [], 'c': [0,1]},
        {'a': [], 'c': []},
        {'a': [1], 'c': [0,1]},
    ]
    subgrids = [grid_quantities.Subgrid(parent_grid, indices_dict, copy_all=False)
                for indices_dict in indices_dicts]
    quantity = utilities.get_random_tensor(size=(3,1,2), seed=1)
    sub_quantities = [grid_quantities.restrict(quantity, subgrid)
                      for subgrid in subgrids]

    combined_quantity = grid_quantities.combine_quantity(
        sub_quantities,
        subgrids,
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity)

    # The whole grid must be covered
    with pytest.raises(Exception) as e_info:
        grid_quantities.combine_quantity(
            sub_quantities[:-1],
            subgrids[:-1],
            parent_grid,
        )

    # No double coverage
    with pytest.raises(Exception) as e_info:
        grid_quantities.combine_quantity(
            sub_quantities + [sub_quantities[1]],
            subgrids + [subgrids[1]],
            parent_grid,
        )

def test_combine_quantity_empty_grid():
    parent_sizes = {'a': 3, 'b': 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grid_quantities.Subgrid(
        parent_grid,
        {'a': [0,1,2], 'b': [0,1]},
        copy_all=False,
    )
    subgrid_empty = grid_quantities.Subgrid(
        parent_grid,
        {'a': [], 'b': []},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(3,2), seed=1)
    quantity_empty = utilities.get_random_tensor(size=(0,0), seed=2)

    combined_quantity = grid_quantities.combine_quantity(
        [quantity_full, quantity_empty],
        [subgrid_full, subgrid_empty],
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity_full)

def test_combine_quantity_expands_sliced_singleton_dimension():
    parent_sizes = {'a': 3, 'b': 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grid_quantities.Subgrid(
        parent_grid,
        {'a': [0,1,2], 'b': [0,1]},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(1,2), seed=1)

    combined_quantity = grid_quantities.combine_quantity(
        [quantity_full],
        [subgrid_full],
        parent_grid,
    )

    assert combined_quantity.size() == (3,2)
    assert torch.all(combined_quantity == quantity_full)

def test_combine_quantity_leaves_unsliced_singleton_dimension():
    parent_sizes = {'a': 3, 'b': 2}
    parent_grid = utilities.get_random_grid(parent_sizes, seed=0)
    subgrid_full = grid_quantities.Subgrid(
        parent_grid,
        {'b': [0,1]},
        copy_all=False,
    )
    quantity_full = utilities.get_random_tensor(size=(1,2), seed=1)

    combined_quantity = grid_quantities.combine_quantity(
        [quantity_full],
        [subgrid_full],
        parent_grid,
    )

    assert torch.equal(combined_quantity, quantity_full)
