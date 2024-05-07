# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os

def get_parameters_path(index: int, data_path: str = ''):
    return data_path + f'parameters/{index:04d}/'

def get_next_parameters_index(data_path: str = ''):
    index = 1
    while os.path.exists(get_parameters_path(index, data_path)):
        index += 1

    return index
