from typing import Optional
import os
import torch

def get_weights_path(index: int, output_label: str, data_path: str = ''):
    return (data_path + f'weights/weights{index:04d}_'
            + output_label
            + '.pth')

def get_next_weights_index(output_labels, data_path: str = ''):
    index = 1
    while any(os.path.exists(get_weights_path(index, output_label, data_path))
              for output_label in output_labels):
        index += 1

    return index
