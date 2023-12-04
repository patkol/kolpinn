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

def load_model_parameters(
        model_parameters: dict[str,torch.tensor],
        loaded_weights_index: Optional[int],
        data_path: str = '',
    ):
    if loaded_weights_index is None:
        return

    for model_parameter_name, model_parameter in model_parameters.items():
        path = get_weights_path(loaded_weights_index, model_parameter_name, data_path)
        with open(path, 'r') as f:
            value = eval(f.read())
        model_parameters[model_parameter_name] = torch.tensor(
            value,
            dtype=model_parameter.dtype,
            requires_grad=True,
        )
        print('Loaded parameter "', model_parameter_name, '" at ' + path)
