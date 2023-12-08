from typing import Callable, Optional
import os
import time
import numpy as np
import torch

from .io import get_next_parameters_index, get_parameters_path
from .loss import get_batch_losses, get_full_losses


def get_mean_numpy_losses(losses):
    """
    losses[loss_name] = Quantity
    """

    numpy_losses = np.array([loss.mean().values.item()
                             for loss in losses.values()])
    numpy_losses = np.append(numpy_losses, [np.sum(numpy_losses)])

    return numpy_losses


class Trainer:
    def __init__(
            self,
            models,
            batchers_training,
            batchers_validation,
            loss_functions: dict[str,dict[str,Callable]],
            quantities_requiring_grad_dict: dict,
            Optimizer,
            learn_rate,
            *,
            saved_parameters_index,
            name,
        ):

        self.models = models
        self.batchers_training = batchers_training
        self.batchers_validation = batchers_validation
        self.loss_functions = loss_functions
        self.quantities_requiring_grad_dict = quantities_requiring_grad_dict
        self.saved_parameters_index = saved_parameters_index
        self.name = name

        all_parameters = []
        for model in models.values():
            all_parameters += model.parameters
        self.optimizer = Optimizer(all_parameters, lr = learn_rate)

        self.n_losses = sum(len(loss_fn) for loss_fn in loss_functions.values())
        self.training_loss_history = np.zeros((0, self.n_losses+1))
        self.training_loss_times = np.zeros((0,))
        self.validation_loss_history = np.zeros((0, self.n_losses+1))
        self.validation_loss_times = np.zeros((0,))
        self.training_start_time = None
        self.min_validation_loss = None
        self.loss_names = [] # In the same order as the training histories
        for batcher_loss_functions in loss_functions.values():
            self.loss_names += batcher_loss_functions.keys()
        self.loss_names += ['Total']

        print('saved_parameters_index =', self.saved_parameters_index)

    def train(self, n_steps, report_each, *, max_time = None):
        if max_time is None:
            max_time = float('inf')
        self.training_start_time = time.perf_counter()
        print(f"[{0:>5d}/{n_steps:>5d}]")
        self.get_validation_losses(save_if_best = n_steps > 0)

        for model in self.models.values():
            model.set_train()

        for step_index in range(1, n_steps+1):
            self.step()
            if step_index % report_each == 0 or step_index == n_steps:
                print(f"[{step_index:>5d}/{n_steps:>5d}]")
                self.get_validation_losses(save_if_best = True)
                if time.perf_counter() - self.training_start_time >= max_time:
                    print("Stopping on time")
                    break

    def step(self):
        if type(self.optimizer) is torch.optim.LBFGS:
            self.optimizer.step(self.closure)
        else:
            self.closure()
            self.optimizer.step()

    def get_training_losses(self):
        losses = get_batch_losses(
            self.models,
            self.batchers_training,
            self.loss_functions,
            self.quantities_requiring_grad_dict,
            models_require_grad = True, # TODO not always the case for LBFGS
        )

        # History
        numpy_losses = get_mean_numpy_losses(losses)
        self.training_loss_history = np.append(
            self.training_loss_history,
            [numpy_losses],
            axis=0,
        )
        execution_time = time.perf_counter() - self.training_start_time
        self.training_loss_times = np.append(
            self.training_loss_times,
            [execution_time],
        )

        return losses

    def get_validation_losses(self, save_if_best):
        losses = get_full_losses(
            self.models,
            self.batchers_validation,
            self.loss_functions,
            self.quantities_requiring_grad_dict,
            models_require_grad = False,
        )

        # History
        numpy_losses = get_mean_numpy_losses(losses)
        self.validation_loss_history = np.append(
            self.validation_loss_history,
            [numpy_losses],
            axis=0,
        )
        execution_time = time.perf_counter() - self.training_start_time
        self.validation_loss_times = np.append(
            self.validation_loss_times,
            [execution_time],
        )

        # Print
        training_losses = (self.training_loss_history[-1,:]
                           if len(self.training_loss_history) > 0
                           else float('nan') * np.ones(self.n_losses+1))
        training_loss_time = (self.training_loss_times[-1]
                              if len(self.training_loss_times) > 0
                              else float('nan'))
        print(f"Elapsed time: {execution_time:>4f} ({training_loss_time:>4f}) s")
        for i, loss_name in enumerate(losses.keys()):
            print(f"{loss_name}: {numpy_losses[i]:>7f} ({training_losses[i]:>7f})")
        print(f"Total loss: {numpy_losses[-1]:>7f} ({training_losses[-1]:>7f})")
        print()

        # Save
        if save_if_best and (self.min_validation_loss == None
                             or numpy_losses[-1] < self.min_validation_loss):
            # Saving at step 0 as well to reserve the `saved_parameters_index`
            self.save()
            self.min_validation_loss = numpy_losses[-1]

        return losses

    def closure(self):
        """
        Calculation of the loss used for training
        """

        self.optimizer.zero_grad() # OPTIM: not always necessary for lbfgs
        losses = self.get_training_losses()
        loss_tensors = [loss.values.reshape([]) for loss in losses.values()]
        loss = sum(loss_tensors)
        loss.backward() # OPTIM: not always necessary for lbfgs

        return loss

    def save(self):
        path = get_parameters_path(self.saved_parameters_index)
        os.makedirs(path, exist_ok=True)
        save_dict = {
            'models': dict((model_name, model.parameters)
                           for model_name, model in self.models.items()),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_dict, path + self.name + '.pth')

    def load(self, parameters_index):
        if parameters_index is None:
            return

        path = get_parameters_path(parameters_index)
        save_dict = torch.load(path + self.name + '.pth')
        for model_name, model in self.models.items():
            model.replace_parameters(save_dict['models'][model_name])
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
