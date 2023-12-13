from typing import Callable, Optional
import os
import time
import numpy as np
import torch

from .mathematics import remove_duplicates
from .io import get_next_parameters_index, get_parameters_path
from .batching import Batcher
from .model import Model, get_extended_qs


def get_numpy_losses(losses):
    """
    losses[loss_name] = Quantity
    """

    numpy_losses = np.array([loss.values.item() for loss in losses.values()])
    numpy_losses = np.append(numpy_losses, [np.sum(numpy_losses)])

    return numpy_losses


class Trainer:
    def __init__(
            self,
            *,
            models_dict: dict[str,dict[str,Model]],
            batchers_training: dict[str,Batcher],
            batchers_validation: dict[str,Batcher],
            used_losses: dict[str,list[str]],
            quantities_requiring_grad_dict: dict,
            Optimizer,
            learn_rate: float,
            saved_parameters_index: int,
            name: str,
        ):
        """
        The loss models referred to by used_losses should have a
        with_grad keyword, it will be controlled by the trainer.
        """

        self.models_dict = models_dict
        self.batchers_training = batchers_training
        self.batchers_validation = batchers_validation
        self.used_losses = used_losses
        self.quantities_requiring_grad_dict = quantities_requiring_grad_dict
        self.saved_parameters_index = saved_parameters_index
        self.name = name

        self.batcher_names = models_dict.keys()
        assert set(self.batchers_training) == set(self.batcher_names)
        assert set(self.batchers_validation) == set(self.batcher_names)

        all_parameters = []
        for batcher_name in self.batcher_names:
            for model in models_dict[batcher_name].values():
                all_parameters += model.parameters
        all_parameters = remove_duplicates(all_parameters)
        self.optimizer = Optimizer(all_parameters, lr = learn_rate)

        self.n_losses = sum(len(losses) for losses in used_losses.values())
        self.training_loss_history = np.zeros((0, self.n_losses+1))
        self.training_loss_times = np.zeros((0,))
        self.validation_loss_history = np.zeros((0, self.n_losses+1))
        self.validation_loss_times = np.zeros((0,))
        self.training_start_time = None
        self.min_validation_loss = None
        self.loss_names = [] # In the same order as the training histories
        for losses in used_losses.values():
            self.loss_names += losses
        self.loss_names += ['Total']

        print('saved_parameters_index =', self.saved_parameters_index)

    def train(self, n_steps, report_each, *, max_time = None):
        if max_time is None:
            max_time = float('inf')
        self.training_start_time = time.perf_counter()
        print(f"[{0:>5d}/{n_steps:>5d}]")
        self.get_validation_losses(save_if_best = n_steps > 0)

        for batcher_name in self.batcher_names:
            for model in self.models_dict[batcher_name].values():
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


    def _extract_losses(self, qs):
        losses = {}
        for batcher_name, batcher in self.batchers_training.items():
            q = qs[batcher_name]
            for loss_name in self.used_losses[batcher_name]:
                losses[loss_name] = q[loss_name].mean()

        return losses


    def set_losses_kwargs(self, **kwargs):
        for batcher_name in self.batcher_names:
            for loss_name in self.used_losses[batcher_name]:
                for label, arg in kwargs.items():
                    self.models_dict[batcher_name][loss_name].kwargs[label] = arg

    def get_extended_qs(self, *, for_training=False):
        qs = get_extended_qs(
            self.batchers_training if for_training else self.batchers_validation,
            models_dict = self.models_dict,
            models_require_grad = for_training, # OPTIM: not always the case for LBFGS
            quantities_requiring_grad_dict = self.quantities_requiring_grad_dict,
            full_grid = not for_training,
        )

        return qs


    def get_training_losses(self):
        self.set_losses_kwargs(with_grad=True)
        qs = self.get_extended_qs(for_training = True)
        losses = self._extract_losses(qs)

        # History
        numpy_losses = get_numpy_losses(losses)
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
        self.set_losses_kwargs(with_grad=False)
        qs = self.get_extended_qs(for_training = False)
        losses = self._extract_losses(qs)

        # History
        numpy_losses = get_numpy_losses(losses)
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

        model_parameters_dict = {}
        for batcher_name in self.batcher_names:
            model_parameters_dict[batcher_name] = {}
            for model_name, model in self.models_dict[batcher_name].items():
                model_parameters_dict[batcher_name][model_name] = model.parameters
        save_dict = {
            'model_parameters_dict': model_parameters_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        torch.save(save_dict, path + self.name + '.pth')


    def load(self, parameters_index):
        if parameters_index is None:
            return

        path = get_parameters_path(parameters_index)
        save_dict = torch.load(path + self.name + '.pth')
        for batcher_name in self.batcher_names:
            for model_name, model in self.models_dict[batcher_name].items():
                self.models_dict[batcher_name][model_name].replace_parameters(
                    save_dict['model_parameters_dict'][batcher_name][model_name]
                )
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
