from typing import Callable, Optional
import copy
import itertools
import os
import time
import numpy as np
import torch

from .mathematics import remove_duplicates
from .io import get_next_parameters_index, get_parameters_path
from .batching import Batcher, get_qs
from .model import MultiModel, set_requires_grad_quantities, set_requires_grad_models


def get_numpy_losses(losses, loss_aggregate_function):
    """
    losses[loss_name]: torch.Tensor
    """

    numpy_losses = np.array([loss.item() for loss in losses.values()])
    numpy_losses = np.append(numpy_losses, [loss_aggregate_function(losses.values()).item()])

    return numpy_losses


class Trainer:
    def __init__(
            self,
            *,
            models: list[MultiModel],
            batchers_training: dict[str,Batcher],
            batchers_validation: dict[str,Batcher],
            used_losses: dict[str,list[str]],
            trained_models_labels: list[str],
            Optimizer,
            optimizer_kwargs: dict,
            Scheduler = None,
            scheduler_kwargs: Optional[dict] = None,
            saved_parameters_index: int,
            save_optimizer: bool,
            loss_aggregate_function: Optional[Callable] = None,
            name: str,
        ):
        """
        The loss models referred to by `used_losses` should have a
        `with_grad` keyword, it will be controlled by the trainer.
        `loss_aggregate_function(losses)` returns the total loss given the
        iterable `losses`.
        """

        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        if loss_aggregate_function is None:
            loss_aggregate_function = lambda losses: sum(losses)

        self.models = models
        self.batchers_training = batchers_training
        self.batchers_validation = batchers_validation
        self.used_losses = used_losses
        self.trained_models_labels = trained_models_labels
        self.saved_parameters_index = saved_parameters_index
        self.save_optimizer = save_optimizer
        self.loss_aggregate_function = loss_aggregate_function
        self.name = name

        self.trained_models = [model for model in models
                                     if model.name in trained_models_labels]
        self.all_parameters = list(itertools.chain.from_iterable(
            [model.parameters for model in self.trained_models],
        ))
        self.all_parameters = remove_duplicates(self.all_parameters)
        self.optimizer = Optimizer(self.all_parameters, **optimizer_kwargs)
        self.initial_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        self.scheduler = (None if Scheduler is None
                          else Scheduler(self.optimizer, **scheduler_kwargs))

        self.used_losses_list = list(itertools.chain.from_iterable(used_losses.values()))
        self.n_losses = len(self.used_losses_list)
        self.training_loss_history = np.zeros((0, self.n_losses+1))
        self.training_loss_times = np.zeros((0,))
        self.validation_loss_history = np.zeros((0, self.n_losses+1))
        self.validation_loss_times = np.zeros((0,))
        self.training_start_time = None
        self.min_validation_loss = None
        self.loss_names = [] # In the same order as the training histories
        for losses in self.used_losses_list:
            self.loss_names.append(losses)
        self.loss_names += ['Total']

        # PROFILING
        self.evaluation_times = {}

    def train(
            self,
            *,
            report_each = None,
            max_n_steps = None,
            max_time = None,
            min_loss = None,
        ):

        print(f'\n\nTraining {self.name}\n')

        self.training_start_time = time.perf_counter()
        step_index = 0

        while True:
            stop = False
            if max_n_steps is not None and step_index >= max_n_steps:
                print(f'Step {max_n_steps} reached, stopping')
                stop = True
            time_passed = time.perf_counter() - self.training_start_time
            if (max_time is not None
                and time_passed >= max_time):
                print(f'{time_passed:.1f}s passed, stopping')
                stop = True

            if step_index % report_each == 0 or stop:
                self.validate(step_index, max_n_steps, save_if_best = True)

            if (min_loss is not None
                and self.validation_loss_history[-1][-1] <= min_loss):
                print(f'Validation loss {self.validation_loss_history[-1][-1]} reached, stopping')
                stop = True

            if stop:
                break

            step_index += 1
            self.step()

            if not torch.isfinite(torch.tensor(self.training_loss_history[-1][-1])):
                print(f'Loss became nonfinite in step {step_index}, resetting the optimizer...')
                self.load(
                    self.saved_parameters_index,
                    load_optimizer=False,
                    load_scheduler=False,
                )
                self.optimizer.load_state_dict(copy.deepcopy(self.initial_optimizer_state_dict))


    def validate(self, step_index, max_n_steps, *, save_if_best):
        for model in self.models:
            model.set_eval()

        max_n_steps_string = '  -  ' if max_n_steps is None else f'{max_n_steps:>5d}'
        print(f'[{step_index:>5d}/{max_n_steps_string}]')
        self.get_validation_losses(save_if_best = save_if_best)
        if not self.scheduler is None:
            self.scheduler.step(self.validation_loss_history[-1][-1])

    def step(self):
        for model in self.models:
            model.set_train()

        if type(self.optimizer) is torch.optim.LBFGS:
            self.optimizer.step(self.closure)
        else:
            self.closure()
            self.optimizer.step()


    def _extract_losses(self, qs):
        losses = {}
        for grid_name, loss_names in self.used_losses.items():
            q = qs[grid_name]
            for loss_name in loss_names:
                losses[loss_name] = q[loss_name].mean()

        return losses


    def set_losses_kwargs(self, **kwargs):
        for multi_model in self.models:
            if multi_model.name not in self.used_losses_list:
                continue
            for model in multi_model.models:
                for label, arg in kwargs.items():
                    model.kwargs[label] = arg

    def get_extended_qs(self, *, for_training):
        # TODO: Support for combining batches
        batchers = self.batchers_training if for_training else self.batchers_validation
        qs = get_qs(batchers)
        for model in self.models:
            #print(f"Evaluating '{model.name}'") # DEBUG
            eval_start_time = time.perf_counter_ns() # PROFILING

            # Provide q_full if necessary
            for submodel in model.models:
                if hasattr(submodel, 'kwargs') and 'q_full' in submodel.kwargs:
                    submodel.kwargs['q_full'] = batchers[model.grid_name].q_full
            model.apply(qs)

            # PROFILING
            eval_time = time.perf_counter_ns() - eval_start_time
            label = f'{model.name} on {model.grid_name}'
            if not label in self.evaluation_times:
                self.evaluation_times[label] = 0
            self.evaluation_times[label] += eval_time

        return qs


    def get_training_losses(self):
        set_requires_grad_models(True, self.trained_models)
        self.set_losses_kwargs(with_grad=True)
        qs = self.get_extended_qs(for_training = True)
        losses = self._extract_losses(qs)

        # History
        numpy_losses = get_numpy_losses(losses, self.loss_aggregate_function)
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
        set_requires_grad_models(False, self.trained_models)
        self.set_losses_kwargs(with_grad=False)
        qs = self.get_extended_qs(for_training = False)
        losses = self._extract_losses(qs)

        # History
        numpy_losses = get_numpy_losses(losses, self.loss_aggregate_function)
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

        self.optimizer.zero_grad()
        losses = self.get_training_losses()
        loss = self.loss_aggregate_function(losses.values())
        loss.backward(inputs=self.all_parameters)

        return loss


    def save(self):
        path = get_parameters_path(self.saved_parameters_index)
        os.makedirs(path, exist_ok=True)

        model_parameters_dict = {}
        for model in self.models:
            if model.name not in self.trained_models_labels:
                continue
            model_parameters_dict[model.name] = model.parameters
        save_dict = {
            'model_parameters_dict': model_parameters_dict,
        }
        if self.save_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        if not self.scheduler is None:
            save_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(save_dict, path + self.name + '.pth')


    def load(self, parameters_index, *, load_optimizer: bool, load_scheduler: bool):
        if parameters_index is None:
            return

        path = get_parameters_path(parameters_index)
        save_dict = torch.load(path + self.name + '.pth')
        for model in self.models:
            if model.name not in self.trained_models_labels:
                continue
            model.replace_parameters(
                save_dict['model_parameters_dict'][model.name]
            )
        if load_optimizer:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        if load_scheduler:
            self.scheduler.load_state_dict(save_dict['scheduler_state_dict'])


    def load_models(self, loaded_models):
        assert len(loaded_models) == len(self.models)
        for model, loaded_model in zip(self.models, loaded_models):
            assert model.name == loaded_model.name
            model.replace_parameters(loaded_model.parameters)

