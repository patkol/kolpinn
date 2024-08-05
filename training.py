# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import dataclasses
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable, Optional, Any, Tuple, Dict
import copy
import itertools
import os
import time
import numpy as np
import torch

from . import mathematics
from .mathematics import remove_duplicates
from .storage import get_parameters_path
from .quantities import QuantityDict
from .model import MultiModel, set_requires_grad_models


@dataclass
class TrainerConfig:
    # Inputs: qs, randomize
    get_batched_qs: Callable[[Dict[str, QuantityDict], bool], Dict[str, QuantityDict]]
    # qs[grid_name][loss_quantities[grid_name]] will be used as losses
    loss_quantities: Dict[str, Sequence[str]]
    loss_aggregate_function: Callable[[Sequence], Any]
    saved_parameters_index: int
    save_optimizer: bool
    max_n_steps: Optional[int] = None
    max_time: Optional[float] = None
    min_loss: Optional[float] = None
    optimizer_reset_tol: float = float("inf")


@dataclass
class TrainerState:
    const_qs: Dict
    trained_models: Sequence[MultiModel]
    dependent_models: Sequence[MultiModel]
    optimizer: torch.optim.Optimizer
    scheduler: Any = None
    n_steps: int = 0
    best_validation_loss: float = float("inf")
    n_loaded_current_parameters: int = 0
    training_start_time: Optional[float] = None
    evaluation_times: Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclass
class TrainerHistory:
    losses: np.ndarray
    times: np.ndarray


def get_empty_trainer_history(config: TrainerConfig):
    loss_quantities_chain = mathematics.get_chained_values(config.loss_quantities)
    n_losses = len(list(loss_quantities_chain))
    return TrainerHistory(
        losses=np.zeros((0, n_losses + 1)),
        times=np.zeros((0,)),
    )


@dataclass
class Trainer:
    state: TrainerState
    config: TrainerConfig
    training_history: TrainerHistory = dataclasses.field(init=False)
    validation_history: TrainerHistory = dataclasses.field(init=False)

    def __post_init__(self):
        self.training_history = get_empty_trainer_history(self.config)
        self.validation_history = get_empty_trainer_history(self.config)


def get_all_parameters(models: Sequence[MultiModel]) -> Sequence[torch.Tensor]:
    all_parameters = list(
        itertools.chain.from_iterable([model.parameters for model in models])
    )
    all_parameters = remove_duplicates(all_parameters)

    return all_parameters


def get_optimizer(
    Optimizer: type,
    trained_models: Sequence[MultiModel],
    **kwargs,
) -> torch.optim.Optimizer:
    all_parameters = get_all_parameters(trained_models)
    return Optimizer(all_parameters, **kwargs)


def get_scheduler(
    Scheduler: Optional[type],
    optimizer: torch.optim.Optimizer,
    **kwargs,
):
    if Scheduler is None:
        return None

    return Scheduler(optimizer, **kwargs)


def stop_training(trainer: Trainer) -> Tuple[bool, Optional[str]]:
    """
    Return (whether to stop, reason)
    """
    if (
        trainer.config.max_n_steps is not None
        and trainer.state.n_steps >= trainer.config.max_n_steps
    ):
        return True, f"{trainer.state.n_steps} training steps executed"

    assert trainer.state.training_start_time is not None
    time_passed = time.perf_counter() - trainer.state.training_start_time
    max_time = trainer.config.max_time
    if max_time is not None and time_passed >= max_time:
        return True, f"{time_passed:.1f}s >= {max_time:.1f} passed"

    validation_loss = trainer.validation_history.losses[-1, -1]
    min_loss = trainer.config.min_loss
    if min_loss is not None and validation_loss <= min_loss:
        return True, f"Validation loss {validation_loss} <= {min_loss} reached"

    return False, None


def reset_to_best(trainer: Trainer) -> Tuple[bool, Optional[str]]:
    """
    Return (whether to reset, reason)
    """
    current_training_loss = trainer.training_history.losses[-1, -1]
    if not np.isfinite(current_training_loss):
        return True, f"Loss became nonfinite ({current_training_loss})"

    assert trainer.state.best_validation_loss is not None
    max_training_loss = (
        trainer.config.optimizer_reset_tol * trainer.state.best_validation_loss
    )
    if current_training_loss > max_training_loss:
        factor = round(current_training_loss / trainer.state.best_validation_loss)
        return (
            True,
            f"Loss became {factor}x > {trainer.config.optimizer_reset_tol}x larger than best validation loss",
        )

    return False, None


def save(trainer: Trainer):
    path = get_parameters_path(trainer.config.saved_parameters_index)
    os.makedirs(path, exist_ok=True)

    model_parameters_dict: Dict[str, Sequence[torch.Tensor]] = {}
    for model in trainer.state.trained_models:
        model_parameters_dict[model.name] = model.parameters
    save_dict = {
        "model_parameters_dict": model_parameters_dict,
    }
    if trainer.config.save_optimizer:
        save_dict["optimizer_state_dict"] = trainer.state.optimizer.state_dict()
    if trainer.state.scheduler is not None:
        save_dict["scheduler_state_dict"] = trainer.state.scheduler.state_dict()

    torch.save(save_dict, path + "all.pth")
    trainer.state.n_loaded_current_parameters = 0


def load(
    parameters_index: Optional[int],
    trainer: Trainer,
    *,
    load_optimizer: bool,
    load_scheduler: bool,
) -> None:
    if parameters_index is None:
        return

    path = get_parameters_path(parameters_index)
    save_dict = torch.load(path + "all.pth")
    for model in trainer.state.trained_models:
        model.replace_parameters(save_dict["model_parameters_dict"][model.name])
    if load_optimizer:
        trainer.state.optimizer.load_state_dict(save_dict["optimizer_state_dict"])
    if load_scheduler:
        trainer.state.scheduler.load_state_dict(save_dict["scheduler_state_dict"])

    trainer.state.n_loaded_current_parameters += 1


def get_extended_qs(
    state: TrainerState,
    const_qs: Optional[Dict[str, QuantityDict]] = None,
):
    """
    Get a new qs including the parameter-dependent quantities, starting from
    const_qs.
    const_qs can lie on batched grids and thus differ from state.const_qs.
    """

    if const_qs is None:
        const_qs = state.const_qs

    # Copy all `QuantityDict`s
    qs = dict((label, copy.copy(q)) for label, q in const_qs.items())
    models = itertools.chain.from_iterable(
        [state.trained_models, state.dependent_models]
    )
    for model in models:
        print(f"Evaluating '{model.name}'")  # DEBUG

        # Provide qs_full if necessary
        if "qs_full" in model.kwargs:
            model.kwargs["qs_full"] = state.const_qs

        eval_start_time = time.perf_counter_ns()  # PROFILING

        model.apply(qs)

        # PROFILING
        eval_time = time.perf_counter_ns() - eval_start_time
        if model.name not in state.evaluation_times:
            state.evaluation_times[model.name] = 0
        state.evaluation_times[model.name] += eval_time

    return qs


def _extract_losses(
    qs: Dict[str, QuantityDict],
    config: TrainerConfig,
) -> Dict[str, torch.Tensor]:
    losses = {}
    for grid_name, loss_names in config.loss_quantities.items():
        q = qs[grid_name]
        for loss_name in loss_names:
            assert loss_name not in losses
            losses[loss_name] = q[loss_name].mean()

    return losses


def log_losses(
    losses: Dict[str, torch.Tensor],
    history: TrainerHistory,
    time_passed: float,
) -> None:
    numpy_losses = np.array([loss.item() for loss in losses.values()])
    history.losses = np.append(history.losses, [numpy_losses], axis=0)
    history.times = np.append(history.times, [time_passed])


def get_losses(trainer: Trainer, *, for_training: bool = False):
    """
    losses['Total'] is the total loss.
    """
    for model in itertools.chain.from_iterable(
        [trainer.state.trained_models, trainer.state.dependent_models]
    ):
        model.set_train() if for_training else model.set_eval()

    set_requires_grad_models(for_training, trainer.state.trained_models)
    const_qs_full = trainer.state.const_qs
    const_qs_batched = trainer.config.get_batched_qs(const_qs_full, for_training)
    qs = get_extended_qs(trainer.state, const_qs_batched)

    losses = _extract_losses(qs, trainer.config)
    losses["Total"] = trainer.config.loss_aggregate_function(list(losses.values()))

    history = trainer.training_history if for_training else trainer.validation_history
    assert trainer.state.training_start_time is not None
    time_passed = time.perf_counter() - trainer.state.training_start_time
    log_losses(losses, history, time_passed)

    return losses


def print_progress(trainer: Trainer):
    max_n_steps_string = (
        "  -  "
        if trainer.config.max_n_steps is None
        else f"{trainer.config.max_n_steps:>5d}"
    )
    print(f"[{trainer.state.n_steps:>5d}/{max_n_steps_string}]")

    n_losses = trainer.training_history.losses.shape[1]
    validation_losses = trainer.validation_history.losses[-1, :]
    validation_loss_time = trainer.validation_history.times[-1]
    training_losses = (
        trainer.training_history.losses[-1, :]
        if len(trainer.training_history.losses) > 0
        else float("nan") * np.ones(n_losses)
    )
    training_loss_time = (
        trainer.training_history.times[-1]
        if len(trainer.training_history.times) > 0
        else float("nan")
    )

    print(f"Elapsed time: {validation_loss_time:>4f} ({training_loss_time:>4f}) s")
    loss_quantities_chain = mathematics.get_chained_values(
        trainer.config.loss_quantities
    )
    for i, loss_name in enumerate(loss_quantities_chain):
        print(f"{loss_name}: {validation_losses[i]:>7f} ({training_losses[i]:>7f})")
    print(f"Total loss: {validation_losses[-1]:>7f} ({training_losses[-1]:>7f})")
    print()


def step(trainer: Trainer):
    def closure():
        """
        Calculation of the loss used for training
        """

        trainer.state.optimizer.zero_grad()
        loss = get_losses(trainer, for_training=True)["Total"]
        all_parameters = get_all_parameters(trainer.state.trained_models)
        loss.backward(inputs=all_parameters)

        return loss

    if type(trainer.state.optimizer) is torch.optim.LBFGS:
        trainer.state.optimizer.step(closure)
    else:
        closure()
        trainer.state.optimizer.step()

    trainer.state.n_steps += 1


def train(
    trainer: Trainer,
    *,
    report_each: int,
    save_if_best: bool,
):
    initial_optimizer_state_dict = copy.deepcopy(trainer.state.optimizer.state_dict())
    trainer.state.training_start_time = time.perf_counter()
    validated_current_state = False

    while True:
        if trainer.state.n_steps % report_each == 0:
            # Validate
            get_losses(trainer)
            print_progress(trainer)
            validated_current_state = True

            # Save
            current_validation_loss = trainer.validation_history.losses[-1, -1]
            if (
                save_if_best
                and current_validation_loss < trainer.state.best_validation_loss
            ):
                # Saving at step 0 as well to reserve the `saved_parameters_index`
                save(trainer)
                trainer.state.best_validation_loss = current_validation_loss

            # Update scheduler
            if trainer.state.scheduler is not None:
                trainer.state.scheduler.step(current_validation_loss)

        # Stop
        stop, stopping_reason = stop_training(trainer)
        if stop:
            print(f"{stopping_reason}, stopping")
            if not validated_current_state:
                get_losses(trainer)
                print_progress(trainer)
            break

        # Train
        step(trainer)
        validated_current_state = False

        # Reset
        reset, resetting_reason = reset_to_best(trainer)
        if reset:
            print(
                resetting_reason,
                f"in step {trainer.state.n_steps}, resetting the optimizer...",
            )
            load(
                trainer.config.saved_parameters_index,
                trainer,
                load_optimizer=False,
                load_scheduler=False,
            )
            trainer.state.optimizer.load_state_dict(
                copy.deepcopy(initial_optimizer_state_dict)
            )
            if trainer.state.n_loaded_current_parameters > 0:
                for param_group in trainer.state.optimizer.param_groups:
                    reduction_factor = trainer.state.n_loaded_current_parameters + 1
                    lr = param_group["lr"] / reduction_factor
                    param_group["lr"] = lr
                    print(
                        f"Learning rate reduced by a factor of {reduction_factor} to {lr} until the next reset"
                    )
