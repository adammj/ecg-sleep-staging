# Copyright (C) 2024  Adam Jones  All Rights Reserved
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


""" This file is to keep all of the training-specific functionality.
"""

import os

# must set variable beforehand
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta

import h5py as h5
import numpy as np
import torch
from adams import AdamS
from dataclasses_json import dataclass_json
from sleep_support import (
    confusion_from_lists,
    get_json_contents_as_dict,
    get_lr,
    get_momentum,
    kappa,
    return_available_gpu,
    stage_kappas,
    trainable_parameters,
)
from sleepdataset import SleepDataset
from sleeploss import SleepLoss
from sleepnet import SleepNet, move_tensors_to_device, sleepnet_collate
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

#  precision and printing constants
stats_precision = 3
np.set_printoptions(precision=stats_precision, suppress=True, floatmode="fixed")

# set everything to be deterministic
torch.use_deterministic_algorithms(True)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.benchmark = False  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
random_seed = 0  # updated and used as a global


def seed_init_fn(worker_id: int = 0):
    """prepare for deterministic execution by setting seeds"""
    # this is called many times (over-precaution)
    # added +worker_id so that each worker gets a slightly different seed

    global random_seed
    if random_seed == 0:
        raise ValueError("random_seed should not be 0")

    torch.manual_seed(random_seed + worker_id)  # seeds both cpu and cuda
    np.random.seed(random_seed + worker_id)
    random.seed(random_seed + worker_id)


@dataclass_json
@dataclass
class SetStats:
    """Statistics for a training/validation/testing set"""

    stage_kappas: list = field(default_factory=list)
    confusion: list = field(default_factory=list)
    loss_confusion: list = field(default_factory=list)
    acc: float = 0
    overall_kappa: float = 0
    avg_batch_time: float = 0
    loss: float = 0
    mean_kappa: float = 0
    median_kappa: float = 0
    avg_subject_weight: float = 0
    epoch_time: float = 0

    def set_confusion(self, confusion: Tensor, precision: int = 3) -> None:
        """Use a confusion matrix to update some variables."""
        stage_kappas_matrix = stage_kappas(confusion).cpu().numpy()
        stage_kappas_matrix = (
            stage_kappas_matrix.astype(float).round(precision).tolist()
        )  # fixed to keep precision
        self.stage_kappas = stage_kappas_matrix
        self.confusion = confusion.tolist()
        self.acc = round(
            (confusion.diag().sum().float() / confusion.sum().float()).item(), precision
        )
        self.overall_kappa = round(kappa(confusion).item(), precision)

    def set_loss_confusion(self, confusion: Tensor, precision: int = 3) -> None:
        self.loss_confusion = (
            confusion.cpu().numpy().astype(float).round(precision).tolist()
        )


@dataclass_json
@dataclass
class EpochStats:
    """Statistics for an epoch."""

    epoch: int = 0
    elapsed: float = 0
    learning_rate: float = 0
    momentum: float = 0
    train_stats: SetStats = field(default_factory=SetStats)
    val_stats: SetStats = field(default_factory=SetStats)
    checkpoint: bool = False
    epoch_time: float = 0


class SleepNetExperiment:
    """Class to train/evaluate the network."""

    def __init__(self, train_params: dict, net_params: dict) -> None:
        super(SleepNetExperiment, self).__init__()

        # prepare everything
        print("preparing...")
        self.time_training_started = time.perf_counter()

        # extract some often used parameters
        self.save_results_and_exit = train_params["save_results_and_exit"]
        self.stage_count = net_params["stage_count"]
        self.epochs_to_train = train_params["epochs_to_train"]

        self.train_params: dict = train_params
        self.net_params: dict = net_params

        # set the random seed
        global random_seed
        random_seed = train_params["rand_seed"]

        # set the other instance variables
        self.epoch_checkpoint_offset: int = 0
        self.highest_eval_kappa: float = 0.0
        self.loaded_checkpoint: bool = False
        self.t_start: float
        self.epoch_num: int = 0

        # placeholders for various objects
        self.device: torch.device
        self.model: SleepNet
        self.optimizer: Optimizer
        self.loss_func: SleepLoss
        self.scheduler: object

        # train and eval objects
        self.train_set: SleepDataset
        self.train_loader: DataLoader
        self.train_stats: SetStats = SetStats(
            stage_kappas=[], confusion=[[]], loss_confusion=[[]]
        )
        self.eval_set: SleepDataset
        self.eval_loader: DataLoader
        self.eval_stats: SetStats = SetStats(
            stage_kappas=[], confusion=[[]], loss_confusion=[[]]
        )

        # pick the device
        self.update_device()

        # so that next job doesn't choose the same gpu
        print("create large temp variable on device")
        temp_var = torch.zeros(10000, 10000, device=self.device)
        temp_var[1, 1] = 1  # set a value so that variable is used

        # these take some time, so block other instances temporarily
        self.create_datasets()
        self.create_dataloaders()

        print("delete temp variable")
        del temp_var

        self.create_model()
        self.create_training_objects()

        # finally, load any checkpoint requested
        self.load_checkpoint()

    def update_device(self) -> None:
        """Update the device used for pytorch."""

        # set default device to cpu
        device = torch.device("cpu")

        if self.train_params["device"] == "cuda":
            # assumes using gpu (find the first with at least enough free memory)
            if torch.cuda.is_available():
                device = return_available_gpu()

                if device.type == "cuda":
                    torch.cuda.set_device(device)
                else:
                    print("no available gpu found, using cpu")

            else:
                print("cuda is not available, using cpu")
        elif self.train_params["device"] == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
            else:
                print("mps not available, using cpu")

        print("using device: ", device)
        self.device = device

    def create_datasets(self) -> None:
        """Create the train/evaluate datasets"""
        print("create datasets...")
        print("evaluation (validation or testing) set:")
        # loaded first so that it can entirely be fit into memory
        self.eval_set = SleepDataset(
            self.train_params,
            self.train_params["eval_set_type"],
            self.train_params["eval_max_samples"],
            self.train_params["cache_datasets"],
            stage_count=self.stage_count,
        )
        if not self.eval_set.is_ready():
            quit()

        if not self.save_results_and_exit:
            print("training set:")
            self.train_set = SleepDataset(
                self.train_params,
                self.train_params["train_set_type"],
                self.train_params["train_max_samples"],
                self.train_params["cache_datasets"],
                stage_count=self.stage_count,
            )
            if not self.train_set.is_ready():
                quit()

    def create_dataloaders(self) -> None:
        """Create the loaders for the datasets."""
        print("create data loaders...")

        # if only saving results, then no workers are necessary
        if self.train_params["save_results_and_exit"] is True:
            self.train_params["num_workers"] = 0

        if not self.save_results_and_exit:
            # shuffle training loader
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=self.train_params["batch_size"],
                shuffle=True,
                num_workers=self.train_params["num_workers"],
                collate_fn=sleepnet_collate,
                pin_memory=(self.device.type == "cuda"),
                worker_init_fn=seed_init_fn,
                persistent_workers=(self.train_params["num_workers"] > 0),
            )

        # evaluation loader is given batch_size = 1 to make sure that results are always reproducible
        self.eval_loader = DataLoader(
            self.eval_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.train_params["num_workers"],
            collate_fn=sleepnet_collate,
            pin_memory=(self.device.type == "cuda"),
            worker_init_fn=seed_init_fn,
            persistent_workers=(self.train_params["num_workers"] > 0),
        )

    def create_model(self) -> None:
        """Create the network."""
        print("create model...")

        # prepare a sample
        sample = self.eval_set.__getitem__(0)
        sample_col = sleepnet_collate([sample])

        # set seed before creating model (random used for parameters)
        seed_init_fn()

        # create the model
        model = SleepNet(self.net_params, sample_col)
        print("model parameters: " + str(trainable_parameters(model)))

        # move model
        self.model = model.to(self.device)

    def create_training_objects(self) -> None:
        """Create the remaining training objects."""
        print("create criterion, optimizer, and scheduler...")

        # criterion (loss function)
        self.loss_func = SleepLoss(stage_count=self.stage_count)

        # optimizer
        print("learning rate: " + str(self.train_params["learning_rate"]))
        self.optimizer = AdamS(
            self.model.parameters(),
            lr=self.train_params["learning_rate"],
            eps=self.train_params["eps"],
            amsgrad=self.train_params["amsgrad"],
            weight_decay=self.train_params["weight_decay"],
        )

        # scheduler
        if self.train_params["lr_reduce_patience"] == -1:
            print("disabling scheduler")
            self.train_params["lr_reduce_patience"] = self.epochs_to_train

        # scheduler is stepped on kappa
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "max",
            factor=self.train_params["lr_reduce_factor"],
            patience=self.train_params["lr_reduce_patience"],
            min_lr=self.train_params["min_learning_rate"],
        )

    def load_checkpoint(self) -> None:
        """Load a checkpoint."""
        if os.path.isfile(self.train_params["resume_checkpoint"]):
            print("load checkpoint file: ", self.train_params["resume_checkpoint"])
            checkpoint = torch.load(
                self.train_params["resume_checkpoint"],
                map_location=self.device,
            )
            print("keys:", checkpoint.keys())

            print("set basics...")
            self.epoch_checkpoint_offset = checkpoint["epoch"]
            self.highest_eval_kappa = checkpoint["val_kappa"]

            print("set model state dict...")
            self.model.load_state_dict(checkpoint["model_state_dict"])

            print("set optimizer state dict...")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("set scheduler state dict...")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore

            if not self.save_results_and_exit:
                print("load subject weights")
                self.train_set.set_all_subject_weights(checkpoint["subject_weights"])

            print("resuming from checkpoint file...")
            self.loaded_checkpoint = True

    def save_checkpoint(self) -> bool:
        """Save a checkpoint."""
        if not self.train_params["save_checkpoints"]:
            return False

        # check if new highest val kappa (with a tiny offset)
        saved_new_checkpoint = False
        if self.eval_stats.overall_kappa >= (self.highest_eval_kappa + 0.001):
            print("saving checkpoint...")

            # create state
            checkpoint_state = {
                "epoch": self.epoch_num,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore
                "scheduler_state_dict": self.scheduler.state_dict(),  # type: ignore
                "val_kappa": self.eval_stats.overall_kappa,
                "val_loss": self.eval_stats.loss,
                "learning_rate": get_lr(self.optimizer),
                "subject_weights": self.train_set.get_all_subject_weights(),
            }

            self.highest_eval_kappa = self.eval_stats.overall_kappa

            file_name = (
                f"epoch_{self.epoch_num:04d}_kappa_{self.highest_eval_kappa:.3f}.pt"
            )
            torch.save(checkpoint_state, file_name)

            saved_new_checkpoint = True

        return saved_new_checkpoint

    def save_useful_results(self) -> None:
        """Save some results to a file"""
        print(
            f"save results and exit. elapsed: {(time.perf_counter() - self.t_start):.1f} sec"
        )
        confusions = self.eval_set.get_all_subject_confusions().numpy()
        predictions = self.eval_set.get_all_subject_predictions().numpy()
        
        with h5.File("results.h5", "w") as file_h:
            file_h.create_dataset("confusions", data=confusions)
            file_h.create_dataset("predictions", data=predictions)
            if not self.save_results_and_exit:
                train_subject_weights = self.train_set.get_all_subject_weights().numpy()
                file_h.create_dataset("train_subject_weights", data=train_subject_weights)

    def run_main_loop(self) -> None:
        """Run the main train/evaluate loop."""
        self.t_start = time.perf_counter()
        tend_last = self.t_start

        # set seed one last time before main loop
        seed_init_fn()
        print("begin training and evaluation...")

        # begin training/evaluation loop
        for epoch_i in range(self.epochs_to_train):
            time_elapsed = time.perf_counter() - self.time_training_started
            self.epoch_num = epoch_i + 1 + self.epoch_checkpoint_offset
            epoch_stats = EpochStats(
                epoch=self.epoch_num, elapsed=round(time_elapsed, 0)
            )

            print("-" * 70)  # pretty epoch separator
            elapsed_pretty = str(timedelta(seconds=round(time_elapsed, 0)))
            print(
                f"epoch: {self.epoch_num: >4}/{self.epochs_to_train}   elapsed: {elapsed_pretty}"
            )

            # get and store the current learning rate and momentum
            optimizer_lr = get_lr(self.optimizer)
            optimizer_momentum = get_momentum(self.optimizer)

            # if saving results, the skip training
            if not self.save_results_and_exit:  # or self.loaded_checkpoint:

                # start training
                print(
                    f"training...  (lr: {optimizer_lr:.6f}, momentum: {optimizer_momentum:.6f})"
                )
                epoch_stats.learning_rate = round(optimizer_lr, 6)
                epoch_stats.momentum = round(optimizer_momentum, 6)

                if epoch_i == 0:
                    print("first epoch, just get stats")
                    with torch.inference_mode():
                        self.train_stats = self.train_or_eval(
                            False,
                            self.train_loader,
                            self.train_set,
                        )
                else:
                    self.train_stats = self.train_or_eval(
                        True,
                        self.train_loader,
                        self.train_set,
                    )
                epoch_stats.train_stats = self.train_stats

            # start evaluating
            print("evaluating... ")

            with torch.inference_mode():
                self.eval_stats = self.train_or_eval(
                    False,
                    self.eval_loader,
                    self.eval_set,
                )
            epoch_stats.val_stats = self.eval_stats

            # save and exit
            if self.save_results_and_exit:
                self.save_useful_results()
                return

            # step the scheduler
            self.scheduler.step(self.eval_stats.overall_kappa)  # type: ignore

            print("epoch stats:")
            print(
                "  train: kappa: overall:",
                np.round(np.array(self.train_stats.overall_kappa), 3),
                " stage: ",
                np.array(self.train_stats.stage_kappas),
            )
            print(
                "  eval : kappa: overall:",
                np.round(np.array(self.eval_stats.overall_kappa), 3),
                " stage: ",
                np.array(self.eval_stats.stage_kappas),
            )

            sch_state_dict = self.scheduler.state_dict()  # type: ignore
            if "num_bad_epochs" in sch_state_dict:
                print("  num_bad_epochs:", sch_state_dict["num_bad_epochs"])
            else:
                print(" ")

            epoch_stats.checkpoint = self.save_checkpoint()

            t_end = time.perf_counter()
            epoch_elapsed = t_end - tend_last
            tend_last = t_end
            epoch_stats.epoch_time = round(epoch_elapsed, 1)
            elapsed_pretty = str(timedelta(seconds=round(t_end - self.t_start, 0)))
            print(
                f"epoch took: {epoch_elapsed:.0f} sec,  total elapsed: {elapsed_pretty}"
            )

            # save to log file
            with open(self.train_params["log_file"], mode="a") as json_file:
                json_file.write(epoch_stats.to_json() + "\n")  # type: ignore

    def train_or_eval(
        self,
        is_training: bool,
        loader: DataLoader,
        dataset: SleepDataset,
    ) -> SetStats:
        """Train or evaluate for a given epoch."""
        t_epoch_start = time.perf_counter()
        batch_count = len(loader)

        # train or evaluate model
        if is_training:
            self.model.train()
            progress_str = "training  "
        else:
            self.model.eval()
            progress_str = "evaluating"

        # create epoch confusion tensors
        epoch_confusion = torch.zeros(
            self.stage_count, self.stage_count, dtype=torch.int64
        )
        epoch_loss_confusion = torch.zeros(self.stage_count, self.stage_count)

        with tqdm(total=batch_count, bar_format="{l_bar}{bar:20}{r_bar}") as progress:
            progress.set_description(progress_str)
            progress.set_postfix(k=-0.000)

            # iterate through data_loader
            for _, batch_data in enumerate(loader):

                # move tensors to device
                batch_data = move_tensors_to_device(batch_data, self.device)

                # run forward pass
                model_output, _ = self.model.forward_with_dict(batch_data)

                # reshape model_output and stages/weights to get back to (subjects*epochs, labels)
                model_output = model_output.view(-1, model_output.size(-1))
                batch_target_stages = batch_data["stages"].view(-1)
                batch_weights = batch_data["weights"].view(-1)

                # get the predicted stage
                _, batch_predicted_stages = torch.max(model_output.detach(), 1)

                # zero gradients before calculating loss
                if is_training:
                    self.optimizer.zero_grad()

                # calculate loss
                batch_loss = self.loss_func(
                    model_output, batch_target_stages, batch_weights
                )

                # back propagation and step for optimizer
                if is_training:
                    batch_loss.backward()
                    self.optimizer.step()

                # build up confusion matrix and get stats efficiently
                batch_confusion = confusion_from_lists(
                    batch_target_stages,
                    batch_predicted_stages,
                    batch_weights,
                    self.stage_count,
                )
                epoch_confusion += batch_confusion
                epoch_loss_confusion += self.loss_func.loss_confusion.cpu()

                # calculate and store the kappa for each subject
                dataset.update_subject_kappas(
                    batch_data["subject_filenames"],
                    batch_predicted_stages,
                    batch_weights,
                )

                # clean up memory
                del (
                    model_output,
                    batch_target_stages,
                    batch_weights,
                    batch_loss,
                    batch_predicted_stages,
                )

                # update progress
                epoch_kappa = round(kappa(epoch_confusion).item(), 3)

                progress.set_postfix(k=epoch_kappa)
                progress.update(1)

        elapsed_time = time.perf_counter() - t_epoch_start
        avg_batch_time = elapsed_time / batch_count

        # store epoch stats
        set_stats = SetStats(stage_kappas=[], confusion=[[]], loss_confusion=[[]])
        set_stats.set_confusion(epoch_confusion, stats_precision)
        set_stats.avg_batch_time = round(avg_batch_time, stats_precision)
        set_stats.loss = round(
            self.loss_func.calculate_loss(
                epoch_loss_confusion, self.stage_count
            ).item(),
            stats_precision,
        )

        # store loss_confusion
        set_stats.set_loss_confusion(epoch_loss_confusion, 1)

        # get subject kappa stats
        mean_kappa, median_kappa = dataset.get_mean_of_kappas()
        set_stats.mean_kappa = mean_kappa
        set_stats.median_kappa = median_kappa

        # update weights
        set_stats.avg_subject_weight = dataset.update_subject_weights()

        t_epoch_end = time.perf_counter()
        set_stats.epoch_time = round(t_epoch_end - t_epoch_start, 1)

        return set_stats


def main():
    """get parameters and call main function"""
    # get train and net params
    train_params = get_json_contents_as_dict("./train_params.json")
    net_params = get_json_contents_as_dict("./net_params.json")

    # arguments: all_datasets_folder
    if len(sys.argv) > 1:
        print("has argument")
        argument_string = str(sys.argv[1])
        if os.path.isdir(argument_string):
            train_params["all_datasets_folder"] = argument_string
        else:
            if os.path.isfile(argument_string):
                print(argument_string)
                train_params["single_datafile"] = argument_string
            else:
                print("argument provided is neither a folder nor a file")
                sys.exit()

    # prepare_and_run(train_params, net_params)
    experiment = SleepNetExperiment(train_params, net_params)
    experiment.run_main_loop()


# this goes at the end of the file
if __name__ == "__main__":
    main()
