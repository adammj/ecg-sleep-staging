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


"""This file is for the SleepDataset and SleepSubject classes."""

import os
import pickle
import time
from typing import Tuple

import h5py as h5
import numpy as np
import pandas as pd
import torch
from sleep_support import combine_stages, confusion_from_lists, kappa, shm_avail_bytes
from sleepnet import load_sample_from_file
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

FILE_EXTENSION = ".h5"

class SleepSubject(object):
    """Holds a single subject. Handles data loading and caching."""

    def __init__(
        self,
        filename: str,
        dataset_index: int,
        original_path: str,
        pickle_path,
        dataset_number: int,
        cache_during_creation: bool,
        stage_count: int,
        is_training: bool = True,
        validate_ecg: bool = True,
    ):
        super(SleepSubject, self).__init__()

        # data variables
        self.filename = filename  # w/o extension
        self.original_path = original_path  # original
        self.pickle_path = pickle_path  # pickled
        self.is_pickled = False
        self.is_training = is_training
        self.validate_ecg = validate_ecg

        # create pickle_path directory, if doesn't exist
        if pickle_path is not None and not os.path.isdir(pickle_path):
            os.mkdir(pickle_path)

        # misc data about subject
        self.dataset_number = dataset_number
        self.dataset_index = dataset_index
        self.stage_count = stage_count
        self.epoch_count: int = 0
        self.weight: float = 1.0
        self.kappa: float = 0.0
        self.confusion = Tensor([])
        self.target_stages = Tensor([])
        self.predicted_stages = Tensor([])

        # try the pickle file first
        if (pickle_path is not None) and self.try_to_load_file(True, True):
            self.is_pickled = True
        else:
            self.try_to_load_file(False)

        if self.is_valid and cache_during_creation and not self.is_pickled:
            self.cache_data()

    @property
    def is_valid(self):
        """check if subject was validly created"""
        return self.epoch_count > 0

    def full_file_name(self, pickle_version: bool):
        """get the file_name based on the type"""
        if pickle_version and (not self.pickle_path is None):
            file_name = self.pickle_path + self.filename + ".pkl"
        else:
            file_name = self.original_path + self.filename + FILE_EXTENSION
            file_name = file_name.replace(FILE_EXTENSION + FILE_EXTENSION, FILE_EXTENSION)
        return file_name

    def check_ecg_variable(self, ecgs: Tensor | np.ndarray):
        """check the ecg variable"""

        if not (torch.is_tensor(ecgs) or isinstance(ecgs, np.ndarray)):
            raise TypeError("ecgs is not a Tensor or Numpy array")

        if ecgs.ndim != 2:
            raise ValueError("ecgs should be a 2D array")

        if ecgs.shape[0] == 0:
            raise ValueError("ecgs should have at least 1 epoch of data")

        value = ecgs.shape[1]
        if value != 7680:
            raise ValueError(
                "ecgs second dimension should be exactly 7680 (30sec * 256Hz), but" +
                f"is {value}"
            )

        if torch.is_tensor(ecgs):
            value = torch.min(ecgs)
            if value < -1:
                raise ValueError(f"no value in ecgs should be below -1 (found: {value})")

            value = torch.max(ecgs)
            if value > 1:
                raise ValueError(f"no value in ecgs should be above 1 (found: {value})")

            value = torch.abs(torch.median(ecgs))
            if value > 0.001:
                raise ValueError(f"median of ecgs should be ~0 (found: {value})")

        else:
            value = np.min(ecgs)
            if value < -1:
                raise ValueError(f"no value in ecgs should be below -1 (found: {value})")

            value = np.max(ecgs)
            if value > 1:
                raise ValueError(f"no value in ecgs should be above 1 (found: {value})")

            value = np.abs(np.median(ecgs))
            if value > 0.001:
                raise ValueError(f"median of ecgs should be ~0 (found: {value})")

    def try_to_load_file(self, use_pickled_file: bool, silent_missing: bool = False):
        """check if file exists and load some data from the file"""
        file_name = self.full_file_name(use_pickled_file)

        if not os.path.isfile(file_name):
            if not silent_missing:
                print(file_name + " is missing")
            return False
        else:
            if not use_pickled_file:
                try:
                    file_h = h5.File(file_name, "r")
                except Exception:
                    print(file_name + " is broken")
                    return False

                if self.validate_ecg:
                    self.check_ecg_variable(file_h["ecgs"][()])  # type: ignore

                if self.is_training:
                    # if training, load the actual stages variable
                    stages_tensor = torch.LongTensor(file_h["stages"][()])  # type: ignore
                else:
                    # otherwise, create a dummy stages tensor
                    stages_tensor = torch.zeros(
                        (file_h["ecgs"].shape[0], 1),  # type: ignore
                        dtype=torch.long,
                    )

                self.epoch_count = stages_tensor.shape[0]
                self.target_stages = stages_tensor.squeeze()
                self.target_stages = combine_stages(
                    self.target_stages, self.stage_count
                )

                # close before returning
                file_h.close()

                return True
            else:
                try:
                    file_h = open(file_name, "rb")
                except Exception:
                    print(file_name + " is broken")
                    return False
                sample = pickle.load(file_h)
                if self.validate_ecg:
                    self.check_ecg_variable(sample["ecgs"])

                self.epoch_count = sample["epoch_count"]
                self.target_stages = sample["stages"].squeeze()
                self.target_stages = combine_stages(
                    self.target_stages, self.stage_count
                )

                # close before returning
                file_h.close()

                return True

    def get_data(self):
        # cannot update any variables within here, as they are across threads
        # and aren't saved
        """get the subject data"""
        assert self.is_valid

        if self.is_pickled:
            file_name = self.full_file_name(True)
            with open(file_name, "rb") as file_h:
                sample = pickle.load(file_h)

        else:
            file_name = self.full_file_name(False)
            with h5.File(file_name, "r") as file_h:
                sample = load_sample_from_file(file_h, self.is_training)
                # make sure sample can be located
                sample.update({"subject_filename": self.filename})
        
        sample.update({"stages": self.target_stages.clone()})

        # add keys not in original dataset files
        sample.update({"dataset_number": self.dataset_number})

        return sample

    def cache_data(self):
        """cache the subject data"""
        assert self.is_valid

        sample = self.get_data()
        file_name = self.full_file_name(True)
        with open(file_name, "wb") as file_h:
            pickle.dump(sample, file_h)
        self.is_pickled = True

    def update_prediction(self, predicted_stages: Tensor, weights: Tensor):
        """get the confusion and kappa for the subject"""
        assert self.is_valid

        # predicted_stages and weights will likely be padded, cut down
        self.predicted_stages = predicted_stages[: self.epoch_count].cpu()
        weights = weights[: self.epoch_count].cpu()

        self.confusion = confusion_from_lists(
            self.target_stages, self.predicted_stages, weights, self.stage_count
        )
        self.kappa = float(kappa(self.confusion).item())


class SleepDataset(Dataset):
    """Holds the (train/validation/test) data set, that can then be loaded from one at a time"""

    def __init__(
        self,
        train_params: dict,
        set_type: int,
        max_samples: int,
        cache_data: bool,
        stage_count: int,
    ):
        super(SleepDataset, self).__init__()

        # extract the common keys
        all_datasets_folder = train_params["all_datasets_folder"]
        self.dataset_number = train_params["dataset_number"]
        sets_file = train_params["sets_file"]
        self.train_invert_ecg = train_params["train_invert_ecg"]
        self.train_noise_per = train_params["train_noise_per"]
        self.weight_subjects = train_params["weight_subjects"]
        self.validate_ecg = train_params["validate_ecg"]

        # 1=train, 2=validation, 3=test, 4=train+validation
        if set_type not in [1, 2, 3, 4]:
            raise ValueError("set_type should in [1, 2, 3, 4]")

        # shm/pickle folder
        self.shm_folder = "/dev/shm/dataset_" + str(self.dataset_number) + "_pickles/"
        # check for shm existence, otherwise remove
        if not os.path.isdir("/dev/shm/"):
            self.shm_folder = None

        # default value
        self.stage_count = stage_count
        self.valid_setup = True
        self.set_type = set_type
        self.is_training = (
            set_type == 1 or set_type == 4
        )  # 1 and 4 are considered training sets
        self.folder = (
            all_datasets_folder + "dataset_" + str(self.dataset_number) + "_files/"
        )

        # check if a single file is supposed to be used
        if "single_datafile" in train_params:
            datafile_string = train_params["single_datafile"]

            # check if this provides a full path or is a file in the CWD
            if len(datafile_string.split("/")) == 1:
                datafile_string = os.getcwd() + "/" + datafile_string
                datafile_string = datafile_string.replace("//", "/")

            train_params["single_datafile"] = datafile_string

            self.folder = "/".join(datafile_string.split("/")[0:-1]) + "/"
            self.filenames = [datafile_string.split("/")[-1]]

        else:
            # extract the sets file
            sets_df = pd.read_excel(sets_file)

            self.filenames: list[str]
            # adjust the set_type to a tuple, if necessary
            if set_type == 4:
                self.filenames = list(
                    sets_df[(sets_df.set == 1) | (sets_df.set == 2)].file
                )
            else:
                self.filenames = list(sets_df[sets_df.set == set_type].file)

        # extract the relevant data
        self.count = len(self.filenames)  # count returned from the matrix

        if (max_samples > 0) and (max_samples < self.count):
            print("only using first " + str(max_samples) + " samples")
            self.count = max_samples

        # prepare all subject data
        self.subjects: list[SleepSubject] = []
        error_preparing_subjects = False
        byte_padding = 5 * (1000**3)  # keep 5GB for misc stuff
        sufficient_cache_space = shm_avail_bytes() > byte_padding

        print(f"there are {self.count} files:")
        with tqdm(total=self.count, bar_format="{l_bar}{bar:20}{r_bar}") as progress:
            progress.set_description("checking")
            for i in range(self.count):
                # check available bytes, and only cache if enough padding is left
                cache_during_creation = cache_data and sufficient_cache_space

                self.subjects += [
                    SleepSubject(
                        self.filenames[i],
                        i,
                        self.folder,
                        self.shm_folder,
                        self.dataset_number,
                        cache_during_creation,
                        self.stage_count,
                        self.is_training,
                        self.validate_ecg,
                    )
                ]

                # if caching, check the space again
                # this minimizes calls to a slow function
                if cache_during_creation:
                    sufficient_cache_space = shm_avail_bytes() > byte_padding

                if not self.subjects[i].is_valid:
                    error_preparing_subjects = True

                progress.update(1)

        if error_preparing_subjects:
            self.valid_setup = False
            print("error during checking")
            quit()

        # stuff for per-subject weighting
        self.weight_limit = train_params["weight_subjects_limit"]
        self.weight_slope = train_params["weight_subjects_slope"]
        self.min_weight = train_params["weight_subjects_min"]

    def is_ready(self):
        """was the dataset initialized completely"""
        return self.valid_setup

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        if index < 0 or index >= self.count:
            raise ValueError("index must be in [0, self.count)")

        t_start = time.perf_counter()

        # load data
        sample = self.subjects[index].get_data()

        # during training, some data is modified
        if self.is_training:
            # 50% chance of inverting the ecg
            if self.train_invert_ecg and (np.random.randint(0, 2) > 0):
                sample["ecgs"] = -sample["ecgs"]

            # if subject weight < 1, then multiply epoch weights by subject weight
            if self.subjects[index].weight < 1:
                sample["weights"] = sample["weights"] * self.subjects[index].weight

            # add gaussian noise to input, up to some % of the stdev
            if self.train_noise_per > 0:
                ecg_std = torch.std(sample["ecgs"])
                noise_level = (
                    self.train_noise_per * torch.rand(1) * ecg_std
                )  # randomly chosen
                sample["ecgs"] += noise_level * torch.randn_like(sample["ecgs"])

        # if stage_count is not 5, then adjust appropriately
        if self.stage_count < 5:
            sample["stages"] = combine_stages(sample["stages"], self.stage_count)

        t_end = time.perf_counter()
        sample.update({"load_time": (t_end - t_start)})
        return sample

    def update_subject_weights(self) -> float:
        """find new subject weights"""
        if not self.is_training or not self.weight_subjects:
            # no weighting allowed for evaluation set
            subject_weights = torch.ones(self.count)
            self.set_all_subject_weights(subject_weights)
            return 1.0

        # aggregate values
        subject_kappas = self.get_all_subject_kappas()
        subject_weights = self.get_all_subject_weights()
        median = subject_kappas.median()

        # if the median is too small, don't update weights
        if median < 0.3:
            print(f"median kappa too low to update weights: {median.item():.3f}")
        else:
            mad = 1.4826 * (subject_kappas - median).abs().median()
            if mad <= 0:
                mad = 1.0  # prevent div by zero
            z_score_kappas = (subject_kappas - median) / mad

            # update weights above limit to 1
            subject_weights[z_score_kappas >= self.weight_limit] = 1
            # update weights below limit to <1
            other_weights = (
                1
                + (
                    z_score_kappas[z_score_kappas < self.weight_limit]
                    - self.weight_limit
                )
                / self.weight_slope
            )
            subject_weights[z_score_kappas < self.weight_limit] = other_weights
            # set a minimum weight
            subject_weights[subject_weights < self.min_weight] = self.min_weight

            # store the new weights
            self.set_all_subject_weights(subject_weights)

        return round(subject_weights.mean().item(), 3)

    def update_subject_kappas(
        self, filenames: list[str], predicted_stages: Tensor, weights: Tensor
    ):
        """call each subject to update their kappas"""
        # detach, reshape, and move to cpu
        predicted_stages = predicted_stages.view(len(filenames), -1).cpu()
        weights = weights.view(len(filenames), -1).cpu()

        # update kappas for each subject in a batch
        for i, subject_filename in enumerate(filenames):
            # get the subject index for the given filename
            subject_index = self.filenames.index(subject_filename)

            self.subjects[subject_index].update_prediction(
                predicted_stages[i, :], weights[i, :]
            )

    def get_all_subject_kappas(self) -> Tensor:
        """get kappas for all subjects"""
        subject_kappas = torch.zeros(self.count)
        for i in range(self.count):
            subject_kappas[i] = self.subjects[i].kappa
        return subject_kappas

    def get_mean_of_kappas(self) -> Tuple[float, float]:
        """get the mean and median of all subject kappas"""

        subject_kappas = self.get_all_subject_kappas()
        mean_kappa = round(subject_kappas.mean().item(), 3)
        median_kappa = round(subject_kappas.median().item(), 3)

        return mean_kappa, median_kappa

    def get_all_subject_confusions(self) -> Tensor:
        """get all subject confusions"""
        confusions = torch.Tensor(self.count, self.stage_count, self.stage_count)
        for i in range(self.count):
            confusions[i, :, :] = self.subjects[i].confusion

        return confusions

    def set_all_subject_weights(self, weights: Tensor) -> None:
        """set the weights for all subjects"""
        for i in range(self.count):
            self.subjects[i].weight = weights[i].item()

    def get_all_subject_weights(self) -> Tensor:
        """get weights for all subjects"""
        subject_weights = torch.zeros([self.count])
        for i in range(self.count):
            subject_weights[i] = self.subjects[i].weight

        return subject_weights

    def get_all_subject_predictions(self) -> Tensor:
        """get all subject predictions"""

        # first get the maximum size
        max_epoch_count = 0
        for i in range(self.count):
            epoch_count = self.subjects[i].epoch_count
            if epoch_count > max_epoch_count:
                max_epoch_count = epoch_count

        # create matrix with nans, to make it obvious where counts end
        predictions = torch.ones(self.count, max_epoch_count) * torch.nan
        for i in range(self.count):
            epoch_count = self.subjects[i].epoch_count
            predictions[i, 0:epoch_count] = self.subjects[i].predicted_stages

        return predictions
