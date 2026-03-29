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


""" This file is for all miscellaneous functions needed for the training or network code.
"""

import os

# must set variable beforehand
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import subprocess

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

if torch.cuda.is_available():
    from gpustat import GPUStatCollection


def get_lr(optimizer: Optimizer) -> float:
    """get the learning rate"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return float("nan")


def get_momentum(optimizer: Optimizer) -> float:
    """get the momentum"""
    for param_group in optimizer.param_groups:
        if "momentum" in param_group:
            return param_group["momentum"]
    return float("nan")


def confusion_from_lists(
    target_stages: Tensor, predicted_stages: Tensor, weights: Tensor, stage_count: int
) -> Tensor:
    """get confusion tensor from targets and predictions"""

    # remove the epochs with weight = 0 (these were un-scored epochs)
    predicted_stages = predicted_stages[weights > 0]
    target_stages = target_stages[weights > 0]

    # remove the padded stages (target_stage == -1)
    predicted_stages = predicted_stages[target_stages != -1]
    target_stages = target_stages[target_stages != -1]

    # create confusion matrix
    confusion = torch.zeros(stage_count, stage_count, dtype=torch.int64)
    for target_i in range(stage_count):
        for predict_i in range(stage_count):
            confusion[target_i, predict_i] = len(
                predicted_stages[
                    (target_stages == target_i) & (predicted_stages == predict_i)
                ]
            )

    return confusion


def stage_kappas(confusion: Tensor, weights=None) -> Tensor:
    """get the stage-based kappas"""
    if weights is not None:
        confusion = confusion * weights

    stage_count = confusion.size(0)
    if stage_count != confusion.size(1):
        raise ValueError("confusion should be square")

    kappas = torch.zeros(stage_count)
    for stage in range(stage_count):
        stage_confusion = torch.zeros(2, 2)

        # class_correct
        stage_confusion[0, 0] = confusion[stage, stage]
        # class_error
        stage_confusion[0, 1] = confusion[stage, :].sum() - stage_confusion[0, 0]
        # other_error
        stage_confusion[1, 0] = confusion[:, stage].sum() - stage_confusion[0, 0]
        # other_correct
        stage_confusion[1, 1] = confusion.sum() - stage_confusion.sum()

        kappas[stage] = kappa(stage_confusion)

    return kappas


def kappa(confusion: Tensor, weights=None) -> Tensor:
    """calculate Cohen's kappa for a given confusion matrix"""

    # make sure confusion is float
    confusion = confusion.float()

    if weights is not None:
        confusion = confusion * weights

    if confusion.size(0) != confusion.size(1):
        raise ValueError("confusion should be square")

    # normalize
    confusion = confusion / confusion.sum()

    # calculate
    po = confusion.diag().sum()
    expected = torch.ger(confusion.sum(1), confusion.sum(0))
    pe = expected.diag().sum()

    if pe == 1.0:
        # edge case where both raters agree 100% of the time
        # but only one label is used
        kappa_value = torch.Tensor([1.0])
    else:
        kappa_value = (po - pe) / (1.0 - pe)
        kappa_value = kappa_value.unsqueeze(0)

    return kappa_value


def pad_tensor(vec: Tensor, pad: int, dim: int, value: float = 0.0) -> Tensor:
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
        value - value to use in padding (default = 0)

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    output = torch.cat([vec, value * vec.new_ones(*pad_size)], dim=dim)

    return output


def trainable_parameters(module: Module) -> int:
    """count the number of trainable parameters"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_json_contents_as_dict(json_filename: str) -> dict:
    """get a dict from a json file"""
    contents = ""
    with open(json_filename) as json_file:
        for line in json_file:
            contents += line
        contents = contents.replace("\n", " ").replace("\r", "").replace(", ", ",")
        json_dict = json.loads(contents)

    return json_dict


def return_available_gpu(max_mb_used: int = 1500) -> torch.device:
    """find an available gpu"""
    if not torch.cuda.is_available():
        print("cuda is not available, using cpu")
        return torch.device("cpu")

    gpu_index = -1
    gpu_stats = GPUStatCollection.new_query()
    for gpu_index_i in range(torch.cuda.device_count()):
        gpu_mem_used = gpu_stats.jsonify()["gpus"][gpu_index_i]["memory.used"]
        gpu_mem_total = gpu_stats.jsonify()["gpus"][gpu_index_i]["memory.total"]
        gpu_mem_free = gpu_mem_total - gpu_mem_used
        print(f"GPU{gpu_index_i}: {gpu_mem_free} MB free, {gpu_mem_used} MB used")
        if (gpu_mem_used < max_mb_used) and (gpu_mem_total > 6000):
            gpu_index = gpu_index_i
            break

    if gpu_index > -1:
        device = torch.device("cuda", gpu_index)
    else:
        print("no available gpu found")
        device = torch.device("cpu")

    return device


def shm_avail_bytes() -> int:
    """get available bytes in /dev/shm  (this is a slow call)"""
    # df defaults to kib
    if not os.path.isdir("/dev/shm/"):
        return 0

    try:
        kibytes = int(
            subprocess.check_output(["df", "/dev/shm"]).split()[10].decode("utf-8")
        )
    except:
        kibytes = 0

    return 1024 * kibytes


def check_weights(model: Module) -> None:
    """do a check on all weights and biases (parameters that require grad)"""

    count_zeros = 0
    count_ones = 0
    max_value = -float("inf")
    min_value = float("inf")
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_check = param.detach().cpu().numpy()
            new_zeros = (params_to_check == 0).sum().item()

            if new_zeros > 0:
                print(f"fixing {new_zeros} zeros")
                param.data[param.data == 0] = 0.01
                params_to_check = param.detach().cpu().numpy()
                new_zeros = (params_to_check == 0).sum().item()
                print("zeros now:", new_zeros)

            count_zeros += new_zeros
            count_ones += (params_to_check == 1).sum().item()
            if params_to_check.max().item() > max_value:
                max_value = params_to_check.max().item()
            if params_to_check.min().item() < min_value:
                min_value = params_to_check.min().item()

    print("weight checks:")
    print(f"  count of zeros: {count_zeros}  ones: {count_ones}")
    print(f"  max weight: {max_value:.4f}  min: {min_value:.4f}")

    # sanity check, don't train if any zeros
    assert count_zeros == 0


def combine_stages(stages: Tensor, stage_count: int) -> Tensor:
    """combine stages if the stage_count is 3 or 4"""
    if stage_count == 5:
        return stages

    if stage_count == 4:
        # 0=W, 1=S1+S2, 2=S3, 3=REM  (W/Light/Deep/REM)
        stages[stages == 2] = 1  # combined S2 and S1
        stages[stages == 3] = 2  # move S3 to 2
        stages[stages == 4] = 3  # move REM to 3
    elif stage_count == 3:
        # 0=W, 1=S1+S2+S3, 2=REM  (W/NREM/REM)
        stages[stages == 2] = 1  # combined S2 and S1
        stages[stages == 3] = 1  # combined S3, S2 and S1
        stages[stages == 4] = 2  # move REM to 2
    else:
        raise ValueError

    return stages
