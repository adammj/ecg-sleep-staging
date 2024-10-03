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


"""This file contains everything directly related to the network, its specific data,
and any other support functions that are directly tied to what the network needs as an input.
"""

import time
from typing import Optional, Tuple

import h5py as h5
import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool1d,
    AdaptiveMaxPool1d,
    AvgPool1d,
    BatchNorm1d,
    Conv1d,
    Dropout,
    LeakyReLU,
    Linear,
    MaxPool1d,
    Module,
    Sequential,
    Softmax,
)
from torch.nn.init import normal_, xavier_normal_

from sleep_support import check_weights, pad_tensor


def load_sample_from_file(file_h: h5.File, is_training: bool = True) -> dict:
    """Load sample from a file. Kept with sleep net, in case there are future changes"""

    # extract all data. from_numpy creates the necessary copies
    epoch_count = torch.LongTensor([file_h["ecgs"].shape[0]])  # type: ignore
    ecgs = torch.Tensor(file_h["ecgs"][()])  # type: ignore

    # squeeze demographics, to make cache smaller
    demographics = torch.Tensor(file_h["demographics"][()]).squeeze()  # type: ignore
    midnight_offset = torch.Tensor(file_h["midnight_offset"][()]).squeeze()  # type: ignore

    # stages was already loaded (or faked) when file was validated

    if is_training:
        # loss requires long for stages
        weights = torch.Tensor(file_h["weights"][()])  # type: ignore
    else:
        # if not training, the variables may not exist in the file
        # and should not be used anyways
        weights = torch.ones((ecgs.shape[0]), dtype=ecgs.dtype)

    # create the sample after closing the file
    sample = {
        "epoch_count": epoch_count,
        "ecgs": ecgs,
        "demographics": demographics,
        "weights": weights,
        "midnight_offset": midnight_offset,
    }

    return sample


def init_weights(model):
    """initialize all weights and biases"""

    if hasattr(model, "weight"):
        if model.weight is not None:
            if model.weight.dim() > 1:
                xavier_normal_(model.weight)
            else:
                # with fewer than 2 dim, cannot use xavier or kaiming
                normal_(model.weight)

    if hasattr(model, "bias"):
        if model.bias is not None:
            normal_(model.bias, std=0.01)


def sleepnet_collate(batch: list) -> dict:
    """Necessary for DataLoader to handle different epoch lengths.
    This function takes the batch and pads each sample to the max epochs."""

    t_start = time.perf_counter()

    # get the maximum number of epochs
    max_epochs = max(map(lambda x: x["epoch_count"].item(), batch))

    # create empty lists to use
    subject_count = len(batch)
    dataset_number: int = 0
    subject_filenames: list[str] = []
    epoch_counts: list[Tensor] = []
    ecgs: list[Tensor] = []
    additional: list[Tensor] = []
    stages: list[Tensor] = []
    weights: list[Tensor] = []
    padding_eliminator: list[Tensor] = []
    load_times: list[float] = []

    # pad everything to the max epochs, and store pointer in list
    for i, sample in enumerate(batch):
        if i == 0:
            dataset_number = sample["dataset_number"]

        # convert to tensor so it can be concatenated
        subject_filenames += [sample["subject_filename"]]
        epoch_counts += [sample["epoch_count"]]
        ecgs += [pad_tensor(sample["ecgs"], max_epochs, 0)]

        # for "better" night/midnight vectors
        # create additional, first using demographics
        sample_additional = sample["demographics"]
        # repeat the demographics data for every epoch
        sample_additional = sample_additional.repeat(sample["epoch_count"], 1)

        # create night (-1=start, 1=end) vector
        night = torch.tensor(range(0, sample["epoch_count"]))
        night = (night / ((sample["epoch_count"] - 1.0) / 2.0)) - 1.0
        night = night.unsqueeze(1)

        # create midnight (0=midnight, +/-1=+/-24hrs) vectors
        midnight_vec = torch.tensor(range(0, sample["epoch_count"]))
        midnight_vec = midnight_vec / ((3600 / 30) * 24) + sample["midnight_offset"]
        midnight_vec = midnight_vec.unsqueeze(1)

        # concat the the demographics, with night and midnight
        sample_additional = torch.cat((sample_additional, night, midnight_vec), dim=1)

        additional += [pad_tensor(sample_additional, max_epochs, 0)]

        # stages are padded with -1, so that they can't affect the loss
        stages += [pad_tensor(sample["stages"], max_epochs, 0, -1)]
        # weights are padded with 0, so that they can't affect the loss
        weights += [pad_tensor(sample["weights"], max_epochs, 0)]

        load_times += [sample["load_time"]]

        # create the padding_eliminator here (with the feature dimension being size=1)
        # this will be expanded inside the network
        padding_temp = torch.zeros((max_epochs, 1))
        padding_temp[: sample["epoch_count"], :] = 1
        padding_eliminator += [padding_temp]

    # stack the lists
    epoch_counts_tensor = torch.stack(epoch_counts).contiguous()
    ecgs_tensor = torch.stack(ecgs).contiguous()
    additional_tensor = torch.stack(additional).contiguous()
    stages_tensor = torch.stack(stages).contiguous()
    weights_tensor = torch.stack(weights).contiguous()
    padding_eliminator_tensor = torch.stack(padding_eliminator).contiguous()

    # insert additional channel dimension for those that need it (and don't have one)
    ecgs_tensor = ecgs_tensor.view(subject_count, -1, 1, ecgs_tensor.size(-1))

    # squeeze the single dimension out of stages and weights
    stages_tensor = stages_tensor.squeeze()
    weights_tensor = weights_tensor.squeeze()

    collated_batch = {
        "subject_count": subject_count,
        "epoch_counts": epoch_counts_tensor,
        "max_epochs": max_epochs,
        "ecgs": ecgs_tensor,
        "additional": additional_tensor,
        "stages": stages_tensor,
        "weights": weights_tensor,
        "subject_filenames": subject_filenames,
        "load_times": load_times,
        "padding_eliminator": padding_eliminator_tensor,
        "dataset_number": dataset_number,
    }

    t_end = time.perf_counter()
    collated_batch.update({"collate_time": (t_end - t_start), "collate_end": t_end})
    return collated_batch


class LayeredCNNTree(Module):
    """A layered CNN Tree (with batchnorm, dropout, activation, cnn)"""

    def __init__(
        self,
        layer_count: int,
        channel_list,
        kernel: int,
        stride: int,
        dropout: float = 0,
        activation: bool = True,
        batch_norm: Optional[list] = None,
    ):
        """layers is number of block layers
        channels = [input, middle (all), output]"""
        super(LayeredCNNTree, self).__init__()

        if batch_norm is None:
            batch_norm = [True, True]
        assert len(channel_list) == 3

        layers: list[Module] = []

        for i in range(layer_count):
            in_channels = channel_list[0] if i == 0 else channel_list[1]
            out_channels = (
                channel_list[2] if i == (layer_count - 1) else channel_list[1]
            )

            if i == 0 and batch_norm[0]:
                layers += [BatchNorm1d(in_channels)]
            if i > 0 and batch_norm[1]:
                layers += [BatchNorm1d(in_channels)]

            # (IC layer = BN+dropout) from Chen et al 2019 (IC, weight, activation)
            if dropout > 0:
                layers += [Dropout(dropout)]

            # activation was found to work better before cnn
            if i > 0 and activation:
                layers += [LeakyReLU()]

            layers += [Conv1d(in_channels, out_channels, kernel, stride=stride)]

            # activation is typically after, but was found to work better before

        self.network = Sequential(*layers)

    def forward(self, x):
        """pytorch forward call"""
        return self.network(x)


class Chomp1d(Module):
    """Cut off the excess ends of the Tensor"""

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()

        # make sure that the chomp is even, so that cuts on both ends are equal in length
        assert (chomp_size % 2) == 0
        self.chomp_size = int(chomp_size / 2)

    def forward(self, x):
        """pytorch forward call"""
        return x[:, :, self.chomp_size : -self.chomp_size].contiguous()


class TemporalBlock(Module):
    """TCN block"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.0,
        activation: bool = True,
    ):
        super(TemporalBlock, self).__init__()

        layers: list[Module] = []

        # order has been rearranged (and dropout swapped with batchnorm) to match resnet
        layers += [BatchNorm1d(n_inputs)]

        # (IC layer = BN+dropout) from Chen et al 2019 (IC, weight, activation)
        if dropout > 0:
            layers += [Dropout(dropout)]

        # activation was found to work better before
        if activation:
            layers += [LeakyReLU()]

        # weight_norm was used in paper, but not explained why
        # removed weight_norm as the literature doesn't show that its necessary
        # (or even helpful with BN)
        layers += [
            Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        ]

        # activation is typically after, but was found to work better before

        # cuts the conv output back to the correct size
        layers += [Chomp1d(padding)]

        self.network = Sequential(*layers)

        if n_inputs == n_outputs:
            self.downsample = None
        else:
            print("downsample: ", n_inputs, n_outputs, "TemporalBlock")
            self.downsample = Conv1d(n_inputs, n_outputs, 1)

    def forward(self, x):
        """pytorch forward call"""
        out = self.network(x)

        # residual connection part
        residual = x if self.downsample is None else self.downsample(x)
        out += residual

        return out


class DenseBlockRes(Module):
    """Residual block of linear/dense layers"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        dropout: float = 0.0,
        activation: bool = True,
    ):
        super(DenseBlockRes, self).__init__()

        layers: list[Module] = []

        layers += [BatchNorm1d(n_inputs)]

        if dropout > 0:
            layers += [Dropout(dropout)]

        # activation was found to work better before
        if activation:
            layers += [LeakyReLU()]

        layers += [Linear(n_inputs, n_outputs)]

        # activation is typically after, but was found to work better before

        self.network = Sequential(*layers)

        if n_inputs == n_outputs:
            self.downsample = None
        else:
            print("downsample: ", n_inputs, n_outputs, "DenseBlockRes")
            self.downsample = Linear(n_inputs, n_outputs)

    def forward(self, x):
        """pytorch forward call"""
        out = self.network(x)

        # residual connection
        residual = x if self.downsample is None else self.downsample(x)
        out += residual

        return out


class TemporalConvNet(Module):
    """Individual TCN layer"""

    def __init__(self, layer_count: int, channel_list, dropout: float = 0.0):
        super(TemporalConvNet, self).__init__()

        assert len(channel_list) == 3
        kernel_size = 3  # fixed for TCN (always 3)
        assert kernel_size > 1
        assert kernel_size % 2 == 1

        layers: list[Module] = []

        for i in range(layer_count):
            dilation_size = 2**i  # 2=overlapping, 3=non-overlapping
            in_channels = channel_list[0] if i == 0 else channel_list[1]
            out_channels = (
                channel_list[2] if i == (layer_count - 1) else channel_list[1]
            )

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    activation=True,
                )
            ]

        self.network = Sequential(*layers)

        if channel_list[0] == channel_list[2]:
            self.downsample = None
        else:
            print("downsample: ", channel_list[0], channel_list[2], "TemporalConvNet")
            self.downsample = Conv1d(channel_list[0], channel_list[2], 1)

    def forward(self, x):
        """pytorch forward call"""
        out = self.network(x)

        # residual connection
        residual = x if self.downsample is None else self.downsample(x)
        out += residual

        return out


class LayeredDenseBlockRes(Module):
    """Layered network of dense/linear layers"""

    def __init__(
        self,
        layer_count: int,
        channel_list,
        dropout: float = 0.0,
        activation_list: Optional[list] = None,
    ):
        super(LayeredDenseBlockRes, self).__init__()

        if activation_list is None:
            activation_list = [True, True, True]

        layers: list[Module] = []

        for i in range(layer_count):
            in_channels = channel_list[0] if i == 0 else channel_list[1]
            out_channels = (
                channel_list[2] if i == (layer_count - 1) else channel_list[1]
            )
            if i == 0:
                block_activation = activation_list[0]
            elif i == (layer_count - 1):
                block_activation = activation_list[2]
            else:
                block_activation = activation_list[1]

            layers += [
                DenseBlockRes(
                    in_channels, out_channels, dropout, activation=block_activation
                )
            ]

        self.network = Sequential(*layers)

        if channel_list[0] == channel_list[2]:
            self.downsample = None
        else:
            print(
                "downsample: ", channel_list[0], channel_list[2], "LayeredDenseBlockRes"
            )
            self.downsample = Linear(channel_list[0], channel_list[2])

    def forward(self, x):
        """pytorch forward call"""
        out = self.network(x)

        # residual connection
        residual = x if self.downsample is None else self.downsample(x)
        out += residual

        return out


class SleepNet(Module):
    """The sleep-stage predicting neural network."""

    def __init__(self, net_parameters: dict, collated_sample: dict):
        super(SleepNet, self).__init__()

        # calculated properties
        epoch_input_features = (
            net_parameters["additional_features"] + net_parameters["ecg_dense_features"]
        )
        # fix the input channel count
        net_parameters["inner_dense_channel_list"][0] = epoch_input_features

        # ecg layers
        self.ecg_cnns = LayeredCNNTree(
            net_parameters["ecg_layer_count"],
            net_parameters["ecg_channels_list"],
            net_parameters["ecg_kernel"],
            net_parameters["ecg_stride"],
            dropout=net_parameters["cnn_dropout"],
            batch_norm=net_parameters["ecg_batch_norm"],
        )
        # placeholder pools
        self.ecg_avg_pool = AdaptiveAvgPool1d(net_parameters["ecg_pool_features"])
        self.ecg_max_pool = AdaptiveMaxPool1d(net_parameters["ecg_pool_features"])
        self.ecg_dense = Linear(
            2
            * net_parameters["ecg_pool_features"]
            * net_parameters["ecg_channels_list"][2],
            net_parameters["ecg_dense_features"],
        )

        # additional layers
        self.additional_dense = Linear(
            net_parameters["additional_features"], net_parameters["additional_features"]
        )

        # inner dense layers
        self.inner_dense = LayeredDenseBlockRes(
            net_parameters["inner_dense_layer_count"],
            net_parameters["inner_dense_channel_list"],
            net_parameters["dense_dropout"],
            activation_list=net_parameters["inner_dense_activation_list"],
        )

        # make sure to set the input size of tcn to the output of the inner dense
        net_parameters["tcn_channel_list"][0] = net_parameters[
            "inner_dense_channel_list"
        ][-1]

        # tcn layers
        self.tcn = TemporalConvNet(
            net_parameters["tcn_layer_count"],
            net_parameters["tcn_channel_list"],
            dropout=net_parameters["tcn_dropout"],
        )

        # correct the final output size
        net_parameters["outer_dense_channel_list"][-1] = net_parameters["stage_count"]

        # outer dense layers
        self.outer_dense = LayeredDenseBlockRes(
            net_parameters["outer_dense_layer_count"],
            net_parameters["outer_dense_channel_list"],
            net_parameters["dense_dropout"],
            activation_list=net_parameters["outer_dense_activation_list"],
        )
        # softmax for the output
        self.softmax = Softmax(1)

        # save what was used
        self.net_parameters = net_parameters

        # calculate pool layers
        self.calculate_pools(collated_sample["ecgs"])

        # init and check all weights
        self.apply(init_weights)
        check_weights(self)

    def calculate_pools(self, ecg_input: Tensor):
        """function to calculate the pools based on input sizes"""
        print("calculate adaptive pools manually:")

        # reshape same as forward function
        ecg_input = ecg_input.view(-1, ecg_input.size(-2), ecg_input.size(-1))

        # calculate stride and kernel for each
        # overwrite the pooling layers

        # ecg pools
        out = self.ecg_cnns(ecg_input)
        in_size = out.shape[-1]
        out_size = self.net_parameters["ecg_pool_features"]
        stride = in_size // out_size
        kernel = in_size - (out_size - 1) * stride
        self.ecg_avg_pool = AvgPool1d(kernel, stride)
        self.ecg_max_pool = MaxPool1d(kernel, stride)
        print(
            f"ecg      input: {in_size}  output: {out_size}  stride: {stride}  kernel: {kernel}"
        )

    def forward(
        self, ecg_input: Tensor, additional_input: Tensor, padding_eliminator: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """pytorch forward call"""

        # inputs shapes:
        #          ecg: (subjects, epochs, channels, samples)
        #   additional: (subjects, epochs, samples)
        #      padding: (subjects, epochs)
        # prior_output: (subjects, epochs, labels)
        # output shape: (subjects, epochs, labels)

        # reshape all inputs to (subjects*epochs, channels, samples)
        subject_count = ecg_input.size(0)
        ecg_input = ecg_input.view(-1, ecg_input.size(-2), ecg_input.size(-1))
        # no channel dimension for additional input
        additional_input = additional_input.view(-1, additional_input.size(-1))

        # ecg layers
        out = self.ecg_cnns(ecg_input)
        ecg_avg = self.ecg_avg_pool(out)
        ecg_max = self.ecg_max_pool(out)
        # flatten and concat
        combined = torch.cat(
            (ecg_avg.view(ecg_avg.size(0), -1), ecg_max.view(ecg_max.size(0), -1)), 1
        )
        ecg_out = self.ecg_dense(combined)

        # additional layers
        add_out = self.additional_dense(additional_input)

        # "inner dense / feature-extractor dense"
        # input: (subjects*epochs, features)
        # output: (subjects*epochs, features)
        features_in = torch.cat((add_out, ecg_out), 1)
        out = self.inner_dense(features_in)

        # reshape from (subjects*epochs, features) to (subjects, epochs, features)
        out = out.view(subject_count, -1, out.size(-1))

        # zero out the features for the padded epochs
        # this is to prevent modifying the feature layers for padded epochs
        # expand to number of features
        features_out = out * padding_eliminator.expand(-1, -1, out.size(2))

        # permute to (subjects, features, epochs)
        out = features_out.permute(0, 2, 1)

        # tcn layers (also includes residual from input all the way to the output)
        out = self.tcn(out)

        # permute to (subject, epochs, features)
        out = out.permute(0, 2, 1)

        # reshape from (subjects, epochs, features) to (subjects*epochs, features)
        out = out.contiguous().view(-1, out.size(-1))  # must use contiguous here

        # "outer dense" layers (also includes residual from input all the way to the output)
        out = self.outer_dense(out)

        # apply softmax to the label (dim=1)
        labels_out = self.softmax(out)

        # reshape from (subjects*epochs, labels) to (subjects, epochs, labels)
        labels_out = labels_out.view(subject_count, -1, labels_out.size(-1))

        # both returns should be in (subjects, epochs, labels/features)
        return labels_out, features_out

    def forward_with_dict(self, data_dict: dict) -> Tuple[Tensor, Tensor]:
        return self.forward(
            data_dict["ecgs"], data_dict["additional"], data_dict["padding_eliminator"]
        )


def move_tensors_to_device(data_dict: dict, device: torch.device) -> dict:
    """move all tensors in dict to device"""

    if device.type == "cpu":
        return data_dict
    
    tensor_list = ["ecgs", "additional", "padding_eliminator", "stages", "weights"]
    for key in tensor_list:
        data_dict[key] = data_dict[key].to(device, non_blocking=(device.type == "cuda"))

    # wait for everything to finish
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    return data_dict
