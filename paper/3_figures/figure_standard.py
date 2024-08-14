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



# code to standardize everything possible about the figures

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.colors import LinearSegmentedColormap


def standardize_plot_settings():
    # standardize all of the plot settings

    # set the theme and font scale first
    font_scale = 7 / (12 * 0.8)  # default is 12, but 7 is required
    sns.set_theme(
        style="whitegrid", context="paper", font_scale=font_scale, font="Arial"
    )

    # round everything to 2 decimal places, and set again
    context = sns.plotting_context()
    for key in context.keys():
        context.update({key: round(context[key], 2)})
    sns.set_context(context)

    # basic formatting stuff
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["pdf.fonttype"] = 42  # required for submission
    plt.rcParams["figure.figsize"] = (3.2, 3)  # just a dummy default size

    # use Symbol font for math
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Symbol"

    # additional formatting stuff
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["axes.edgecolor"] = "black"


def create_main_palette() -> list[tuple[float, float, float]]:
    # create the main palette

    main_palette = [sns.desaturate(sns.color_palette("Blues", 5)[3], 0.75)]
    for i in range(5):
        main_palette += [sns.color_palette("Dark2", 5, desat=1)[i]]
    return main_palette


def create_stage_cmaps() -> list[LinearSegmentedColormap]:
    # create the color maps for each stage

    main_palette = create_main_palette()
    stage_cmaps = []
    for i in range(5):
        end_color = main_palette[i + 1]
        temp_cmap = sns.blend_palette(
            ["#FFFFFF", end_color], n_colors=30, as_cmap=False
        )
        stage_cmaps += [sns.blend_palette([temp_cmap[1], end_color], as_cmap=True)]
    return stage_cmaps


def create_blue_cmap() -> LinearSegmentedColormap:
    # create the color map to use for all-blue figures
    # this uses the main blue as the near-end target, and I manually tweaked the end point

    cmap1 = sns.blend_palette(
        ["#FFFFFF", "#4884AF"], n_colors=21, as_cmap=False, input="hex"
    )
    blue_cmap = sns.blend_palette(
        [cmap1[1], cmap1[5], cmap1[10], "#4884AF", "#35607F"], as_cmap=True
    )
    return blue_cmap


def scale_figure_by_axes(axes: Axes, height: float, width: float):
    # scale the figure to get a desired y-axis length

    desired_left_height = height  # inches
    left_points = axes.spines["left"].get_window_extent()._points
    left_height = (left_points[1, 1] - left_points[0, 1]) / axes.figure.dpi
    y_scale = desired_left_height / left_height

    desired_bottom_length = width  # inches
    bottom_points = axes.spines["bottom"].get_window_extent()._points
    bottom_length = (bottom_points[1, 0] - bottom_points[0, 0]) / axes.figure.dpi
    x_scale = desired_bottom_length / bottom_length

    figure_size = axes.figure.get_size_inches()
    axes.figure.set_size_inches(figure_size[0] * x_scale, figure_size[1] * y_scale)


def save_figure_files(
    filename: str = "",
    save_pdf: bool = True,
    pdf_transparent: bool = True,
    save_png: bool = True,
    png_transparent: bool = False,
):

    if len(filename) == 0:
        raise ValueError("filename cannot be empty")

    if save_pdf:
        plt.savefig(
            f"{filename}.pdf",
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=pdf_transparent,
        )
    if save_png:
        plt.savefig(
            f"{filename}.png",
            bbox_inches="tight",
            pad_inches=0.05,
            transparent=png_transparent,
        )
