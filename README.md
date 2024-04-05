# Expert-level sleep staging using an electrocardiography-only feed-forward neural network

This repository contains all the code and data used in the work (**insert-final-URL**) by Adam M. Jones, Laurent Itti, and Bhavin R. Sheth.

If you find this repository helpful, please cite our work:
> **insert-final-citation**

---

Below is a description of the contents of each folder.

## Dataset
The study included training, validating, and testing on a dataset of 4,000 recordings that were randomly sampled from five different source studies. Additionally, a held-out study was used to evaluate any study-specific learning. Although we do not have permission to share the source data, they are available at the National Sleep Research Resource (https://sleepdata.org/). To facilitate the creation and use of a standardized benchmark (which this field sorely needs), we have provided a listing of all of the file names, so that others can train, validate, and test on the exact same dataset we used.

1. **dataset_files**
    - `main_sets.xlsx` contains the listing for the train, validation, and testing sets 
    - `heldout_set.xlsx` contains the listing for the held-out study
2. **dataset_preprocessing**
    - Contains all of the pre-processing code that was used

## Network and Training
We have included the necessary code and weights to exactly duplicate the training and testing of the final model.

1. **network_and_training_code**
    - A self-contained directory of the code, including:
	    - `adams.py` (optimizer, citation and source link inside)
	    - `main_sets.xlsx` (duplicate sets file—see above)
	    - `net_params.json` (network hyperparameters)
	    - `sleep_support.py` (various support functions)
	    - `sleepdataset.py` (dataset loader and processing)
	    - `sleeploss.py` (loss function)
	    - `sleepnet.py` (network and batch collation)
	    - `train_params.json` (training hyperparameters)
	    - `train.py` (main program)
    - To run, call:  `python train.py`
2. **network_weights**
    - The log and network weights of the final trained model.
    - The weights can be loaded using the `"resume_checkpoint"` and `"save_results_and_exit"` keys in the `train_params.json` file.

## Paper
All of the code and intermediate results that were used to evaluate the final model is included here.

1. **1_meta_analysis**
    - The inputs, R code, and output for the meta analysis.
2. **2_intermediate_data**
    - The intermediate data that was created from the various experiments described and plotted in the paper.
3. **3_figures**
    - All of the figure files (Jupyter notebooks and one Keynote file). Each notebook will produce a pdf and png file for each figure.
    - The scatter plot in Figure 8 requires a large (280MB) file, `tsne_results.mat`. It was not uploaded due to GitHub limitations. Please contact us if you would like a copy of the file.

---

## Requirements
The toolbox and package requirements to run the code are as follows:

- MATLAB (2023a)
    - Parallel Computing Toolbox (7.8)
    - Signal Processing Toolbox (9.2)
- Python
    - Provided in requirements.txt file.
- R (4.3.2)
    - DescTools (0.99.54)
    - pimeta (1.1.3)
    - readxl (1.4.3)

---

Copyright (C) 2024  Adam M. Jones

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
