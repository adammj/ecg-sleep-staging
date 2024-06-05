# Expert-level ECG-only sleep staging

This repository contains all the code and data used in the work by Adam M. Jones, Laurent Itti, and Bhavin R. Sheth (<https://doi.org/10.1016/j.compbiomed.2024.108545>).

If you find this repository helpful, please cite our work:

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545

---

1. [Sleep scoring your own data](#sleep-scoring-your-own-data)
2. [Data file description](#data-file-description)
3. [Benchmark dataset](#benchmark-dataset)
4. [Loss function](#loss-function)
5. [Paper replication](#paper-replication)
6. [Requirements](#requirements)

---

## Sleep scoring your own data

Below are all the steps necessary to score sleep on your own ECG data.

### data processing

While all of the code that was used for everything described in the paper are in the `dataset` directory, the code was originally designed around processing thousands of files in parallel in specific steps (which I found easier to write in MATLAB at the time).
To process your own data, you can take it through that same pipeline (either a file at a time or many files simultaneously). Or, you can instead just extract your own ECG and filter it (as described in the paper and in the `Data file description` section below).
Eventually, I will rewrite the pipeline to accommodate the processing of individual files and convert all MATLAB code to Python. This code will go in the `data_processing` folder (which currently just has a placeholder file).

### sleep stage scoring

I have provided a complete folder for both the `primary` and `real-time` models. They are completely self-contained, and, as a consequence, are duplicates of files inside the `network_and_training` folders. However, the parameters in the `train_params.json` file are set up to sleep score on individual files right away. The code can run with or without a GPU.

To use either model, just run the following from your python environment:

```
python train.py your_datafile.mat
```

The `your_datafile.mat` can either be in the same folder, or elsewhere (as long as the complete path is provided). The code will load the appropriate model, check the file, score the sleep, and save a `results.mat` file in the same folder.

If you need any assistance, please feel free to contact me (contact details provided in the paper). I will be happy to help you use and modify the code to work on your own data, as well as replicate anything from the paper.

---

## Data file description

Each file, representing a single night of sleep (or a portion thereof) that the network code will ingest, should be provided in HDF5 files with the following `datasets` (the term HDF5 uses for variables):

All five variables are required for the loader to operate. However, only the first three are necessary to perform inference (just scoring instead of training). For scoring only, the remaining two variables could be provided as arrays of random numbers of the correct dimensions.

- `ecgs`:
  - 2D array of floats (size: epoch_count x 7680)
  - Where 7680 = 30 x 256Hz.
  - Network was trained on ECG data:
    - High-pass filtered (0.5 Hz).
    - Scaled or clipped to -/+ 1, with 90% of the maximum R waves (or other tallest wave in the heartbeat) within + or - 0.5.
- `demographics`:
  - 2D array of floats (size: 2 x 1):
    - 1st: sex (0=female, 1=male)
    - 2nd: age (age/100)
- `midnight_offset`:
  - A float that represents the first epoch's clock time where 0 = midnight and -1 = 24hr before midnight and 1 = 24hr after midnight.
- `stages`:
  - 2D array of floats (size: epoch_count x 1):
    - Common 0=Wake to 4=REM mapping
    - All "unscored" epochs should be marked as 0.
- `weights`:
  - 2D array of floats (size: epoch_count x 1):
    - 0 (no weight) to 1 (full weight)
    - All "unscored" epochs should be given a weight of 0.

---

## Benchmark dataset

The study included training, validating, and testing on a dataset of 4,000 recordings that were randomly sampled from five different source studies. Additionally, a held-out study was used to evaluate any study-specific learning. Although we do not have permission to share the source data, they are available at the National Sleep Research Resource (<https://sleepdata.org/>). To facilitate the creation and use of a standardized benchmark (which this field sorely needs), we have provided a listing of all of the file names so that others can train, validate, and test the exact same dataset we used.

1. **benchmark_dataset**
   - `main_sets.xlsx` contains the listing for the train, validation, and testing sets
   - `heldout_set.xlsx` contains the listing for the held-out study

---

## Loss function

A new Cohen's-kappa-correlated loss function was designed for this work, which is included in the self-contained model folders. However, its main repository (with looser licensing) is at (<https://github.com/adammj/loss-functions>).

---

## Paper replication

Below is a description of the contents of each folder.

### Dataset

1. **dataset_preprocessing**
   - Contains all of the pre-processing code that was used

### Network and Training

We have included the necessary code and weights to duplicate the training and testing of the final model exactly.

1. **network_and_training_code**
   - A self-contained directory of the code, including:
     - `adams.py` (optimizer, citation, and source link inside)
     - `main_sets.xlsx` (duplicate sets file—see above)
     - `net_params.json` (network hyperparameters)
     - `sleep_support.py` (various support functions)
     - `sleepdataset.py` (dataset loader and processing)
     - `sleeploss.py` (loss function)
     - `sleepnet.py` (network and batch collation)
     - `train_params.json` (training hyperparameters)
     - `train.py` (main program)
   - To run, call: `python train.py`
2. **network_weights**
   - The log and network weights of the final trained model.
   - The weights can be loaded using the `"resume_checkpoint"` and `"save_results_and_exit"` keys in the `train_params.json` file.
3. **real-time_network**
   - Only the files that are different from the primary model are included here.

### Paper

All of the code and intermediate results that were used to evaluate the final model are included here.

1. **1_meta_analysis**
   - The inputs, R code, and output for the meta-analysis.
2. **2_intermediate_data**
   - The intermediate data that was created from the various experiments described and plotted in the paper.
3. **3_figures**
   - All of the figure files (Jupyter notebooks and one Keynote file). Each notebook will produce a PDF and PNG file for each figure.
   - The scatter plot in Figure 8 requires a large (280MB) file, `tsne_results.mat`. It was not uploaded due to GitHub limitations. Please contact us if you would like a copy of the file.

---

## Requirements

The toolbox and package requirements to run the code are as follows:

- MATLAB (2023a)
  - Parallel Computing Toolbox (7.8)
  - Signal Processing Toolbox (9.2)
- Python
  - I've provided three different `requirements.txt` files, depending on your needs. If you just want to score sleep using one of the models, then you only need to `requirements_cpu.txt` file. However, if you would like to train the model, or otherwise make use of your GPU, then use `requirements_gpu.txt`. Finally, if you want to reproduce the figures from the paper, use `requirements_paper.txt`.
- R (4.3.2)
  - DescTools (0.99.54)
  - pimeta (1.1.3)
  - readxl (1.4.3)

---

Copyright (C) 2024 Adam M. Jones

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
