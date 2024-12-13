# Expert-level ECG-only sleep staging

We propose the term “**cardiosomnography**” (**CSG**) for any sleep study that could be conducted using only electrocardiography (ECG/EKG) data. Our intention is for CSG to take more expert-level sleep studies outside the confines of clinics and laboratories and into realistic settings. By eliminating the need for the most cumbersome equipment and a human scorer, it makes less-expensive, higher-quality studies more widely accessible.

This repository contains all the code and data used in the paper (<https://doi.org/10.1016/j.compbiomed.2024.108545>), which demonstrates expert-level, five-stage, sleep scoring. It also contains the code for you to score your own data.

If you find this repository helpful, please cite the paper:

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545

Also, make sure to check out the website, [cardiosomnography.com](https://cardiosomnography.com), as I'm keeping a blog for updates there.

---

1. [Sleep scoring your own data](#sleep-scoring-your-own-data)
2. [Data file description](#data-file-description)
3. [Benchmark dataset](#benchmark-dataset)
4. [Loss function](#loss-function)
5. [Paper replication](#paper-replication)
6. [Using CUDA or CPU or MPS](#using-cuda-or-cpu-or-mps)
7. [Requirements](#requirements)
8. [Paper links](#paper-links)
9. [Contact](#contact)

---

## Sleep scoring your own data

To get this into the hands of others and in use as soon as possible, there is a separate folder structure just for new and future users (verses documenting what was done for the paper, further below).

**Please read the [Requirements](<https://cardiosomnography.com/requirements/>) page** on the website for more details on the ECG requirements—especially if you have a background in machine learning and view ECG as merely a 1D image, or if you assume this model relies on HRV.

### Data processing

While all of the code that was used for everything described in the paper is in the `paper` folder, that code was originally designed around processing thousands of files in parallel in specific steps (which was easier to write in MATLAB at the time).
To process your own data, you can take it through that same pipeline (either a file at a time or many files simultaneously). Or, you can instead just extract your own ECG/EKG data and filter it (as described in the paper and in the `Data file description` section below).
Eventually, I will rewrite the pipeline to accommodate the processing of individual files and convert all MATLAB code to Python. This code will go in the `data_processing` folder (which currently just has a placeholder file).

If you're planning on using your own pipeline, please see the [Requirements](<https://cardiosomnography.com/requirements/>) page on the website for more details about the basic steps.

### Sleep stage scoring models

#### Primary or Real-Time model

There is a folder for each model: `primary` and `real-time`. They are completely self-contained, and, as a consequence, are almost exact duplicates. However, the parameters in the `train_params.json` file are set up to sleep score on individual files right away. The code can run with or without a GPU.

To use either model, just run the following from your python environment:

```
python train.py your_datafile.h5
```

The `your_datafile.h5` can either be in the same folder, or elsewhere (as long as the complete path is provided). The code will load the appropriate model, check the file, score the sleep, and save a `results.h5` file in the same folder.

#### Primary model without demographics (new)

A new primary model that does not require the subject's demographics (age and sex) is now available. The Cohen's kappa for this model on the testing set is 0.718 (which is >99% of the performance of the primary model with demographics). To clarify, this model will only need the variables `ecgs` and `midnight_offset` to perform inference.

This model was created after the paper was published, and is therefore not discussed in the paper.

### FYI: description of files inside model folders

- `adams.py` (optimizer, original source here: [zeke-xie/stable-weight-decay-regularization](<https://github.com/zeke-xie/stable-weight-decay-regularization>))
- `net_params.json` (network hyperparameters)
- `sleep_support.py` (various support functions)
- `sleepdataset.py` (dataset loader and processing)
- `sleeploss.py` (loss function)
- `sleepnet.py` (network and batch collation)
- `train_params.json` (training hyperparameters)
- `train.py` (main program)
- `[the name is unique].pt` (the weights for the model)

---

## Data file description

Each file, representing a single night of sleep (or a portion thereof) that the network code will ingest, should be provided in HDF5 files with the following `datasets` (the term HDF5 uses for variables):

If just performing inference (only scoring data, and not training), then only the first three variables are required for the loader to operate. However, all five variables are necessary if training the network.

- `ecgs`:
  - 2D array of floats (size: epoch_count x 7680)
  - Where 7680 = 30 x 256Hz.
  - Network was trained on raw ECG data that had been filtered and scaled appropriately. See **Expectations for ECG input** above.
- `demographics`:
  - 2D array of floats (size: 2 x 1):
    - 1st: sex (0=female, 1=male)
    - 2nd: age (age_in_years/100)
- `midnight_offset`:
  - A float that represents the clock time offset to the nearest midnight of when the recording began:
    - 0 = midnight
    - -1 = 24hr before midnight and 1 = 24hr after midnight
    - For example, 9pm = -0.125 and 3am = 0.125.
- `stages` (only required for training):
  - 2D array of floats (size: epoch_count x 1):
    - Stage mapping: 0=Wake, 1=N1/S1, 2=N2/S2, 3=N3/S3/S4, 4=REM.
      - It is not uncommon to find REM mapped to 5. However, the network was trained with data with both AASM and R&K scorings, so a unified "deep sleep" score was mapped to 3. And because it's inconvenient to have a gap, REM was remapped to 4.
    - All "unscored" epochs should be mapped to 0 (also see weight below).
- `weights` (only required for training):
  - 2D array of floats (size: epoch_count x 1):
    - 0 (no weight) to 1 (full weight)
    - All "unscored" epochs should be given a weight of 0.

---

## Benchmark dataset

The study included training, validating, and testing on a dataset of 4,000 recordings that were randomly sampled from five different source studies. Additionally, a held-out study was used to evaluate any study-specific learning. Although I do not have permission to share the source data, they are available at the National Sleep Research Resource (<https://sleepdata.org/>). To facilitate the creation and use of a standardized benchmark (which this field sorely needs), I have provided a listing of all of the file names so that others can train, validate, and test the exact same dataset I used.

And, please, for the sake of proper and valid science, **DO NOT** test against the testing set until **after** you have selected your final model. So many papers mess this up. If you are using the testing set to choose which model to use, you are leaking data that is supposed to represent the model's performance on an unbiased, unseen, dataset back into the development of the model. **If you do this, your results are sus**.

### Benchmark dataset folder:

- `main_sets.xlsx` contains the listing for the train, validation, and testing sets
- `heldout_set.xlsx` contains the listing for the held-out study

---

## Loss function

A new Cohen's-kappa-correlated loss function was designed for this work, which is included in the self-contained model folders. However, its main repository (with looser licensing) is at (<https://github.com/adammj/loss-functions>).

---

## Paper replication

All of the code and intermediate results that were used to evaluate the final model are included here.

0. **0_data_processing**
   - Contains all of the pre-processing code that was used. As described above, this was written in MATLAB with the goal of processing thousands of files in specific steps. The files can (fairly easily) be modified to process a single file at a time. However, I am working on a streamlined Python version to future users.
1. **1_meta_analysis**
   - The inputs, R code, and output for the meta-analysis.
2. **2_intermediate_data**
   - The intermediate data that was created from the various experiments described and plotted in the paper.
3. **3_figures**
   - All of the figure files (Jupyter notebooks and one Keynote file). Each notebook will produce a PDF and PNG file for each figure.
   - The scatter plot in Figure 8 requires a large (280MB) file, `tsne_results.mat`. It was not uploaded due to GitHub limitations. Please contact me if you would like a copy of the file.

---

## Using CUDA or CPU or MPS

In running the scaling tests described above, I noticed some differences when running the model on CUDA vs CPU vs MPS (Apple GPUs). The results are nearly identical, but understand that the results will be slightly different if you are not running on an NVIDIA GPU with CUDA.

- **CUDA**: This is what the model was trained and evaluated on.
- **CPU**: This works, but the kappas are slightly lower on average (by -0.003, or -0.3%).
- **MPS**: This works, but the kappas are slightly lower on average (by -0.003, or -0.3%).

**Note:** The differences between CUDA, CPU, and MPS are well below being statistically meaningful.

Interestingly, the CPU and MPS results are nearly identical to each other. Only 8 of the 500 recordings in the testing set had different predictions between those two backends, and it was for only one epoch in each of those recordings.

(FYI: This is probably because I tested both on the same M1 MacBook Air. However, I would expect slightly different CPU results if I ran it on an Intel processor, for instance. This all comes down to differences in and tradeoffs made with different floating point representations and calculations. Although the CPU/MPS results are slightly below on one processor and compiled library combination, they could easily be higher on another. And these differences are not even approaching being statistically meaningful.)

**Same stage predictions for testing set**

Overall, 98.2% of the 571,141 scored epochs in the testing set have the same prediction when inference is performed on CUDA versus CPU/MPS. As stated above, there were only 8 epochs of the 571,141 scored epochs where the CPU and MPS predictions differed.

| Statistic | Same (CUDA v CPU) |  Same (CUDA v MPS) |
| ----- | ----- | ----- |
| Max | 99.9% | 99.9%  |
| Median | 98.4% | 98.4% |
| Mean | 98.2% | 98.2% |
| Min | 94.2% | 94.2% |

**Median and Mean kappas for testing set**
The CUDA values are reported in Paper (Supplementary Table S7).

|  | Overall | Wake | N1 | N2 | N3 | REM |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **Median** of recordings | | | | | |
| CUDA | 0.725 | 0.871 | 0.326 | 0.682 |0.625 | 0.825 |
| CPU | 0.724 | 0.868 | 0.327 | 0.674 | 0.630 | 0.826 |
| MPS | 0.724 | 0.868 | 0.327 | 0.674 | 0.630 | 0.826 |
| **Mean** of recordings | | | | | |
| CUDA | 0.697 | 0.830 | 0.333 | 0.651 | 0.505 | 0.777 |
| CPU | 0.694 | 0.829 | 0.331 | 0.646 | 0.508 | 0.775 |
| MPS | 0.694 | 0.829 | 0.331 | 0.646 | 0.508 | 0.775 |

---

## Requirements

The toolbox and package requirements to run the code are as follows:

- Python
  - I've provided three different `requirements.txt` files, depending on your needs. These were created with [conda](<https://docs.anaconda.com/miniconda/miniconda-install/>), as this is the package manager I use for the more complex requirements of [PyTorch](<https://pytorch.org/get-started/locally/>). 
  - If you just want to score sleep using one of the models, then you only need to `requirements_cpu.txt` file. However, if you would like to train the model, or otherwise make use of your GPU, then use `requirements_gpu.txt`. Finally, if you want to reproduce the figures from the paper, use `requirements_paper.txt`.
- MATLAB (2023a)
  - Parallel Computing Toolbox (7.8)
  - Signal Processing Toolbox (9.2)
- R (4.3.2)
  - DescTools (0.99.54)
  - pimeta (1.1.3)
  - readxl (1.4.3)

---

## Paper links

- [Published paper](<https://doi.org/10.1016/j.compbiomed.2024.108545>)
    - Contact details provided on this website and inside the paper.
- [Preprint version](<https://www.medrxiv.org/content/10.1101/2023.10.13.23297018v1>)
    - There were some large changes between the preprint and the final, published, version (mainly revolving around adding a large meta-analysis). However, the network and testing set results are the same.

---

## Contact

If you need any assistance, please feel free to file an issue in the repository or contact me (contact details provided in the paper and on the journal's website). I will be happy to help you use and modify the code to work on your data, as well as replicate anything from the paper.

---

Copyright (C) 2024 Adam Jones. All Rights Reserved.

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
