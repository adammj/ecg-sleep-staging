---
title: "Cardiosomnography:<br>ECG-only sleep staging"
layout: default
---

# Introduction

It is now possible to score sleep, at equivalent performance, to expert human-scored polysomonography (PSG) using only electrocardiography (ECG). Our method offers an inexpensive, automated, and convenient alternative for sleep stage classification—further enhanced by a real-time scoring option.

Our intention is that cardiosomnography (CSG), or a sleep study conducted with ECG only, could take expert-level sleep studies outside the confines of clinics and laboratories and into realistic settings. It makes less-expensive, higher-quality studies accessible to a broader community, enabling improved sleep research and more personalized, accessible sleep-related healthcare interventions.

Read the paper to find out more (<https://doi.org/10.1016/j.compbiomed.2024.108545>):

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024

# Sleep scoring your own data

Both the primary (after-the-fact) and real-time models are provided in the repository in self-contained folders.

# Preparing your data

# Paper replication

# Requirements

I've provided three different `requirements.txt` files, depending on your needs. If you just want to score sleep using one of the models, then you only need to `requirements_cpu.txt` file. However, if you would like to train the model, or otherwise make use of your GPU, then use `requirements_gpu.txt`. Finally, if you want to reproduce the figures from the paper, use `requirements_paper.txt`.

# Cite

If you find this repository helpful, please cite our work:

> Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545
