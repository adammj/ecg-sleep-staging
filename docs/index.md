---
title: "Cardiosomnography: <br>ECG-only sleep studies"
layout: default
---

# Introduction to Cardiosomnography

We propose the term "**cardiosomnography**" (**CSG**) for any sleep study that could be conducted with only electrocardiography (ECG/EKG). In addition to expert-level sleep staging [[1](#citations)], numerous studies have demonstrated that sleep apnea detection can be reliably performed using only ECG [[2,3,4](#citations)].

# Sleep Staging with CSG

We demonstrate in our recent [paper](<https://authors.elsevier.com/a/1j58I2OYd7gUK>) [[1](#citations)] that it is now possible to score sleep at equivalent performance to expert human-scored polysomnography (PSG) using only ECG. Our method offers an inexpensive, automated, and convenient alternative for sleep stage classification—further enhanced by a real-time scoring option.

We suggest that everyone interested should read the [paper](<https://authors.elsevier.com/a/1j58I2OYd7gUK>) to find out more. (Make sure to also check out the supplemental—I had too many results for the main text! 😭) However, here's a brief summary figure of our meta-analysis results:

<div style="text-align: center;">
<b>Meta-analysis comparison of PSG- and CSG-based sleep staging:</b>
<img src="assets/meta-analysis.png" alt="Meta-analysis comparison of PSG- and CSG-based sleep staging">
</div>

The meta-analysis of 11 papers on PSG sleep staging, demonstrate the on-par results of CSG with the current gold standard. See our [paper](<https://authors.elsevier.com/a/1j58I2OYd7gUK>) for more details.

# Applications of CSG

Our intention is for CSG to take expert-level sleep studies outside the confines of clinics and laboratories and into realistic settings. Eliminating the need for most of the cumbersome equipment and a human scorer it makes less-expensive, higher-quality studies accessible to a broader community. We hope this will enable improved sleep research and more personalized, accessible sleep-related healthcare interventions.

# Code Repository

<a href="{{ site.github.repository_url }}" class="btn">View on GitHub</a>

The [GitHub repository](https://github.com/adammj/ecg-sleep-staging) contains everything you need to do the following:

1. Prepare your data
2. Sleep score your data
3. Use the benchmark dataset
4. Use the loss function for your own models
5. Replicate everything in the paper's Methods and Results sections

If you need any assistance, please feel free to contact me (contact details provided in the paper). I will be happy to help you use and modify the code to work on your data, as well as replicate anything from the paper.

# References

If you find this repository helpful, please cite our work:

- [1] [Adam M. Jones, Laurent Itti, Bhavin R. Sheth, "Expert-level sleep staging using an electrocardiography-only feed-forward neural network," Computers in Biology and Medicine, 2024, doi: 10.1016/j.compbiomed.2024.108545.](<https://authors.elsevier.com/a/1j58I2OYd7gUK>)

Additional works cited above:

- [2] [T. Wang, J. Yang, Y. Song, F. Pang, X. Guo, and Y. Luo, "Interactions of central and autonomic nervous systems in patients with sleep apnea–hypopnea syndrome during sleep," Sleep Breath, vol. 26, pp. 621–631, 2022, doi: 10.1007/s11325-021-02429-6.](<https://doi.org/10.1007/s11325021-02429-6>)
- [3] [T. Penzel, J. McNames, P. de Chazal, B. Raymond, A. Murray, and G. Moody, "Systematic comparison of different algorithms for apnoea detection based on electrocardiogram recordings," Medical and Biological Engineering and Computing, vol. 40, no. 4, pp. 402–407, Jul. 2002, doi: 10.1007/bf02345072.](<https://doi.org/10.1007/bf02345072>)
- [4] [H. Hilmisson, N. Lange, and S. P. Duntley, "Sleep apnea detection: accuracy of using automated ECG analysis compared to manually scored polysomnography (apnea hypopnea index)," Sleep Breath, vol. 23, no. 1, pp. 125–133, Mar. 2019, doi: 10.1007/s11325-018-1672-0.](<http://doi.org/10.1007/s11325-018-1672-0>)

# Online Mentions

- [NSRR (Twitter/X)](<https://x.com/sleepdatansrr/status/1805564095875498136?s=46>)
- [NSRR (Sleepdata.org)](<https://sleepdata.org/blog/2024/06/expert-level-sleep-staging-using-an-electrocardiography-only-feed-forward-neural-network>)
- [NSRR (YouTube)](<https://youtu.be/qRTqVaaiX6s>)
- [UH Newsroom](<http://uscholars.uh.edu/news-events/stories/2024/july/07022024-sheth-sleep-staging-monitoring.php>)
