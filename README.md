# Neural network unrolling of greedy weighted sparse recovery algorithms
Code used to generate the figures of the article: 
* **main paper:** [*Deep greedy unfolding: Sorting out argsorting in greedy sparse recovery algorithms*](https://arxiv.org/abs/2303.00844) by *Sina Mohammad-Taheri, Matthew J. Colbrook, and Simone Brugiapaglia*.
* [*OMP-Net: Neural network unrolling of weighted orthogonal mathcing pursuit*](https://arxiv.org/abs/2303.00844) by *Sina Mohammad-Taheri, Matthew J. Colbrook, and Simone Brugiapaglia*.

## Article's abstract
Gradient-based learning imposes (deep) neural networks to be differentiable at all steps. This includes model-based architectures constructed by unrolling iterations of an iterative algorithm onto layers of a neural network, known as algorithm unrolling. However, greedy sparse recovery algorithms depend on the non-differentiable argsort operator, which hinders their integration into neural networks. In this paper, we address this challenge in Orthogonal Matching Pursuit (OMP) and Iterative Hard Thresholding (IHT), two popular representative algorithms in this class. We propose permutation-based variants of these algorithms and approximate permutation matrices using "soft" permutation matrices derived from softsort, a continuous relaxation of argsort. We demonstrate--both theoretically and numerically--that Soft-OMP and Soft-IHT, as differentiable counterparts of OMP and IHT and fully compatible with neural network training, effectively approximate these algorithms with a controllable degree of accuracy. This leads to the development of OMP- and IHT-Net, fully trainable network architectures based on Soft-OMP and Soft-IHT, respectively. Finally, by choosing weights as "structure-aware" trainable parameters, we connect our approach to structured sparse recovery and demonstrate its ability to extract latent sparsity patterns from data.

## Algorithms
The package contains the following algorithms:
* **Orthogonal Matching Pursuit (OMP):** G. M. Davis, S. G. Mallat, and Z. Zhang, *Adaptive time-frequency decompositions*, Optical Engineering, 33 (1994), pp. 2183–2191.
* **Iterative Hard Thresholding (IHT):** T. Blumensath and M. E. Davies, *Iterative thresholding for sparse approximations*, Journal of Fourier analysis and Applications, 14 (2008), pp. 629–654.

## How to use this code
### Python dependencies
The scripts depends on the following python packages:
* Pytorch, Matplotlib, numpy, math, sys, os
### Code organization
<pre>
script directory/
|
├── main_fig_1.py            # main code to generate figure 1: Sort vs. Soft-sort
├── models/                  # including algorithms and models
│   ├── algs_module.py       # including functions for algorithms in the package
│   └── nnet_module.py       # including modules to build unrolled neural networks in the package
│
├── utils/
│   ├── data_gen.py          # including data classes to generate the training and validation data for each unrolled network
│   ├── my_config.py         # configs required in the training of networks, used in train_module
│   ├── sort_proxy.py        # including soft-(arg)sorting functions to generate approximate permutation matrices 
│   └── train_module.py      # including the trainer function for training networks
│
├── OMP/                     # main folder for unrolling OMP algorithm
│   ├── main_fig_2_OMP.py    # main code to generate figure 2: difference error vs. tau (recovery error vs. tau in the second paper)
│   ├── main_fig_3_OMP.py    # main code to build and train neural networks generating figure 3: MSE-Loss, oracle weights, learned weights and relative ℓ2-error boxplots
│   ├── data/                # data folder to store data of figures 2 and 3
│       ├── fig_2/           # the folder contains data to generate figure 2
│       └── fig_3/           # the folder contains the data to generate figure 3: dataset, checkpoints, best checkpoint dictionary
│           └── checkpoints/ # contains checkpoints saved during the training procedure
│   └── figs/                # the folder contains figures in the main paper
│       ├── fig_2/
│       └── fig_3/
│
├── IHT/
│   ├── main_fig_2_IHT.py
│   .
│   .
│   .
.
.
.
</pre>

### main_fig_2
Experimental setup:
0 for recovery error vs. $\tau$ and 1 for difference error vs. $\tau$.
### main_fig_3
  
## Disclaimer
This code is provided as a complementary resource for readers and without warranty of any kind. The code may not be suitable for all purposes and the user assumes all risks associated with its use.
