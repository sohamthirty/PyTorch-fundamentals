# PyTorch-fundamentals

Repository for documenting my progress on learning PyTorch.

## Installation
- • Install PyTorch version-Stable(2.0.1) for CUDA 11.7 from **[PyTorch.org](https://pytorch.org/get-started/locally/)** in virtual environment using Anaconda Prompt.
- • Check if torch package can be imported, is CUDA available and version of CUDA.

## Tensor Basics
- • In PyTorch, Tensor is just like a numpy array and acts as the fundamental datastructure.
- • It is a generic n-dimensional array which is used for numeric computation.

## Autograd
- • Creates a computational graph to calculate gradients in BackPropogation.

## Back Propagation
- • During the training of the network, it calculates and adjusts the weights based on the error between the predicted output and the desired output. The goal is to minimize this error and improve the network's performance.
- • It contains 1. Forward Pass, 2. Computing Local Gradients 3. Backward Pass: compute dLoss/dWeights using chain rule and computational graph.

![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/4d8b758a-a8a3-48f7-9c25-137a93a9f53e)
