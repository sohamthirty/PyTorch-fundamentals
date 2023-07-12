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

## Gradient Descent
- • It is an optimization algorithm that iteratively adjusts weights in the direction of steepest descent of the loss function to find the minimum.
- • Run epochs for n_iters times with a learning-rate: use forward(x), loss(y, y_pred), gradient using Autograd, Update the weights (not a part of the computational graph) & ensure to make gradients zero at the end of each epoch.
- • import torch.nn as nn


## Training Pipeline:
1) Design Model (input, output size, forward pass) using torch.nn
2) Construct loss and optimizer
3) Training Loop
    - forward pass  : compute prediction
    - backward pass : gradients ([w,b] = model.parameters())
    - update weights (optimizer.step())

## Linear Regression:
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/9af22054-0753-4549-b7f4-2c553ba24d56)


## Logistic Regression:
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/c06a5f12-293d-4ab0-80a4-24a2a86ab145)


## Dataset & Dataloader:
- • It is time-consuming to do gradient calculatuions on training data of large datasets, so we divide samples into smaller batches.</br>Do loop over epoch: loop over batches -> optimization based only on the batches.

- • epoch = one forward and backward pass of ALL training samples
- • batch_size = number of training samples used in one forward/backward pass
- • number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of samples
- e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/084c1e9d-fdb9-4cf8-807b-a924a2ac692c)


## Transforms:
- • it allows to modify data in various ways before feeding it into the model.
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/adcff2f4-afe8-4b56-b325-284256edc1ce)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/cfad5cf5-2d79-4d72-ba2e-14b16b1abdef)


