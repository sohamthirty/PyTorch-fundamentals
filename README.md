# PyTorch-fundamentals

Repository for documenting my progress on learning PyTorch.</br>
Checkout the article here: **[Fundamentals of PyTorch](https://sohamthirty.hashnode.dev/fundamentals-of-pytorch)**

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


## Softmax layer & Cross-Entropy loss:
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/fc17708c-60a9-4aa7-a474-44b6565e0bc0)
- • **Softmax Layer** applies the exponential function to each element & normalizes them by the sum of all exponentials.</br>
• it converts logits to probabilities such that the output is between 0 and 1. (highest logit has highest probability)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/1774404e-930f-4486-99f2-82d41e42064b)

- • **Cross-Entropy loss** is generally combined with softmax function. (bad prediction means high cross-entropy loss)</br>
• PyTorch already implements softmax layer along with Cross-Entropy loss.</br>
• here, Y is not one-hot encoded and Y_pred is logits and not softmax normalized values.

- • **Neural Net classification**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/aa0119cb-f3a7-4aa9-b0d1-ba639f015246)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/b9573071-04b9-429a-ad53-6a05b5468fb9)
  

## Activation functions:
- • Activation functions apply a Non-Linear transformation and decide whether a neuron should be activated or not.</br>
• without Activation functions, the network would not learn anything and just be a linear model.
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/61812d44-a058-4357-a603-66a787f57ef5)

1) **Step function** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/6c1ecf1e-4a8d-445d-81bf-0b26204eadcf)

2) **Sigmoid** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/e4834096-b518-4b07-8fd4-2b468fc593a4)
- – used in last layer of binary classification problem. (outputs probability between 0 and 1)

3) **TanH** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/d09088ba-35a4-45c0-bdab-c46804d3f963)
- – used for hidden layers. (Scaled & Shifted with values between -1 and 1)

4) **ReLU** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/dfeaaedb-dcd0-48c7-8cf0-f2870a04e915)
- – used for hidden layers. (outputs 0 for -ve values & the value for +ve values)

5) **Leaky ReLU** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/657d8f01-f901-4f66-a926-64ada220e876)
- – solves vanishing gradient problem. (multiplies the output for -ve values by some small values)

6) **Softmax** : ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/7e860c1b-0b5b-42d5-b2db-d624bdb7cdec)
- – used in last layer of multi-class classification problem. (outputs probabilities between 0 and 1)


## Feed Forward Neural Network:
- • **Device Configuration**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/95ece859-a02e-4622-99e4-0da52dfc7e84)
- • **Hyperparameters**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/914b12c4-d335-4180-9e66-7464cb2f504b)
- • **DataLoader**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/273b2c9f-157e-471c-93aa-049400e7c89c)
- • **Model**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/046935c4-44da-45a8-8de9-b3cbea7d7dbd)
- • **Training loop**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/1cc35a9f-e3ca-4cf3-ba2c-f5a41e871c29)
- • **Test**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/dd14de87-c0af-4feb-9313-312bbb8a11ac)


## Convolution Neural Network:
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/b683a16a-aa40-4191-98ee-5addb5708621)
- • **Hyperparameters**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/95070429-5374-4740-87b4-212715cbe347)
- • **DataLoader**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/29241a9f-903d-4633-aa01-6bd8935b0cd8)
- • **Model**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/353ac5b8-b847-47ad-9a8b-0f22cfe806c2)


## Transfer Learning:
- • A pre-trained model, trained on a large dataset, is utilized as a starting point for a new task, typically by reusing and fine-tuning the learned features.
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/a821a7c4-4e90-4fee-b215-8542432b9a27)

- • **DataLoader**</br>
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/28f88221-76eb-47c0-837b-9ebc834ce165)

- • **Model**</br>
• model.state_dict() contains the learnable parameters, such as weights and biases organized as a Python dictionary.
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/a4489bd1-6ab8-4309-bb3b-41d3c8d26920)

- • **ResNet**</br>
• Freeze the network except the final layer. (Load a pre-trained model and reset final fully connected layer)</br>
• set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().</br>
• Learning rate scheduling should be applied after optimizer’s update.
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/965633b1-0e0c-448f-af2f-57a39415978e)


## TensorBoard:
• provides the visualization and tooling needed for machine learning experimentation.</br>
• pip install tensorboard</br>
• tensorboard --logdir=runs</br>
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/11309416-e3dc-4408-99d7-70798605deb4)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/cd28f84b-5859-4d0c-8f0c-b7d25d9dd264)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/454155a7-0570-4660-bd40-dd9daea5d1b8)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/75bd3275-9155-4e57-ad42-2035ae95645d)
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/a6c32634-a76b-4e68-8aff-b5f372837447)</br></br>
• **Visuals:**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/e1aaeb85-7513-414d-a50b-6513a24cfb6c)


## Save & Load models:
• **1. lazy way: save whole model**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/e6431400-098f-47a2-ae30-acf643711e6f)
- 
• **2. recommended way: save only the state_dict**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/2dec3b30-c005-4df3-b8d9-e62da2444237)
- 
• **3. Checkpoints**
- ![image](https://github.com/sohamthirty/PyTorch-fundamentals/assets/56295513/b62d6bc9-da7b-4e9a-b7c3-aaf3fb7943f4)


## References:
- **1. [PyTorch Tutorials | PyTorch.org](https://pytorch.org/tutorials/)**
- **2. [Deep Learning With PyTorch - Full Course | Patrick Loeber](https://youtu.be/c36lUUr864M)**
