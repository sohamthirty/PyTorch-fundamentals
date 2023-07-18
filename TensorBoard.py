import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

############## TENSORBOARD ########################

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist2") # specify where the log files will be saved

###################################################

import sys
import matplotlib.pyplot as plt


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# hyperparameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 200
learning_rate = 0.01



# DataLoader
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

data_iter = iter(train_loader)
samples, labels = next(data_iter)
print(samples.shape, labels.shape)



# Add images to tensorboard
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

img_grid = torchvision.utils.make_grid(samples) # arrange images into grid
writer.add_image('mnist images', img_grid)

# writer.close()
# sys.exit()



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # dont apply softmax here due to cross entropy
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)



# loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model graph
writer.add_graph(model, samples.reshape(-1, 28*28).to(device))

# writer.close()
# sys.exit()

# training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_corrects = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1 , 28, 28
        # we need 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        
        # loss
        loss = criterion(outputs, labels)
        
        # backward pass
        loss.backward()
    
        # update
        optimizer.step()
    
        # zero gradients
        optimizer.zero_grad()
        
        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        running_corrects += (pred == labels).sum().item()
        
        # print running values
        if (i+1)%100 == 0:
            print('Epoch {}/{}: step {}/{}, loss = {:.4f}'.format(epoch, num_epochs, i+1, n_total_steps, loss))
            writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps + i)
            writer.add_scalar('accuracy', running_corrects/100, epoch*n_total_steps + i)
            running_loss = 0.0
            running_corrects = 0


# test

labels = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
    for i, (images, lab) in enumerate(test_loader):
        images = images.reshape(-1, 28*28).to(device)
        lab = lab.to(device)
        
        outputs = model(images)
        
        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += lab.size(0)
        n_correct += (predictions == lab).sum().item()
        
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
        labels.append(predictions)
        
        
    labels = torch.cat(labels)
    # for each class
    preds = torch.cat([torch.stack(batch) for batch in preds])
        
acc = 100 * n_correct/n_correct
print('Accuracy: ', acc)


classes = range(10)
for i in classes:
    labels_i = labels==i
    preds_i = preds[:, i]
    
    # add Precision-Recall curve for each label class
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    
    writer.close()

sys.exit()