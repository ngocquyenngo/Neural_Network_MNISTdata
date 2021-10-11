import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch import optim

#parameters
num_epochs = 5
learning_rate = 0.005
batch_size = 100
# load data 
mnist_train = torchvision.datasets.MNIST(root='data',train=True,
                                         transform = transforms.ToTensor(),
                                         download=True)
mnist_test = torchvision.datasets.MNIST(root='data',train=False,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False)
# get device
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
from model import modelCNN
model = modelCNN().to(device)
# loss function
loss_f = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
num_steps = len(train_loader)
for epoch in range(num_epochs):
    # training loop
    model.train()
    total_loss=0
    for i,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device),labels.to(device)
        #forward
        output = model(images)
        #compute loss
        loss = loss_f(output,labels)
        total_loss+=loss.item()
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print('epoch :',epoch+1,'/',num_epochs,'- step: ',i+1,'/',num_steps,'- loss: ',total_loss/(i+1))
    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        correct = 0
        total_samples = 0
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            _,predict = torch.max(output,1)
            correct +=(predict==labels).sum().item()
            total_samples+=labels.size(0)
            loss = loss_f(output,labels)
            val_loss+=loss.item()
        print('epoch ',epoch+1,'- accuracy : ',100*correct/total_samples,'- validation loss :',val_loss/len(test_loader))
    
        