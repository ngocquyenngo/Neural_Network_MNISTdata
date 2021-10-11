import torch
import torchvision 
from torchvision import transforms
import torch.nn as nn
from torch import optim

#parameters
batch_size = 100
num_epochs = 2
learning_rate = 0.001
input_size = 784
hidden_size = 500
out_classes = 10
# load data
mnist_train = torchvision.datasets.MNIST('data',train = True,
                                         download = True,
                                         transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('data',train =False,
                                        download = True,
                                        transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train,
                                           batch_size=batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(mnist_test,
                                          batch_size = batch_size,
                                          shuffle = False)

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model_FF import model
modelFF = model(input_size,hidden_size,out_classes).to(device)
# loss function
loss_func = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(modelFF.parameters(),lr=learning_rate)
#training loop
num_step = len(train_loader)
for epoch in range(num_epochs):
    modelFF.train()
    total_loss = 0
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        #forward
        output = modelFF(images) 
        #compute loss
        loss = loss_func(output,labels)
        total_loss+=loss.item()
        #optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100==0:
            print('epoch ',epoch+1,'/',num_epochs,'- step : ',i+1,'/',num_step,'-loss :',total_loss/(i+1))
    # validation
    modelFF.eval()
    val_loss=0
    with torch.no_grad():
        correct =0
        total = 0
        for images,labels in test_loader:
            images = images.reshape(-1,28*28).to(device)
            labels = labels.to(device)
            output = modelFF(images)
            _,predict = torch.max(output,1)
            total+=labels.size(0)
            correct += (predict==labels).sum().item()
            loss = loss_func(output,labels)
            val_loss+=loss.item()
        print('epoch ',epoch+1,'- validation loss : ',val_loss/len(test_loader),'- accuracy : ',100*correct/total,'%')
