import torch
import torchvision 
from torch import optim
from torchvision import transforms,datasets
import torch.nn as nn
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=15,kernel_size=5,
                                           stride=1,padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc = nn.Linear(in_features=15*14*14,out_features=10)
    def forward(self,x):
        out = self.cv1(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
def train_loop(train_loader,num_epochs,device,loss_f,optimizer,model,val_loader):
    times = 0
    patience =3
    the_last_loss = 0
    for epoch in range(1,num_epochs+1):
        model.train()
        train_loss = 0
        for i,(images,labels) in enumerate(train_loader):
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            out_put = model(images)
            loss = loss_f(out_put,labels)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
        print('epoch ',epoch,'/',num_epochs,'- loss : ',train_loss)
        the_current_loss = validation(val_loader,device,model,loss_f)
        print('the current loss : ',the_current_loss)
        if the_current_loss > the_last_loss:
            times+=1
            print('times : ',times)
            if times>=patience:
                print('early stopping !')
                return model
        else :
            print('times = 0')
            times = 0
        the_last_loss = the_current_loss
    return model
def validation(val_loader,device,model,loss_f):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels = images.to(device),labels.to(device)
            out_put = model(images)
            loss = loss_f(out_put,labels)
            val_loss +=loss.item()
    return val_loss/len(val_loader)
def  test(test_loader,device,model):
    model.eval()
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            _,predict = torch.max(output,1)
            correct+=(predict==labels).sum().item()
            total_samples+=labels.size(0)
        print('accuracy :',correct/total_samples)
def main():
    #device 
    device = torch.device('cpu')
    simple_model = model().to(device)
    batch_size=64
    num_epochs = 10
    learning_rate = 0.002
    # data loader
    train_set = datasets.MNIST(root='data',train=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(0.1307,0.3081)]),
                               download=True)
    test_set = datasets.MNIST(root='data',train=False,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize(0.1307,0.3081)]),
                                download=True)
    # split train_set into train & val
    train_set_size = int(len(train_set)*0.75)
    val_set_size = len(train_set)-train_set_size
    train_set,val_set = torch.utils.data.random_split(train_set,[train_set_size,val_set_size])
    #load
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle = False)
    #loss function 
    loss_f=nn.CrossEntropyLoss()
    #optimizer
    optimizer = optim.Adam(simple_model.parameters(),lr=learning_rate)
    train_loop(train_loader,num_epochs,device=device,loss_f=loss_f,optimizer=optimizer,model=simple_model,val_loader=val_loader)
    test(test_loader=test_loader,device=device,model=simple_model)
if __name__ == '__main__':
    main()