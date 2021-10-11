#create Feedforward Neural Network
import torch.nn as nn
class model(nn.Module):
    def __init__(self,input_size,hidden_size,out_classes):
        super(model,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,out_classes)
    def forward(self,x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out
