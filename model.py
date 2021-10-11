# create model
import torch.nn as nn
class modelCNN(nn.Module):
    def __init__(self):
        super(modelCNN,self).__init__()
        self.cv1 = nn.Conv2d(in_channels=1,out_channels=10,
                             kernel_size=5,stride=1,padding=2)
        self.act1 = nn.ReLU()
        self.cv2 = nn.Conv2d(in_channels=10,out_channels=15,
                             kernel_size=3,stride=1,padding=1)
        self.act2 = nn.ReLU()
        self.fc = nn.Linear(in_features=28*28*15,out_features=10)
    def forward(self,x):
        out = self.cv1(x)
        out = self.act1(out)
        out = self.cv2(out)
        out = self.act2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
