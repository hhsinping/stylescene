import torch
import torch.nn as nn

class encoder3(nn.Module):

  def __init__(self):
    super(encoder3,self).__init__()
    self.conv1 = nn.Conv2d(3,3,1,1,0)
    self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
    self.conv2 = nn.Conv2d(3,64,3,1,0)
    self.relu2 = nn.ReLU(inplace=True)
    self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
    self.conv3 = nn.Conv2d(64,64,3,1,0)
    self.relu3 = nn.ReLU(inplace=True)
    self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
    self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
    self.conv4 = nn.Conv2d(64,128,3,1,0)
    self.relu4 = nn.ReLU(inplace=True)
    self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
    self.conv5 = nn.Conv2d(128,128,3,1,0)
    self.relu5 = nn.ReLU(inplace=True)
    self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
    self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
    self.conv6 = nn.Conv2d(128,256,3,1,0)
    self.relu6 = nn.ReLU(inplace=True)

  def forward(self,x):
    out = self.conv1(x)
    out = self.reflecPad1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.reflecPad3(out)
    out = self.conv3(out)
    pool1 = self.relu3(out)
    out,pool_idx = self.maxPool(pool1)
    out = self.reflecPad4(out)
    out = self.conv4(out)
    out = self.relu4(out)
    out = self.reflecPad5(out)
    out = self.conv5(out)
    pool2 = self.relu5(out)
    out,pool_idx2 = self.maxPool2(pool2)
    out = self.reflecPad6(out)
    out = self.conv6(out)
    out = self.relu6(out)
    return out

