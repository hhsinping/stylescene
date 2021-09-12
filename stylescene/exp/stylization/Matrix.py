import torch
import torch.nn as nn
import numpy as np
import sys


from stylization.pointnet2_ssg_cls import PointNet2ClassificationSSG


class CNN(nn.Module):
  def __init__(self,layer,matrixSize=32):
    super(CNN,self).__init__()
    self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(128,64,3,1,1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(64,matrixSize,3,1,1))
    self.pointnet = PointNet2ClassificationSSG()
    self.fc1 = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
    self.fc2 = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)


  def forward(self,feat,img,xyz):
    feat = feat.reshape(1,256,-1)
    outx = self.convs(img)
    b,c,h,w = outx.size()
    outx = outx.view(b,c,-1)
    outx = torch.bmm(outx,outx.transpose(1,2)).div(h*w)
    outx = outx.view(outx.size(0),-1)
    
    skip = int(feat.shape[-1] // 100000)
    xyz = xyz[:,::skip,:].contiguous()
    feat = feat[:,:,::skip].contiguous()
    outy = self.pointnet(xyz, feat)
    
    b,c,p = outy.size()
    outy = outy.view(b,c,-1)
    outy = torch.bmm(outy,outy.transpose(1,2)).div(p)
    outy = outy.view(outy.size(0),-1)

    return self.fc1(outx), self.fc2(outy)


class MulLayer(nn.Module):
  def __init__(self,layer,matrixSize=32):
    super(MulLayer,self).__init__()
    self.cnet = CNN(layer,matrixSize)
    self.matrixSize = matrixSize
    self.compress = nn.Conv2d(256,matrixSize,1,1,0)
    self.unzip = nn.Conv2d(matrixSize,256,1,1,0)
    self.transmatrix = None

  def forward(self,cF,sF, points,trans=True):
    cF = cF.reshape(1,256,-1,1)
    cMean = torch.mean(cF,dim=2,keepdim=True)
    cMeanC = cMean.expand_as(cF)
    cF = cF - cMeanC
     
    sMean = torch.mean(sF.view(1,256,-1,1),dim=2,keepdim=True)
    sMeanC = sMean.expand_as(cF)
    sMeanS = sMean.expand_as(sF)
    sF = sF - sMeanS

    compress_pointcloud = self.compress(cF).view(1,32,-1)

    cMatrix, sMatrix = self.cnet(cF, sF, points) 
    sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
    cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
    transmatrix = torch.bmm(sMatrix,cMatrix)

    compress_pointcloud = torch.bmm(transmatrix,compress_pointcloud).view(1,32,-1)
    feat = (self.unzip(compress_pointcloud.view(1,32,-1,1)) + sMeanC).squeeze(-1)
    return feat
