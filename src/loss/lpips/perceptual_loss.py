'''
Adapted from the official implementation of PerceptualSimilarity
Only keep the [net='alex', version='0.1', lpips=True, ]
'''
import os
import torch
import torch.nn as nn
from torchvision import models as tv

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat / (norm_factor + eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    return nn.functional.interpolate(in_tens, size=out_HW, mode='bilinear', align_corners=False)

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, spatial=False, use_dropout=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        spatial = True, Return a spatial map of perceptual distance.
        """
        super(LPIPS, self).__init__()
        self.spatial = spatial
        self.scaling_layer = ScalingLayer()

        self.chns = [64,192,384,256,256]
        self.net = alexnet(pretrained=True)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = nn.ModuleList([self.lin0,self.lin1,self.lin2,self.lin3,self.lin4])

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'weights/alex.pth'))
        self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        self.eval()
    
    def forward(self, in0, in1, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1
        
        # reapeat into 3 channels
        if in0.shape[1] == 1:           
            in0 = in0.repeat(1,3,1,1)
            in1 = in1.repeat(1,3,1,1)

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        
        diffs = [(normalize_tensor(out0) - normalize_tensor(out1)) ** 2 for out0, out1 in zip(outs0, outs1)]

        if self.spatial:
            res = [upsample(self.lins[kk](diff), out_HW=in0.shape[2:]) for kk, diff in enumerate(diffs)]
        else:
            res = [spatial_average(self.lins[kk](diff), keepdim=True) for kk, diff in enumerate(diffs)]

        val = sum(res)
        return torch.mean(val)

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class alexnet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        
        self.slices = nn.ModuleList([
            nn.Sequential(*alexnet_pretrained_features[:2]),
            nn.Sequential(*alexnet_pretrained_features[2:5]),
            nn.Sequential(*alexnet_pretrained_features[5:8]),
            nn.Sequential(*alexnet_pretrained_features[8:10]),
            nn.Sequential(*alexnet_pretrained_features[10:12])
        ])
        for param in self.parameters():
            param.requires_grad = False
                
    def forward(self, X):
        h = X
        outputs = []
        for slice in self.slices:
            h = slice(h)
            outputs.append(h)
        return outputs

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)