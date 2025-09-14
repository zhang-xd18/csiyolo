import torch
from pathlib import Path, PosixPath
import torch.nn as nn
from .modules import *

__all__ = ["create_model", "EnvNet"]

def create_model(cfg='config.yaml'):
    if isinstance(cfg, PosixPath):
        cfg = str(cfg)
    if isinstance(cfg, str) and cfg.endswith('.yaml'):
        import yaml
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)    
    model = EnvNet(hidden_width=cfg['hidden_width'], padding_mode=cfg['padding_mode'], factorization=cfg['factorization'])
    return model


class EnvNet(nn.Module):
    def __init__(self, hidden_width=5, padding_mode='circular', factorization=False):
        super(EnvNet, self).__init__()
        # Backbone
        # Only use the real and imag part of the channel
        self.multiconv1 = MultiConv(in_channels=2, out_channels=hidden_width, 
                                    kernel_in=3, kernel_1=7, kernel_2=9, padding_mode=padding_mode, shortcut=True, factorization=factorization)
        self.conv1 = ConvBN(in_channels=hidden_width, out_channels=hidden_width*2, 
                            kernel_size=3, stride=2, padding_mode=padding_mode, factorization=factorization)
        self.multiconv2 = MultiConv(in_channels=hidden_width*2, out_channels=hidden_width*2, 
                                    kernel_in=3, kernel_1=5, kernel_2=7, padding_mode=padding_mode, shortcut=True, factorization=factorization)
        self.conv2 = ConvBN(in_channels=hidden_width*2, out_channels=hidden_width*4, 
                            kernel_size=3, stride=2, padding_mode=padding_mode, factorization=factorization)
        self.multiconv3 = MultiConv(in_channels=hidden_width*4, out_channels=hidden_width*4, 
                                    kernel_in=3, kernel_1=3, kernel_2=5, padding_mode=padding_mode, shortcut=True, factorization=factorization)
        self.conv3 = ConvBN(in_channels=hidden_width*4, out_channels=hidden_width*2, 
                            kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        # Neck
        self.upcatconv1 = UpCatConv(in_channels=hidden_width*2, out_channels=hidden_width, 
                                    kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        self.upcatconv2 = UpCatConv(in_channels=hidden_width, out_channels=hidden_width, 
                                    kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        self.downcatconv1 = DownCatConv(in_channels=hidden_width, out_channels=hidden_width*2, 
                                        kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        self.downcatconv2 = DownCatConv(in_channels=hidden_width*2, out_channels=hidden_width*4, 
                                        kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        # Head
        self.head1 = HeadConv(in_channels=hidden_width, out_channels=2+1, 
                              kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        self.head2 = HeadConv(in_channels=hidden_width*2, out_channels=2+1, 
                              kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        self.head3 = HeadConv(in_channels=hidden_width*4, out_channels=2+1, 
                              kernel_size=3, stride=1, padding_mode=padding_mode, factorization=factorization)
        
        
    def forward(self, x):
        # Backbone
        x1 = self.multiconv1(x)
        x2 = self.multiconv2(self.conv1(x1))
        x3 = self.multiconv3(self.conv2(x2))
        # Neck
        x3_2 = self.conv3(x3)
        x2_2 = self.upcatconv1(x3_2, x2)
        x1_2 = self.upcatconv2(x2_2, x1)
        x1_3 = x1_2
        x2_3 = self.downcatconv1(x1_3, x2_2)
        x3_3 = self.downcatconv2(x2_3, x3_2)
        # Head
        out1 = self.head1(x1_3)
        out2 = self.head2(x2_3)
        out3 = self.head3(x3_3)
        
        out = [out1, out2, out3]
        
        if not self.training:
            z = []
            for i in range(len(out)):
                bs, _, ny, nx = out[i].shape
                d, t = out[i].device, out[i].dtype
                grid = self._make_grid(nx, ny, d, t)
                stride = (torch.tensor(x.shape[-2:]) // torch.tensor(out[i].shape[-2:])).to(x.device)
                
                out[i] = out[i].sigmoid()
                xy, conf = out[i][:,:2,:,:], out[i][:,2,:,:]
                xy = (xy * 2 + grid) * stride.view(1,2,1,1)
                y = torch.cat((xy, conf.unsqueeze(1)), 1)
                z.append(y.view(bs, 3, -1))
                
        return out if self.training else (torch.cat(z, 2),)


    def _make_grid(self, nx, ny, d, t):
        shape = 1, ny, nx
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 0) - 0.5 
        return grid