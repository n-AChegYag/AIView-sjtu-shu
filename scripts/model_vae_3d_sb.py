import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

class ResizeConv3d(nn.Module):
    # 上采样-->conv
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale_factor, bias=False, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    # 编码器残差块
    # relu(x + [311conv-->bn-->relu-->311conv-->bn](x))
    # relu([120conv-->bn](x) + [321conv-->bn-->relu-->311conv-->bn](x))
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    # 解码器残差块
    # relu(x + [311conv-->bn-->relu-->311conv-->bn](x))
    # relu([上采样-->110conv-->bn](x) + [上采样-->311conv-->bn-->relu-->311conv-->bn](x))
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv3d(in_planes, planes, kernel_size=3, stride=1, padding=1, scale_factor=stride)
            self.bn1 = nn.BatchNorm3d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv3d(in_planes, planes, kernel_size=1, stride=1, padding=0, scale_factor=stride),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, img_size=64, num_blocks=[1,1,1,1], z_dim=512, nc=3):
        super().__init__()
        self.in_planes = 32     # 输入通道数
        self.z_dim = z_dim
        z_size = img_size // (2**len(num_blocks))
        layers = [nn.Conv3d(nc, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False), 
                  nn.BatchNorm3d(self.in_planes),
                  nn.ReLU(False)]
        layers += self._make_layer(BasicBlockEnc, self.in_planes, num_blocks[0], stride=1)
        
        fm_size = img_size // 2
        
        for c in num_blocks[1:]:
            layers += self._make_layer(BasicBlockEnc, self.in_planes*2, c, stride=2)
            fm_size = fm_size // 2
        
        assert fm_size == z_size
        
        if self.z_dim is not None:
            layers += [nn.Conv3d(self.in_planes, self.in_planes//8, kernel_size=1, stride=1, padding=0, bias=False),
                       nn.BatchNorm3d(self.in_planes//8),
                       nn.ReLU(False),]
            
            self.linear = nn.Linear((self.in_planes//8)*(fm_size**3), 2*z_dim)
        else:
            layers += [nn.Conv3d(self.in_planes, self.in_planes*2, kernel_size=1, stride=1, padding=0, bias=False)]
        
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return layers

    def forward(self, x):                           # (bs, 1, 64, 64, 64)
        x = self.layers(x)
        if self.z_dim is not None:
            x = self.linear(x.view(x.size(0),-1))
        mu, logvar = torch.chunk(x, 2, dim=1)           
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, img_size=64, num_blocks=[1,1,1,1], z_dim=512, nc=3):
        super().__init__()
        
        self.in_planes = 32 * (2**(len(num_blocks)-1))
        self.z_dim = z_dim
        self.z_size = img_size // (2**len(num_blocks))
        
        fm_size = self.z_size
        layers = []
        if self.z_dim is not None:
            self.linear = nn.Linear(z_dim, (self.in_planes//8)*fm_size**3)
            layers += [nn.Conv3d(self.in_planes//8, self.in_planes, kernel_size=1, stride=1, padding=0, bias=False),
                       nn.BatchNorm3d(self.in_planes),
                       nn.ReLU(False),]
        else:
            layers += [nn.Conv3d(self.in_planes, self.in_planes, kernel_size=1, stride=1, padding=0, bias=False)]
          
        for c in num_blocks[:0:-1]:
            layers += self._make_layer(BasicBlockDec, self.in_planes//2, c, stride=2)
            fm_size *= 2
            
        layers += self._make_layer(BasicBlockDec, self.in_planes, num_blocks[0], stride=1)
        layers += [ResizeConv3d(self.in_planes, nc, kernel_size=3, stride=1, padding=1, scale_factor=2)]
        
        fm_size *= 2 
        assert fm_size == img_size
        
        self.layers = nn.Sequential(*layers)
        
        
    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):                       # (bs, 512)
        if self.z_dim is not None:
            z = self.linear(z)
            z = z.view(z.size(0), -1, self.z_size, self.z_size, self.z_size)
        x = self.layers(z)
        return x


class VAE(nn.Module):

    def __init__(self, img_size=64, num_blocks=[1,1,1,1], z_dim=None, nc=1):
        super().__init__()

        self.encoder = ResNet18Enc(img_size=img_size, num_blocks=num_blocks, z_dim=z_dim, nc=nc)
        self.decoder = ResNet18Dec(img_size=img_size, num_blocks=num_blocks, z_dim=z_dim, nc=nc)
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        # x = self.decoder(z.clone())
        x = torch.tanh(self.decoder(z))
        return x, z, mean, logvar
    
    def rec_loss(self, x_r, x, xc=None, idc=None, fill=False):
        
        bs = x.size(0)
        loss = torch.pow(x_r-x, 2)
        # if xc is not None:
        #     loss[:, :, idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] *= 0
        #     if fill:
        #         loss[:, :, idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]] += torch.pow(x_r[:, :, idc[0]:idc[1], idc[2]:idc[3], idc[4]:idc[5]]-xc, 2)
        loss_rec = loss.sum() / bs
        mean_loss_rec = loss.mean()      
        return loss_rec, mean_loss_rec
        
    
    def z_loss(self, z, x_r):
        mu_r, logvar_r = self.encoder(x_r)
        z_r = self.reparameterize(mu_r, logvar_r)
        loss = torch.pow(z-z_r, 2).sum()
        return loss / z.size(0)
    
    def kl_loss(self, z_mu, z_logvar, min_kl_loss=None):
        loss = 0.5 * (-z_logvar + torch.exp(z_logvar) + torch.pow(z_mu, 2) - 1).sum()
        bs = z_mu.size(0)
        if min_kl_loss is not None:
            loss = F.relu(loss / bs - min_kl_loss) + min_kl_loss
        else:
            loss = loss / bs
        return loss
    
    def tv_loss(x):
        bs = x.size0
        tv = torch.sqrt(torch.sum(torch.pow(x[:, :, 1:, :-1, :-1]-x[:, :, :-1, :-1, :-1], 2) 
                                   + torch.pow(x[:, :, :-1, 1:, :-1]-x[:, :, :-1, :-1, :-1], 2)
                                   + torch.pow(x[:, :, :-1, :-1, 1:]-x[:, :, :-1, :-1, :-1], 2))
                        + 1e-8)
        return tv / bs
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            epsilon = torch.randn_like(std)
            return epsilon * std + mean
        else:
            return mean



    
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)

        return out

  
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.linear = nn.Sequential(
            Linear(latent_dim, 1000),
            Linear(1000, 1000),
            Linear(1000, 1000),
            nn.Linear(1000, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out



class AAE(nn.Module):

    def __init__(self, z_dim, nc=3):
        super().__init__()

        self.encoder = ResNet18Enc(z_dim=z_dim, nc=nc)
        self.decoder = ResNet18Dec(z_dim=z_dim, nc=nc)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, z
    
    def rec_loss(self, x_r, x):
        loss = torch.pow(x_r-x, 2)
        bs = x.size(0)
        rec_loss = loss.sum() / bs
        mean_rec_loss = loss.mean()
        return rec_loss, mean_rec_loss
    
    def z_reg_loss(self, z, x_r):
        mu_r, logvar_r = self.encoder(x_r.detach())
        z_r = self.reparameterize(mu_r, logvar_r)
        loss = torch.pow(z-z_r, 2).sum()
        return loss / z.size(0)
    
    def dist_loss(self, z_real, z_fake):
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z_fake)
        d_loss = -0.02 * torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
        return d_loss
    
    def enc_loss(self, z_fake):
        d_fake = self.discriminator(z_fake)
        enc_loss = -torch.mean(torch.log(d_fake + 1e-8))
        return enc_loss
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            epsilon = torch.randn_like(std)
            return epsilon * std + mean
        else:
            return mean