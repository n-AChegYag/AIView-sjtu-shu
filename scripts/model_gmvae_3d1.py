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

    def __init__(self, num_Blocks=[1,1,1,1], z_dim=512, w_dim=512, nc=1):
        super().__init__()
        self.in_planes = 32
        self.z_dim = z_dim
        self.conv1 = nn.Conv3d(nc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.layer1 = self._make_layer(BasicBlockEnc, 32, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[3], stride=2)
        self.conv2 = nn.Conv3d(256, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.linear = nn.Linear(32*4**3, 2 * z_dim + 2 * w_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):                           # (bs, 1, 64, 64, 64)
        x = torch.relu(self.bn1(self.conv1(x)))     # (bs, 32, 32, 32, 32)
        x = self.layer1(x)                          # (bs, 32, 32, 32, 32)
        x = self.layer2(x)                          # (bs, 64, 16, 16, 16)
        x = self.layer3(x)                          # (bs, 128, 8, 8, 8)
        x = self.layer4(x)                          # (bs, 256, 4, 4, 4)
        x = torch.relu(self.bn2(self.conv2(x)))     # (bs, 32, 4, 4, 4)
        x = x.view(x.size(0), -1)                   # (bs, 32*4**3)
        x = self.linear(x)                          # (bs, 512*2)
        z_mu, z_logvar = torch.chunk(x[:, :2*self.z_dim], chunks=2, dim=1)
        w_mu, w_logvar = torch.chunk(x[:, 2*self.z_dim:], chunks=2, dim=1)
        
        return z_mu, z_logvar, w_mu, w_logvar


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[1,1,1,1], z_dim=512, nc=1):
        super().__init__()
        self.in_planes = 256
        # self.nc = nc
        self.linear = nn.Linear(z_dim, 32*4**3)
        self.conv2 = nn.Conv3d(32, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(256)
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv3d(32, nc, kernel_size=3, stride=1, padding=1, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):                       # (bs, 512)
        x = self.linear(z)                      # (bs, 32*4**3)
        x = x.view(z.size(0), -1, 4, 4, 4)      # (bs, 32, 4, 4, 4)
        x = torch.relu(self.bn2(self.conv2(x))) # (bs, 256, 4, 4, 4)
        x = self.layer4(x)                      # (bs, 128, 8, 8, 8)
        x = self.layer3(x)                      # (bs, 64, 16, 16, 16)
        x = self.layer2(x)                      # (bs, 32, 32, 32, 32)
        x = self.layer1(x)                      # (bs, 32, 32, 32, 32)
        x = self.conv1(x)                       # (bs, 1, 64, 64, 64)

        return torch.tanh(x)


class GMVAE(nn.Module):

    def __init__(self, z_dim, w_dim, c_dim, nc=1):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        
        self._Qwz_x = ResNet18Enc(z_dim=z_dim, w_dim=w_dim, nc=nc)
        
        self._Px_z = ResNet18Dec(z_dim=z_dim, nc=nc)
        
        self.fc_z_wc = nn.Linear(w_dim, z_dim*2)
        self.fc_z_mus = nn.ModuleList()
        self.fc_z_logvars = nn.ModuleList()
        for i in range(c_dim):
            self.fc_z_mus.append(nn.Linear(z_dim*2, z_dim))
            self.fc_z_logvars.append(nn.Linear(z_dim*2, z_dim))
        self.z_mus_list = list()
        self.z_logvars_list = list()
            
            
    def Qzw_x(self, x):
        z_mu, z_logvar, w_mu, w_logvar = self._Qwz_x(x)
        return z_mu, z_logvar, w_mu, w_logvar


    def Px_z(self, z): 
        x = self._Px_z(z)
        return x


    def Pz_wc(self, w):
        h = self.fc_z_wc(w)
        self.z_mus_list = []
        self.z_logvars_list = []
        for i, l in enumerate(self.fc_z_mus):
            self.z_mus_list.append(l(h))
        for i, l in enumerate(self.fc_z_logvars):
            self.z_logvars_list.append(l(h))

        return self.z_mus_list, self.z_logvars_list


    def Qc_z(self, z, z_mus_list, z_logvars_list):  
        z_mus_stack = torch.stack(z_mus_list, dim=1)           # (bs, c_dim, z_dim)
        z_logvars_stack = torch.stack(z_logvars_list, dim=1)
        qc = self.gaussian_pdf_log(z.unsqueeze(1), z_mus_stack, z_logvars_stack) - np.log(np.pi)
        qc = F.softmax(qc.sum(-1), dim=1)
        return qc


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            epsilon = torch.randn_like(std)
            return epsilon * std + mu
        else:
            return mu
    
    def rec_term(self, x, x_r):
        bs = x.size(0)
        return torch.pow(x - x_r, 2).sum() / bs
        
    
    def kl_z(self, z_mu, z_logvar, z_mus_list, z_logvars_list, qc):
        # KL  = 1/2(logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1)
        # qc: (bs, c_dim), z_mu: (bs, z_dim), z_logvar: (bs, z_dim)
        z_mus_stack = torch.stack(z_mus_list, dim=1)           # (bs, c_dim, z_dim)
        z_logvars_stack = torch.stack(z_logvars_list, dim=1)
    
        kl = 0.5 * torch.sum(z_logvars_stack 
                             - z_logvar.unsqueeze(1) 
                             + z_logvar.exp().unsqueeze(1)/z_logvars_stack.exp()
                             + (z_mu.unsqueeze(1)-z_mus_stack).pow(2)/z_logvars_stack.exp()
                             -1, 2)
        # print('kl ', kl.mean())
        # print('qc ', qc.sum(1).mean())
        kl = torch.mean(torch.sum(qc*kl, 1))
        # print('klm ', kl)

        return kl
    
    def kl_w(self, w_mu, w_logvar):
        loss = 0.5 * (-w_logvar + torch.exp(w_logvar) + torch.pow(w_mu, 2) - 1).sum()
        bs = w_mu.size(0)
        return loss / bs
        
    def kl_c(self, qc, c_lambda=0):
        # qc: (bs, c_dim)
        loss = torch.sum(qc * torch.log(qc * self.c_dim + 1e-10), 1)
        if c_lambda > 0:
            loss = torch.max(loss, torch.zeros_like(loss).fill_(c_lambda))
        return loss.mean()
    
    
    def ELBO(self, x, rec_lambda=1, reg_lambda=1, c_lambda=0):

        z_mu, z_logvar, w_mu, w_logvar = self.Qzw_x(x)
        
        z_sample = self.reparameterize(z_mu, z_logvar)
        x_r = self.Px_z(z_sample)
        L_rec = self.rec_term(x, x_r)

        L_w = self.kl_w(w_mu, w_logvar)
        
        w_sample = self.reparameterize(w_mu, w_logvar)
        z_mus_list, z_logvars_list = self.Pz_wc(w_sample)
        qc = self.Qc_z(z_sample, z_mus_list, z_logvars_list)
        # print(qc)
        L_c = self.kl_c(qc, c_lambda)
            
        L_z = self.kl_z(z_mu, z_logvar, z_mus_list, z_logvars_list, qc)
        
        Loss = rec_lambda * L_rec + reg_lambda * (L_z + L_w + L_c)
        
        return Loss, L_rec, L_z, L_w, L_c, x_r
    
    def forward(self, x):
        z_mu, z_logvar, _, _ = self.Qzw_x(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_r = self.Px_z(z)
        return x_r


    @staticmethod
    def gaussian_pdf_log(z, mu, logvar):
        # z: (bs, z_dim), mu: (bs, z_dim), logvar: (bs, z_dim)
        # log(N(z|mu, var))
        p = -0.5*(np.log(np.pi*2) + logvar + (z-mu).pow(2)/logvar.exp()) 
        return p



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