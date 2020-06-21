import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tps
from cv2.ximgproc import guidedFilter
import torchvision.transforms as transforms
import PIL

def composer(bg_img, transformed_obj):
    ''' 
    input shape = (N, C, H, W)
    transformed_obj - (N, 1, H, W)
    '''
    return bg_img * (1-transformed_obj)

# https://github.com/cheind/py-thin-plate-spline
class STN_TPN(nn.Module):
    def __init__(self, ctrlshape = (6, 6)):
        super().__init__()

        self.nctrl = ctrlshape[0]*ctrlshape[1]
        self.nparam = (self.nctrl + 2)
        ctrl = tps.uniform_grid(ctrlshape)
        self.register_buffer('ctrl', ctrl.view(-1,2))

        # Spatial transformer localization-network
        self.loc = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3),        
            nn.MaxPool2d(2),
            nn.ReLU(True),                           
            nn.Conv2d(32, 64, kernel_size=3),       
            nn.MaxPool2d(2),
            nn.ReLU(True),                           
            nn.Conv2d(64, 128, kernel_size=3),     
            nn.MaxPool2d(2),
            nn.ReLU(True)                         
        )

        # Regressor for the thin plate spline matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*30*30, 512),
            nn.ReLU(True),
            nn.Linear(512, self.nparam * 2),
            nn.Tanh()
        )

        # Initialize the weights/bias
        self.fc_loc[2].weight.data.normal_(0, 1e-3)
        self.fc_loc[2].bias.data.zero_()

    def forward(self, background, foreground):
        x = torch.cat((background, foreground), dim=1)
        bs = x.shape[0]
        x = self.loc(x)
        theta = self.fc_loc(x.view(bs, -1)).view(-1, self.nparam, 2)
        grid = tps.tps_grid(theta, self.ctrl, foreground.size())
        xt = F.grid_sample(foreground, grid)
        return xt

# Pytorch affine STN : https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class STN_AFFINE(nn.Module):
    def __init__(self):
        super().__init__()

        # Spatial transformer localization-network
        self.loc = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3),        
            nn.MaxPool2d(2),
            nn.ReLU(True),                           
            nn.Conv2d(32, 64, kernel_size=3),       
            nn.MaxPool2d(2),
            nn.ReLU(True),                           
            nn.Conv2d(64, 128, kernel_size=3),     
            nn.MaxPool2d(2),
            nn.ReLU(True)                         
        )

        # Regressor for the thin plate spline matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*30*30, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 2)
        )

        # Initialize the weights/bias
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, background, foreground):
        x = torch.cat((background, foreground), dim=1)
        bs = x.shape[0]
        x = self.loc(x)
        theta = self.fc_loc(x.view(x.shape[0], -1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, foreground.size())
        xt = F.grid_sample(foreground, grid)
        return xt

class GeometrySynthesizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn_tpn = STN_TPN()
        self.stn_affine = STN_AFFINE()

    def forward(self, background, foreground):
        affine_transformed_foreground = self.stn_affine(background, foreground)
        tpn_transformed_foreground = self.stn_tpn(background, affine_transformed_foreground)
        composed = composer(background, tpn_transformed_foreground)
        return composed

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    '''
    A Resnet Generator with Instance norm and 64 output filters
    '''
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)
        self.filter = GuidedFilter(7, 1.3)

    def forward(self, x):
        y = self.model(x)
        # return y
        return self.filter(y, x)

class Discriminator(nn.Module):
    '''
    A PatchGAN discriminator
    '''
    def __init__(self, input_nc):
        super().__init__()

        # A bunch of convolutions one after another
        model = [   
            # nn.Dropout(0.2),
                    nn.utils.spectral_norm(nn.Conv2d(input_nc, 64, 4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  
            nn.Dropout(0.1),
                    nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  
            nn.Dropout(0.1),
                    nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # model += [  
        #     # nn.Dropout(0.2),
        #             nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, padding=1)),
        #             nn.InstanceNorm2d(512), 
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, padding=1))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class GuidedFilter(nn.Module):
    '''
    Kaiming He Guided Filter
    '''
    def __init__(self, radius=4, eps=0.02):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, I, R):
        '''I is the Generated Image from G1, to be filtered
        R is the Composed Image, acts as the guidance image
        '''
        "TODO: check input shape for guided filter and change it accordingly"
        to_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        bs = I.shape[0]
        res = []
        for j in range(bs):
            i = np.asarray(to_image(I[j].cpu()))
            r = np.asarray(to_image(R[j].cpu()))
            filtered = to_tensor(PIL.Image.fromarray(guidedFilter(guide=r, src=i, radius = self.radius, eps = self.eps)))
            res.append(filtered[None])
        res = torch.cat(res, dim=0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res = res.to(device)
        return res






