import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, Discriminator, GeometrySynthesizer
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal, set_requires_grad, tensor2image, dotDict, save_ckp, load_ckp
from datasets import MyDataset, TextUtils
from loss import *
import matplotlib.pyplot as plt
import os

def train(opt):
    opt = dotDict(opt)

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    if not os.path.exists(os.path.join(opt.out_dir, opt.run_name)):
        os.makedirs(os.path.join(opt.out_dir, opt.run_name))

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    G0 = GeometrySynthesizer(type = opt.stn_type)
    D2 = Discriminator(opt.output_nc)

    if opt.cuda:
        G0.cuda()
        D2.cuda()

    D2.apply(weights_init_normal)

    # Optimizers & LR schedulers
    optimizer_G0 = torch.optim.Adam(G0.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    if opt.resume_checkpoint is not None:
        opt.epoch, G0, D2, optimizer_G0, optimizer_D2 = load_ckp(opt.resume_checkpoint, G0, D2, optimizer_G0, optimizer_D2) 

    lr_scheduler_G0 = torch.optim.lr_scheduler.LambdaLR(optimizer_G0, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    background_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    foregound_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    real_t = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    composed_buffer = ReplayBuffer()

    # Dataset loader
    transforms_dataset = [ 
                    transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(opt.size), 
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ]

    transforms_masks = [transforms.ToTensor()]

    text = TextUtils(opt.root, transforms_=transforms_masks)

    dataset = MyDataset(opt.root, transforms_=transforms_dataset)
    print("No. of Examples = ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader), os.path.join(opt.out_dir, opt.run_name), opt.epoch+1)
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            background = Variable(background_t.copy_(batch['X']), requires_grad=True)
            # foreground = Variable(foregound_t.copy_(batch['Y']), requires_grad=True)
            real = Variable(real_t.copy_(batch['Z']), requires_grad=True)
            foreground = Variable(foregound_t.copy_(text.get_text_masks(opt.batchSize)), requires_grad=True)

            ###### Geometric Synthesizer ######
            composed = G0(background, foreground) # concatenate background and foreground object 
            
            ## optimize G0 loss
            optimizer_G0.zero_grad()

            loss_G0 = criterion_discriminator(D2(composed), target_real)
            
            loss_G0.backward()
            optimizer_G0.step()
            
            ## optimize D2 Geometry loss
            optimizer_D2.zero_grad()

            # real loss
            loss_D2_real = criterion_discriminator(D2(real), target_real)
            # composed loss
            new_composed = composed_buffer.push_and_pop(composed)
            loss_D2_composed  = criterion_discriminator(D2(new_composed), target_fake)
            
            loss_D2 = (loss_D2_real + loss_D2_composed) * 0.5

            if i % 5 == 0:
                loss_D2.backward()
                optimizer_D2.step()

            # Progress report (http://localhost:8097)
            losses = {
                'loss_G0': loss_G0,
                'loss_D2': loss_D2
                } 
            images = {
                'background': background,
                'foreground': foreground,
                'real': real,
                'composed': composed
                }
                
            logger.log(losses, images)

        # Update learning rates
        lr_scheduler_G0.step()
        lr_scheduler_D2.step()

        # Save models checkpoints
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': {
                "G0": G0.state_dict(),
                "D2": D2.state_dict()
            },
            'optimizer': {
                "G0": optimizer_G0.state_dict(),
                "D2": optimizer_D2.state_dict()
            }
        }
        save_ckp(checkpoint, os.path.join(opt.checkpoints_dir, opt.run_name + '.pth'))
    ###################################

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
    parser.add_argument('--root', type=str, default='/', help='root directory')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoints_dir', type=str, default='/checkpoints/', help='checkpoints output directory')
    parser.add_argument('--out_dir', type=str, default='/output/', help='output directory')
    parser.add_argument('--run_name', type=str, default='initial', help='experiment run name')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='resume checkpoint path')
    opt = parser.parse_args()

    opt = vars(opt)
    print(opt)
    train(opt)

