import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, Discriminator, GeometrySynthesizer
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal, set_requires_grad, tensor2image, dotDict
from datasets import MyDataset, TextUtils
from loss import *
import matplotlib.pyplot as plt
import os

global G0, G1, G2, D1, D2

def is_training(isG0=False, isG1=False, isG2=False, isD1=False, isD2=False):
    global G0, G1, G2, D1, D2
    set_requires_grad(G0, isG0)
    # set_requires_grad(G1, isG0)
    # set_requires_grad(G2, isG0)
    # set_requires_grad(D1, isG0)
    set_requires_grad(D2, isG0)

def train(opt):
    opt = dotDict(opt)

    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.run_name)):
        os.makedirs(os.path.join(opt.checkpoints_dir, opt.run_name))

    if not os.path.exists(os.path.join(opt.out_dir, opt.run_name)):
        os.makedirs(os.path.join(opt.out_dir, opt.run_name))

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    global G0, G1, G2, D1, D2
    G0 = GeometrySynthesizer()
    # G1 = Generator(opt.input_nc, opt.output_nc)
    # G2 = Generator(opt.input_nc, opt.output_nc)
    # D1 = Discriminator(opt.input_nc)
    D2 = Discriminator(opt.output_nc)

    if opt.cuda:
        G0.cuda()
        # G1.cuda()
        # G2.cuda()
        # D1.cuda()
        D2.cuda()

    # G1.apply(weights_init_normal)
    # G2.apply(weights_init_normal)
    D2.apply(weights_init_normal)
    # D1.apply(weights_init_normal)

    # Optimizers & LR schedulers
    optimizer_G0 = torch.optim.Adam(G0.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # optimizer_G1 = torch.optim.Adam(G1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # optimizer_G2 = torch.optim.Adam(G2.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G0 = torch.optim.lr_scheduler.LambdaLR(optimizer_G0, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_G1 = torch.optim.lr_scheduler.LambdaLR(optimizer_G1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_G2 = torch.optim.lr_scheduler.LambdaLR(optimizer_G2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    background_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    foregound_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    real_t = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    # Dataset loader
    transforms_dataset = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
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
    logger = Logger(opt.n_epochs, len(dataloader), os.path.join(opt.out_dir, opt.run_name))
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            background = Variable(background_t.copy_(batch['X']), requires_grad=True)
            foreground = Variable(foregound_t.copy_(text.get_text_masks(opt.batchSize)), requires_grad=True)
            real = Variable(real_t.copy_(batch['Y']), requires_grad=True)

            ###### Geometric Synthesizer ######
            composed = G0(background, foreground) # concatenate background and foreground object 
            # adapted_real = G2(real)

            # optimize D2 Geometry loss
            is_training(isD2 = True)
            optimizer_D2.zero_grad()

            # Real_Loss
            loss_D2_real = criterion_discriminator(D2(real), target_real)
            loss_D2_composed  = 
            loss_D2.backward(retain_graph=True)
            optimizer_D2.step()

            # optimize G0 loss
            is_training(isG0 = True)
            optimizer_G0.zero_grad()
            loss_G0 = criterion_generator(D2(composed))
            loss_G0.backward(retain_graph=True)
            optimizer_G0.step()

            # ###### Appearance Synthesizer ######
            # composed = G0(background, foreground)

            # ### G1: Y -> Z ###
            # synthesized = G1(composed)

            # # optimize D1 loss
            # is_training(isD1 = True)
            # optimizer_D1.zero_grad()
            # loss_D1 = criterion_discriminator(D1(synthesized), D1(real)) # DOUBT: D2(real) as given in the paper 
            # loss_D1.backward(retain_graph=True)
            # optimizer_D1.step()

            # # optimize G1 loss
            # is_training(isG1 = True)
            # optimizer_G1.zero_grad()
            # loss_G1 = criterion_generator(D1(synthesized))
            # loss_G1_cyc = criterion_cycle(G2(synthesized), composed)
            # loss_G1_id = criterion_identity(synthesized, composed)
            # total_loss_G1 = loss_G1 + loss_G1_cyc + loss_G1_id
            # total_loss_G1.backward(retain_graph=True)
            # optimizer_G1.step()

            # ### G2: Z -> Y ###
            # adapted_real = G2(real)

            # # optimize D2 loss
            # optimizer_D2.zero_grad()
            # is_training(isD2 = True)
            # loss_D2 = criterion_discriminator(D2(adapted_real), D2(composed)) # DOUBT: D1(adapted_real) as given in the paper 
            # loss_D2.backward(retain_graph=True)
            # optimizer_D2.step()

            # # optimize G2 loss
            # optimizer_G2.zero_grad()
            # is_training(isG2 = True)
            # loss_G2 = criterion_generator(D2(adapted_real))
            # loss_G2_cyc = criterion_cycle(G1(adapted_real), real)
            # loss_G2_id = criterion_identity(adapted_real, real)
            # total_loss_G2 = loss_G2 + loss_G2_cyc + loss_G2_id
            # total_loss_G2.backward(retain_graph=True)
            # optimizer_G2.step()

            # Progress report (http://localhost:8097)
            losses = {
                'loss_G0': loss_G0,
                # 'total_loss_G1': total_loss_G1,
                # 'total_loss_G2': total_loss_G2,
                # 'loss_D1': loss_D1,
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
        # lr_scheduler_G1.step()
        # lr_scheduler_G2.step()
        # lr_scheduler_D1.step()
        lr_scheduler_D2.step()

        # Save models checkpoints
        torch.save(G0.state_dict(), os.path.join(opt.checkpoints_dir, opt.run_name, 'G0.pth'))
        # torch.save(G1.state_dict(), os.path.join(opt.checkpoints_dir, opt.run_name, 'G1.pth'))
        # torch.save(G2.state_dict(), os.path.join(opt.checkpoints_dir, opt.run_name, 'G2.pth'))
        # torch.save(D1.state_dict(), os.path.join(opt.checkpoints_dir, opt.run_name, 'D1.pth'))
        torch.save(D2.state_dict(), os.path.join(opt.checkpoints_dir, opt.run_name, 'D2.pth'))
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
    opt = parser.parse_args()

    opt = vars(opt)
    print(opt)
    train(opt)

