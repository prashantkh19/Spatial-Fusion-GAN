import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, Discriminator, GeometrySynthesizer
from utils import *
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
    G0 = GeometrySynthesizer()
    G1 = Generator(opt.input_nc, opt.output_nc)
    G2 = Generator(opt.input_nc, opt.output_nc)
    D1 = Discriminator(opt.input_nc)
    D2 = Discriminator(opt.output_nc)

    if opt.cuda:
        G0.cuda()
        G1.cuda()
        G2.cuda()
        D1.cuda()
        D2.cuda()

    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D2.apply(weights_init_normal)
    D1.apply(weights_init_normal)

    # Optimizers & LR schedulers
    optimizer_G0 = torch.optim.Adam(G0.parameters(), lr=opt.lr_GS, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(itertools.chain(G1.parameters(), G2.parameters()), lr=opt.lr_AS, betas=(0.5, 0.999))
    optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr_AS, betas=(0.5, 0.999))
    optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr_AS, betas=(0.5, 0.999))

    if opt.G0_checkpoint is not None:
        G0 = load_G0_ckp(opt.G0_checkpoint, G0)

    if opt.AS_checkpoint is not None:
        _, G1, D1, G2, D2, optimizer_G, optimizer_D1, optimizer_D2 = load_AS_ckp(opt.AS_checkpoint, G1, D1, G2, D2, optimizer_G, optimizer_D1, optimizer_D2) 

    if opt.resume_checkpoint is not None:
        opt.epoch, G0, G1, D1, G2, D2, optimizer_G0, optimizer_G, optimizer_D1, optimizer_D2 = load_ckp(opt.resume_checkpoint, G0, G1, D1, G2, D2, optimizer_G0, optimizer_G, optimizer_D1, optimizer_D2) 

    lr_scheduler_G0 = torch.optim.lr_scheduler.LambdaLR(optimizer_G0, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    background_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    foregound_t = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    real_t = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    composed_buffer = ReplayBuffer()
    fake_real_buffer = ReplayBuffer()
    fake_composed_buffer = ReplayBuffer()

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
            composed_GS = G0(background, foreground) # concatenate background and foreground object 
            
            ## optimize G0 loss
            optimizer_G0.zero_grad()
            loss_G0 = criterion_discriminator(D2(composed_GS), target_fake)
            loss_G0.backward()
            optimizer_G0.step()
            
            ###### Appearance Synthesizer ######
            composed = composed_buffer.push_and_pop(composed_GS)
            ###### Generators G1 and G2 ######
            optimizer_G.zero_grad()

            ## Identity loss
            # G1(X) should equal X if X = real
            same_real = G1(real)
            loss_identity_1 = criterion_identity(real, same_real) * 5.0
            # G2(X) should equal X if X = composed
            same_composed = G2(composed)
            loss_identity_2 = criterion_identity(composed, same_composed) * 5.0

            loss_identity = loss_identity_1 + loss_identity_2

            ## GAN loss
            fake_real = G1(composed)
            loss_G1 = criterion_generator(D1(fake_real), target_real) 

            fake_composed = G2(real)
            loss_G2 = criterion_generator(D2(fake_composed), target_real) 

            loss_GAN = loss_G1 + loss_G2

            ## Cycle loss
            recovered_real = G1(fake_composed)
            loss_cycle_real = criterion_cycle(recovered_real, real) * 10.0

            recovered_composed = G2(fake_real)
            loss_cycle_composed = criterion_cycle(recovered_composed, composed) * 10.0

            loss_cycle = loss_cycle_composed + loss_cycle_real

            # Total loss
            loss_G = loss_identity + loss_GAN + loss_cycle

            loss_G.backward()
            optimizer_G.step()
            #####################################

            ###### Discriminator D1 ######
            # real loss
            loss_D1_real = criterion_discriminator(D1(real), target_real)

            # fake loss
            new_fake_real = fake_real_buffer.push_and_pop(fake_real)
            loss_D1_fake = criterion_discriminator(D1(new_fake_real.detach()), target_fake)

            loss_D1 = (loss_D1_real + loss_D1_fake) * 0.5
            loss_D1.backward()
            optimizer_D1.step()

            ###### Discriminator D2 ######
            # real loss
            new_composed = composed_buffer.push_and_pop(composed)
            loss_D2_real = criterion_discriminator(D2(new_composed.detach()), target_real)

            # fake loss
            new_fake_composed = fake_composed_buffer.push_and_pop(fake_composed)
            loss_D2_fake = criterion_discriminator(D2(new_fake_composed.detach()), target_fake)

            loss_D2 = (loss_D2_real + loss_D2_fake) * 0.5
            loss_D2.backward()
            optimizer_D2.step()

            #####################################        

            # Progress report (http://localhost:8097)
            losses = {
                'loss_G0': loss_G0,
                'loss_G': loss_G,
                'loss_D1': loss_D1,
                'loss_D2': loss_D2
                } 
            images = {
                'background': background,
                'foreground': foreground,
                'real': real,
                'composed_GS': composed_GS,
                'composed': composed,
                'synthesized': fake_real,
                'adapted_real': fake_composed 
                }
                
            logger.log(losses, images)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D1.step()
        lr_scheduler_D2.step()

        # Save models checkpoints
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': {
                "G0": G0.state_dict(),
                "G1": G1.state_dict(),
                "D1": D1.state_dict(),
                "G2": G2.state_dict(),
                "D2": D2.state_dict()
            },
            'optimizer': {
                "G0": optimizer_G0.state_dict(),
                "G": optimizer_G.state_dict(),
                "D1": optimizer_D1.state_dict(),
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
    parser.add_argument('--lr_GS', type=float, default=0.0002, help='initial learning rate for GS')
    parser.add_argument('--lr_AS', type=float, default=0.0002, help='initial learning rate for AS')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoints_dir', type=str, default='/checkpoints/', help='checkpoints output directory')
    parser.add_argument('--out_dir', type=str, default='/output/', help='output directory')
    parser.add_argument('--run_name', type=str, default='initial', help='experiment run name')
    parser.add_argument('--G0_checkpoint', type=str, default=None, help='G0 checkpoint path')
    parser.add_argument('--AS_checkpoint', type=str, default=None, help='AS models checkpoint path')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='resume checkpoint path')
    opt = parser.parse_args()

    opt = vars(opt)
    print(opt)
    train(opt)

