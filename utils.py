import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from livelossplot import PlotLosses

def save_ckp(state, f_path):
    torch.save(state, f_path)

def load_GS_ckp(checkpoint_fpath, G0, D2, optimizer_G0, optimizer_D2):
    checkpoint = torch.load(checkpoint_fpath)
    G0.load_state_dict(checkpoint['state_dict']['G0'])
    D2.load_state_dict(checkpoint['state_dict']['D2'])
    optimizer_G0.load_state_dict(checkpoint['optimizer']['G0'])
    optimizer_D2.load_state_dict(checkpoint['optimizer']['D2'])
    return checkpoint['epoch'], G0, D2, optimizer_G0, optimizer_D2

def load_AS_ckp(checkpoint_fpath, G1, D1, G2, D2, optimizer_G, optimizer_D1, optimizer_D2):
    checkpoint = torch.load(checkpoint_fpath)
    G1.load_state_dict(checkpoint['state_dict']['G1'])
    G2.load_state_dict(checkpoint['state_dict']['G2'])
    D1.load_state_dict(checkpoint['state_dict']['D1'])
    D2.load_state_dict(checkpoint['state_dict']['D2'])
    optimizer_G1.load_state_dict(checkpoint['optimizer']['G'])
    optimizer_D1.load_state_dict(checkpoint['optimizer']['D1'])
    optimizer_D2.load_state_dict(checkpoint['optimizer']['D2'])
    return checkpoint['epoch'], G1, D1, G2, D2, optimizer_G, optimizer_D1, optimizer_D2

def load_G0_ckp(checkpoint_fpath, G0):
    checkpoint = torch.load(checkpoint_fpath)
    G0.load_state_dict(checkpoint['state_dict']['G0'])
    return G0


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().detach().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch, out_dir, start_epoch=1):
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = start_epoch
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.out_dir = out_dir
        self.to_image = transforms.ToPILImage()
        self.liveloss = PlotLosses()

    def log(self, losses=None, images=None):
        pass
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        plots = {}

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data
            else:
                self.losses[loss_name] += losses[loss_name].data

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        if self.batch % 10 == 0: 
            # Save images
            plt.ioff()
            fig = plt.figure(figsize=(100, 50))
            for i, (image_name, tensor) in enumerate(images.items()):
                ax = plt.subplot(1, len(images), i+1)
                ax.imshow(self.to_image(tensor.cpu().data[0]))
            fig.savefig(self.out_dir + '/%d_%d.png' % (self.epoch, self.batch))
            plt.close(fig)
            # self.to_image(images["composed"].cpu().data[0]).save(self.out_dir + '/%d_%d.png' % (self.epoch, self.batch))
            # plt.close(fig)

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for i, (loss_name, loss) in enumerate(self.losses.items()):
        #         if loss_name not in self.loss_windows:
        #             self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
        #                                                             opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
        #         else:
        #             self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')

                plots[loss_name] = self.losses[loss_name]/self.batch 

                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.liveloss.update(plots)
            self.liveloss.send()

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return),requires_grad=False)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
class dotDict(dict):
    """dot . notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    