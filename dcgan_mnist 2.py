import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
import imageio
import natsort
from glob import glob

import argparse
import logging
import time


import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )
  
    def forward(self, input):
      output = self.network(input)
      return output

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)
    




def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))])

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, duration=fps)

    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--ndf', type=int, default=32, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=32, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=100, help='Size of the noise')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1, help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--fps', type=int, default=5, help='frames-per-second value for the gif')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')

    opt = parser.parse_args()
    print(opt)

    # Gather MNIST Dataset    
    train_loader = get_data_loader(opt.batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    # Define Discriminator and Generator architectures
    netG = Generator(opt.nc, opt.nz, opt.ngf).to(device)
    netD = Discriminator(opt.nc, opt.ndf).to(device)

    # loss function
    criterion = nn.BCELoss()

    # optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.d_lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.g_lr)
    
    # initialize other variables
    real_label = 1.
    fake_label = 0.
    num_batches = len(train_loader)
    fixed_noise = torch.randn(opt.num_test_samples, 100, 1, 1, device=device)

    for epoch in range(opt.num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            bs = real_images.shape[0]
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, device=device)

            output = netD(real_images)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(bs, opt.nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_images.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ##########################
            #   Training generator   #
            ##########################

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_images)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if (i+1)%100 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, opt.num_epochs, 
                                                            i+1, num_batches, lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
        netG.eval()
        generate_images(epoch, opt.output_path, fixed_noise, opt.num_test_samples, netG, device, use_fixed=opt.use_fixed)
        netG.train()

    # Save gif:
    save_gif(opt.output_path, opt.fps, fixed_noise=opt.use_fixed)