# -*- coding: utf-8 -*-
"""WGAN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_30u-jqvJytn6fGhyUfyeduy4A9h_TKj
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

train_size = 1.0
lr = 1e-4
betas = (.9, .99)
batch_size = 256
epochs = 200
plot_every = 10
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mnist_data(train_part, path, transform):
    dataset = MNIST(path, download=True, transform=transform)
    train_part = int(train_part * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_part, len(dataset)-train_part])
    return train_dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32)
])
train_data = mnist_data(train_size, '.', transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


def plotn(n, generator, device):
    generator.eval()
    noise = torch.FloatTensor(np.random.normal(0.0, 1.0, (n, 100))).to(device)
    imgs = generator(noise).detach().cpu()
    fig, ax = plt.subplots(1, n)
    for i, img in enumerate(imgs):
        ax[i].imshow(img[0])
    plt.show()

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super().__init__()
        """
        dim=16
        latent_dim=100
        """
        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.features = img_size // 16
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.features * self.features),
            nn.ReLU()
        )
        self.features_to_image = nn.Sequential(
            # H, W变为原来的2倍
            nn.ConvTranspose2d(in_channels=8 * dim,
                               out_channels=4 * dim,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.latent_to_features(input)
        x = x.view(-1, 8 * self.dim, 2, 2)
        return self.features_to_image(x)


class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        super().__init__()
        self.img_size = img_size
        self.img_to_features = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )
        output_size = 8 * dim * 2 * 2
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
      batch_size = input.shape[0]
      x = self.img_to_features(input)
      x = x.view(batch_size, -1)
      return self.features_to_prob(x)

def _gradient_penalty(real_data, generated_data, discriminator, device=None):
    batch_size = real_data.shape[0]

    # calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)

    if device is not None:
        alpha = alpha.to(device)

    # generated_data -> batch_size, 1, 28, 28
    interpolated = alpha * real_data + (1-alpha) * generated_data
    interpolated = interpolated.requires_grad_(True)

    if device is not None:
        interpolated = interpolated.to(device)

    pro_interpolated = discriminator(interpolated)

    gradients = torch.autograd.grad(outputs=pro_interpolated, inputs=interpolated, grad_outputs=torch.ones(pro_interpolated.size()).to(device),
                                   create_graph=True,
                                   retain_graph=True)

    # gradients -> batch_size, 1, 28, 28
    gradients = gradients[0].view(batch_size, -1)
    L2_gradients = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    return 10 * ((L2_gradients - 1) ** 2).mean()

def train(epochs, data_loaders, models, optimizers, plot_every, device):
    tqdm_iter = tqdm(range(epochs))

    g, d = models[0], models[1]
    optim_g, optim_d = optimizers[0], optimizers[1]


    for epoch in tqdm_iter:
        train_g_loss = 0
        train_d_loss = 0
        num_steps = 0
        for batch in data_loaders:

            num_steps += 1
            img, _ = batch
            img = img.to(device)
            d.train()
            optim_d.zero_grad()

            noise_data = torch.FloatTensor(np.random.normal(0.0, 1.0, (img.shape[0], 100))).to(device)

            real_data = d(img)
            generated_data = g(noise_data)
            d_fake = d(generated_data)

            gradient_penalty = _gradient_penalty(img, generated_data, d, device=device)

            d_loss = d_fake.mean() - real_data.mean() + gradient_penalty
            d_loss.backward()
            optim_d.step()

            if num_steps % 5 == 0:
                g.train()
                optim_g.zero_grad()

                noise_data = torch.FloatTensor(np.random.normal(0.0, 1.0, (img.shape[0], 100))).to(device)

                generated_data = g(noise_data)

                g_loss = -d(generated_data).mean()
                g_loss.backward()
                optim_g.step()

                train_g_loss += g_loss.item()


            train_d_loss += d_loss.item()

        if epoch % plot_every == 0 or epoch - 1 == 0:
            plotn(5, g, device=device)

        train_g_loss = train_g_loss / len(data_loaders)
        train_d_loss = train_d_loss / len(data_loaders)

        tqdm_dict = {'generator loss: ': train_g_loss, 'discriminator loss: ': train_d_loss}
        tqdm_iter.set_postfix(tqdm_dict, refresh=True)
        tqdm_iter.refresh()


if __name__ == '__main__':
    g = Generator(32, 100, 16).to(device)
    d = Discriminator(32, 16).to(device)
    optim_g = torch.optim.Adam(g.parameters(), lr=lr, betas=betas)
    optim_d = torch.optim.Adam(d.parameters(), lr=lr, betas=betas)
    models = (g, d)
    optimizers = (optim_g, optim_d)

    train(epochs, train_loader, models, optimizers, plot_every, device)