import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset, Subset

import numpy as nps
import matplotlib.pyplot as plt




FFQH_PATH = # set to your filepath
JWL_PATH = # set to your filepath
WEIGHTS_PATH = # set to your filepath
if not os.path.isdir(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_PATH)

workers = 2
batch_size = 64
image_size = 128
num_channels = 3
z_length = 100
dgf = image_size # size of feature maps in Generator
ddf = image_size # size of feature maps in Discriminator
num_epochs = 5 # for now
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
neg_slope = 0.2
iter_interval = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device = {device}')


# setting up a custom image folder to make it easier to assign labels
# labels are used for a Conditional GAN
class CustomImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform, label):
        super().__init__(root, transform)
        self.new_label = label
    
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, self.new_label

resize = v2.Compose([v2.Resize(image_size, image_size),
                     v2.ToTensor()])

ffqh_dataset = CustomImageFolder(root = FFQH_PATH, transform = resize, label  = 0)
# if label = 1, GAN will generate images more akin to those in this folder
jwl_dataset = CustomImageFolder(root = JWL_PATH, transform = resize, label = 1)

total_ffqh_photos = len(ffqh_dataset)
subsample = int(0.1 * total_ffqh_photos)
indices = torch.randperm(total_ffqh_photos).tolist()
sub_indices = indices[:subsample]
smaller_ffqh = Subset(ffqh_dataset, sub_indices)

dataset = ConcatDataset([smaller_ffqh, jwl_dataset])


# model weights randomly initialized from normal distribution of N(0, 0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator definition
class Generator(nn.Module):

    def __init__(self):
        super.__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_channels = z_length + 1,
                                                    out_channels = dgf * 16,
                                                    kernel_size = 4, 
                                                    stride = 1,
                                                    padding = 0, 
                                                    bias = False),
                                nn.BatchNorm2d(num_features = dgf * 16),
                                nn.ReLU(inplace = True),

                                nn.ConvTranspose2d(dgf * 16, dgf * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
                                nn.BatchNorm2d(num_features = dgf*8),
                                nn.ReLU(inplace = True),
                                
                                nn.ConvTranspose2d(dgf * 8, dgf * 4, 4, 2, 1, bias = False),
                                nn.BatchNorm2d(dgf * 4),
                                nn.ReLU(inplace = True),

                                nn.ConvTranspose2d(dgf*4, dgf*2, 4, 2, 1, bias = False),
                                nn.BatchNorm2d(dgf*2),
                                nn.ReLU(inplace = True),

                                nn.ConvTranspose2d(dgf*2, dgf, 4, 2, 1, bias = False),
                                nn.BatchNorm2d(dgf),
                                nn.ReLU(inplace = True),

                                nn.ConvTranspose2d(dgf, num_channels, 4, 2, 1, bias = False),
                                nn.Tanh()
                                )
    
    def forward(self, input, label):
        label = label.view(-1, 1, 1, 1)
        input = torch.cat((input, label), 1)
        return self.net(input)

# Discriminator definition
class Discriminator(nn.Module):

    def __init__(self):
        self.net = nn.Sequential(
            nn.Conv2d(num_channels + 1, ddf, 4, 2, 1, bias = False),
            nn.LeakyReLU(negative_slope = neg_slope, inplace = True),

            nn.Conv2d(ddf, ddf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ddf * 2),
            nn.LeakyReLU(neg_slope, inplace = True),

            nn.Conv2d(ddf*2, ddf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ddf*4),
            nn.LeakyReLU(neg_slope, inplace = True),

            nn.Conv2d(ddf*4, ddf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ddf*8),
            nn.LeakyReLU(neg_slope, inplace = True),

            nn.Conv2d(ddf*8, ddf*16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ddf*16),
            nn.LeakyReLU(neg_slope, True),

            nn.Conv2d(ddf*16, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()

        )

    def forward(self, input, label):
        label = label.view(-1, 1, 1, 1)
        label = label.expand(input.size(0), 1, input.size(2), input.size(3))
        input = torch.cat((input, label), 1)
        return self.net(input)


g = Generator()
d = Discriminator()

criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size, z_length, 1, 1, device = device)
fixed_conditions = torch.randn(batch_size, 1, 1, device = device)
fixed_conditions = fixed_conditions.masked_fill(fixed_conditions <= 0, 0)
fixed_conditions = fixed_conditions.masked_fill(fixed_conditions > 0, 1)

real_label = 1.0
fake_label = 0.0

Goptimizer = torch.optim.Adam(g.parameters(), lr = lr, betas = (beta1, beta2))
Doptimizer = torch.optim.Adam(d.optimizers(), lr = lr, betas = (beta1, beta2))

dataloader = DataLoader(dataset, batch_size, shuffle = True, num_workers = workers, pin_memory = True)

all_images = []
all_labels = []

# loading into memory upfront so training is faster
for image, label in dataloader:
    all_images.append(image)
    all_labels.append(label)



D_losses = []
G_losses = []
iters = 0

for epoch in range(num_epochs):
    for i, image in enumerate(all_images):
        d.zero_grad()
        realData = image.to(device)
        realConditions = all_labels[i].to(device)
        batch_size = realData.size(0)
        labels = torch.full((batch_size,), real_label, dtype = torch.float, device = device)
        d_output_real = d(realData, realConditions).view(-1)
        loss_d_real = criterion(d_output_real, labels)
        loss_d_real.backward()
        d_x = d_output_real.mean().item()

        noise = torch.randn(batch_size, z_length, 1, 1, device = device)
        fakeData = g(noise, realConditions)
        labels.fill_(fake_label)
        d_output_fake = d(fakeData.detach(), realConditions).view(-1)
        loss_d_fake = criterion(d_output_fake, labels)
        loss_d_fake.backward()
        d_gz1 = d_output_fake.mean().item()
        d_error = d_gz1 + d_x
        Doptimizer.step()

        g.zero_grad()
        labels.fill_(real_label)
        d_g_output = d(fakeData, realConditions).view(-1)
        loss_g = criterion(d_g_output, labels)
        d_gz2 = d_g_output.mean().item()
        Goptimizer.step()

        if i % 50 == 0:
            print(f'Epoch {epoch} / {num_epochs}, data {i} / {len(dataloader)}:\t Loss D: {d_error:.2f}, Loss G: {loss_g.item():.2f}, D(x): {d_x:.2f}, D(G(z)): {d_gz1:.2f} / {d_gz2 :.2f}')

        G_losses.append(loss_g.item())
        D_losses.append(d_error)


        if (iters % iter_interval == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = g(fixed_noise, fixed_conditions).detach().cpu()
            torch.save(d, WEIGHTS_PATH + f'/D_{epoch}')
            torch.save(g, WEIGHTS_PATH + f'/G_{epoch}')
        
        iters += 1
    
plt.figure(figsize = (10, 5))
plt.title('Generator & Discriminator Loss During Training')
plt.plot(G_losses, label = "G")
plt.plot(D_losses, label = "D")
plt.xlabel("Iterations")
plt.ylabel("Losses")
plt.legend()
plt.show()