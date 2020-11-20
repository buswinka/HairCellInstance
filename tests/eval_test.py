import src.dataloader
import src.loss
import src.functional
import torch
from src.models.RDCNet import RDCNet
from src.models.unet import Unet_Constructor as unet
from src.models.RecurrentUnet import RecurrentUnet
import torch
import torch.optim
import torch.nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import skimage.io as io

epochs = 100

model = RDCNet(in_channels=4, out_channels=3, complexity=20).cuda()
model.eval()
model.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/src/trained_model.mdl'))

data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test')
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
for data_dict in data:
    image = data_dict['image']
    image = (image - 0.5) / 0.5
    mask = data_dict['masks'] > 0.5
    centroids = data_dict['centroids']

with torch.no_grad():
    out, embedding, centroids = model(image.cuda())

print(embedding.shape, out.shape, centroids.shape)

device = embedding.device
embedding = embedding.detach().cpu().squeeze(0).reshape((3, -1))

x = embedding[0, :]
y = embedding[1, :]
z = embedding[2, :]

ind_x = torch.logical_and(x > 0, x < 1)
ind_y = torch.logical_and(y > 0, y < 1)
ind_z = torch.logical_and(z > 0, z < 1)
ind = torch.logical_and(ind_x, ind_y)
ind = torch.logical_and(ind, ind_z)

x = x[ind]
y = y[ind]
z = z[ind]

ind = torch.randperm(len(x))
n_samples = 100000
ind = ind[0:n_samples:1]

x = x[ind].numpy()
y = y[ind].numpy()
z = z[ind].numpy()

plt.figure()
plt.plot(x, y, 'k.', alpha=0.01)
plt.show()


