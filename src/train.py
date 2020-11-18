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
# model = unet(image_dimensions=3,
#                  in_channels=4,
#                  out_channels=4,
#                  feature_sizes=[16, 32, 64],
#                  kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
#                  upsample_kernel=(8, 8, 2),
#                  max_pool_kernel=(2, 2, 1),
#                  upsample_stride=(2, 2, 1),
#                  dilation=1,
#                  groups=2).to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fun = src.loss.jaccard_loss()

data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test')
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

for e in range(epochs):
    time_1 = time.clock_gettime_ns(1)
    for data_dict in data:
        image = data_dict['image']
        image = (image - 0.5) / 0.5
        mask = data_dict['masks'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()

        out = model(image.cuda())

        # for recording later
        vector = out.detach().cpu().clone()

        out = src.functional.vector_to_embedding(out)

        embed = out.detach().cpu().clone()
        out = src.functional.embedding_to_probability(out, centroids.cuda(), torch.tensor([0.015]))

        # This is jank
        loss = loss_fun(out[:, :, :, :, :-1:], mask.cuda())

        loss.backward()
        optimizer.step()

    time_2 = time.clock_gettime_ns(1)
    delta_time = np.round((np.abs(time_2 - time_1) / 1e9) / 60, decimals=2)
    if e % 5 == 0:
        progress_bar = '[' + '█' * +int(np.round(e / epochs, decimals=1) * 10) + \
                       ' ' * int(
            (10 - np.round(e / epochs, decimals=1) * 10)) + f'] {np.round(e / epochs, decimals=3)}%'

        out_str = f'epoch: {e} ' + progress_bar + f'| time remaining: {np.round(delta_time * (epochs - e), decimals=3)} min | loss: {loss.item()}'
        if e > 0:
            print('\b \b' * len(out_str), end='')
        print(out_str, end='')

    # If its the final epoch print out final string
    elif e == epochs - 1:
        print('\b \b' * len(out_str), end='')
        progress_bar = '[' + '█' * 10 + f'] {1.0}'
        out_str = f'epoch: {epochs} ' + progress_bar + f'| time remaining: {0} min | loss: {loss.item()}'
        print(out_str)

torch.save(model.state_dict(), 'trained_model.mdl')

for i in [22]:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(out.detach().cpu().numpy()[0, 0, :, :, i] > 0.5)
    # ax[1].imshow(embed.detach().cpu().numpy()[0, 0, :, :, i])
    # ax[2].imshow(vector.detach().cpu().numpy()[0, 0, :, :, i])
    ax[1].imshow(mask.detach().cpu().numpy()[0, 0, :, :, i])
    plt.show()

slice = embed[0, :, :, :, 22].reshape(3, -1)
x = slice[0, :]
y = slice[1, :]
z = slice[2, :]

ind_x = torch.logical_and(x > 0, x < 1)
ind_y = torch.logical_and(y > 0, y < 1)
ind_z = torch.logical_and(z > 0, z < 1)
ind = torch.logical_and(ind_x, ind_y)
ind = torch.logical_and(ind, ind_z)

x = x[ind]
y = y[ind]
z = z[ind]

plt.plot(y.detach(), x.detach(), '.', alpha=1, markersize=0.05)
plt.show()


render = (out > 0.5).int().squeeze(0)
for i in range(render.shape[0]):
    render[i,:,:,:] = render[i,:,:,:] * (i+1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2,1,0)) / i+1)
