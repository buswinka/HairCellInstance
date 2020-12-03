import src.dataloader
import src.loss
import src.functional
# import torch
# from src.models.RDCNet import RDCNet
from src.models.HCNet import HCNet
# from src.models.unet import Unet_Constructor as unet
# from src.models.RecurrentUnet import RecurrentUnet
# import torch
# import torch.optim
import torch.nn
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast
import time
import numpy as np
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter

import src.transforms as t
import skimage.io as io

epochs = 50

model = torch.jit.script(HCNet(in_channels=3, out_channels=4, complexity=30)).cuda()
model.train()
#model.load_state_dict(torch.load('./trained_model_hcnet_long.mdl'))

writer = SummaryWriter()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fun = src.loss.jaccard_loss()

print('Loading Train...')
transforms = torchvision.transforms.Compose([
    t.nul_crop(),
    t.random_crop(shape=(256, 256, 23)),
    t.to_cuda(),
    # t.random_h_flip(),
    # t.random_v_flip(),
    # t.random_affine(),
    # t.adjust_brightness(),
    t.adjust_centroids(),
])
data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test', transforms=transforms)
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
print('Done')

print('Loading Val...')
transforms = torchvision.transforms.Compose([
    # t.nul_crop(),
    # t.random_crop(shape=(256, 256, 23)),
])

val = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/validate', transforms=transforms)
val = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
print('Done')

for e in range(epochs):
    time_1 = time.clock_gettime_ns(1)
    epoch_loss = []
    model.train()
    for data_dict in data:
        image = data_dict['image']
        image = (image - 0.5) / 0.5
        mask = data_dict['masks'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()

        out = model(image.cuda(), 5)
        sigma = torch.sigmoid(out[:, -1, ...])
        out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])
        out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)

        # This is jank
        loss = loss_fun(out, mask.cuda())

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.detach().cpu().item())
        break

    del out, sigma, image, mask, centroids, loss

    writer.add_scalar('Loss/train', torch.mean(torch.tensor(epoch_loss)).item(), e)

    time_2 = time.clock_gettime_ns(1)
    delta_time = np.round((np.abs(time_2 - time_1) / 1e9) / 60, decimals=2)

    with torch.no_grad():
        val_loss = []
        model.eval()
        for data_dict in val:
            image = data_dict['image']
            image = (image - 0.5) / 0.5
            mask = data_dict['masks'] > 0.5
            centroids = data_dict['centroids']

            out = model(image.cuda(), 5)
            sigma = out[:, -1, ...]
            out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])
            out = src.functional.embedding_to_probability(out, centroids.cuda(), sigma)
            loss = loss_fun(out, mask.cuda())

            val_loss.append(loss.item())
        val_loss = torch.tensor(val_loss).mean()
    writer.add_scalar('Loss/validate', val_loss.item(), e)
    del out, loss, image, mask, val_loss, sigma

    if e % 1 == 0:
        progress_bar = '[' + '█' * +int(np.round(e / epochs, decimals=1) * 10) + \
                       ' ' * int(
            (10 - np.round(e / epochs, decimals=1) * 10)) + f'] {np.round(e / epochs, decimals=3)}%'

        out_str = f'epoch: {e} ' + progress_bar + f'| time remaining: {np.round(delta_time * (epochs - e), decimals=3)}' \
                                                  f' min | loss: {torch.mean(torch.tensor(epoch_loss)).item()}'
        if e > 0:
            print('\b \b' * len(out_str), end='')
        print(out_str, end='')

    # If its the final epoch print out final string
    elif e == epochs - 1:
        print('\b \b' * len(out_str), end='')
        progress_bar = '[' + '█' * 10 + f'] {1.0}'
        out_str = f'epoch: {epochs} ' + progress_bar + f'| time remaining: {0} min | loss: {loss.item()}'
        print(out_str)

torch.save(model.state_dict(), 'overtrained_model_hcnet_long.mdl')


render = (out > 0.25).int().squeeze(0)
for i in range(render.shape[0]):
    render[i,:,:,:] = render[i,:,:,:] * (i+1)
io.imsave('test.tif', render.sum(0).detach().cpu().int().numpy().astype(np.int).transpose((2,1,0)) / i+1)


