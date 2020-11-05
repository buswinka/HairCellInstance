import src.dataloader
import src.loss
import src.utils
import torch
from src.models.RDCNet import RDCNet
from src.models.RecurrentUnet import RecurrentUnet
import torch
import torch.optim
import torch.nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import time
import numpy as np
import matplotlib.pyplot as plt

epochs = 50

model = RDCNet(in_channels=4, out_channels=3, complexity=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = src.loss.jaccard_loss()

data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test')
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)


for e in range(epochs):
    time_1 = time.clock_gettime_ns(1)
    for data_dict in data:
        image = data_dict['image']
        mask = data_dict['mask'] > 0.5
        centroids = data_dict['centroids']

        optimizer.zero_grad()

        out = model(image.cuda())
        vector = out.cpu().clone()
        out = src.utils.vector_to_embedding(out)
        embed = out.cpu().clone()
        out = src.utils.embedding_to_probability(out, centroids.cuda(), torch.tensor([0.05]))


        #This is jank
        loss = loss_fun(out[:,:,:,:,:-1:], mask.cuda())

        loss.backward()
        optimizer.step()

    time_2 = time.clock_gettime_ns(1)
    delta_time = np.round((np.abs(time_2 - time_1) / 1e9) / 60, decimals=2)
    if e % 1 == 0:
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

for i in range(out.shape[-1]):
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(out.detach().cpu().numpy()[0, 0, :, :, i])
    ax[1].imshow(embed.detach().cpu().numpy()[0, 0, :, :, i])
    ax[2].imshow(vector.detach().cpu().numpy()[0, 0, :, :, i])
    ax[3].imshow(mask.detach().cpu().numpy()[0, 0, :, :, i])
    plt.show()
