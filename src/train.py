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
import torch.nn.functional as F

epochs = 500

model = RDCNet(in_channels=4, out_channels=4, complexity=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
        prob_map = out[:,0,:,:,:].unsqueeze(1)
        out = out[:,1::,:,:,:]


        vector = out.cpu().clone()
        out = src.utils.vector_to_embedding(out)
        embed = out.cpu().clone()
        out = src.utils.embedding_to_probability_vector(out, centroids.cuda(), torch.tensor([0.05]))


        #This is jank
        loss_vec = loss_fun(out[:,:,:,:,:-1:], mask.cuda())
        mask,_ = mask.max(dim=1)
        mask = mask.unsqueeze(1)

        loss_mask = loss_fun(F.sigmoid(prob_map[:,:,:,:,:-1:]), mask.cuda())
        loss = loss_mask+loss_vec

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
    ax[0].imshow(out.detach().cpu().numpy()[0, 0, :, :, i])
    # ax[1].imshow(embed.detach().cpu().numpy()[0, 0, :, :, i])
    # ax[2].imshow(vector.detach().cpu().numpy()[0, 0, :, :, i])
    ax[1].imshow(mask.detach().cpu().numpy()[0, 0, :, :, i])
    plt.show()

slice = embed[0,:,:,:,22].reshape(3,-1)
x = slice[0,:]
y = slice[1,:]
z = slice[2,:]

ind_x = torch.logical_and(x > 0, x < 1)
ind_y = torch.logical_and(y > 0, y < 1)
ind_z = torch.logical_and(z > 0, z < 1)
ind = torch.logical_and(ind_x, ind_y)
ind = torch.logical_and(ind, ind_z)

x = x[ind]
y = y[ind]
z = z[ind]

plt.plot(y.detach(), x.detach(),'.', alpha=0.5)
plt.show()

for i in range(29):
    plt.imshow(prob_map[0, 0, :, :, i].detach().cpu().numpy())
    plt.show()