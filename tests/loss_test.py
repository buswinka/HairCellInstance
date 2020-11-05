import torch
import src.loss
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
import pickle


def test_jaccard_loss():

    x = torch.randn((1, 10, 100, 100, 10)) > 0.5
    y = torch.randn((1, 10, 100, 100, 10)) > 0.1

    assert x.max() == 1
    assert y.max() == 1

    out = src.loss.jaccard_loss()(x, y)

    assert out > 0
    assert out < 1

def test_sanity():
    loss_fun = src.loss.jaccard_loss()

    data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test')
    data = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    for data_dict in data:
        pass

    image = data_dict['image']
    mask = data_dict['mask']
    centroids = data_dict['centroids']

    out = pickle.load(open('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/tiny.vector.pkl', 'rb'))

    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert out.shape[-1] == image.shape[-1]
    assert out.max() > 0.21
    assert out.min() < -0.25
    assert centroids.max() != 0
    assert centroids.max() < 2
    assert centroids.min() > 0

    for i in [20]:
        plt.imshow(out[0,0,:,:,i])
        plt.show()

    out = src.utils.vector_to_embedding(out)



    for i in [20]:
        plt.imshow(out[0,0,:,:,i])
        plt.show()


    out = src.utils.embedding_to_probability(out, centroids, torch.tensor([0.05]))

    assert out.shape[1] == centroids.shape[1]

    for i in range(out.shape[-1]):
        plt.imshow(out[0,12,:,:,i])
        plt.title(i)
        plt.show()

    assert out[0,12,:,:,i].max() < 0.1

    # This is jank
    assert out.shape == mask.shape

    loss = loss_fun(out, mask)

    assert loss < 1
