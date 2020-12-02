import torch
import torchvision
import src.dataloader
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

from hdbscan import HDBSCAN


@torch.jit.script
def calculate_vector(mask: torch.Tensor) -> torch.Tensor:
    """
    Have to use a fixed deltas for each pixel
    else it is size variant and doenst work for big images...

    :param mask:
    :return: [1,1,x,y,z] vector to the center of mask
    """
    x_factor = 1 / 512
    y_factor = 1 / 512
    z_factor = 1 / 40

    # com = torch.zeros(mask.shape)
    vector = torch.zeros((1, 3, mask.shape[2], mask.shape[3], mask.shape[4]))
    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * mask.shape[2], mask.shape[2]),
                                 torch.linspace(0, y_factor * mask.shape[3], mask.shape[3]),
                                 torch.linspace(0, z_factor * mask.shape[4], mask.shape[4])])

    for u in torch.unique(mask):
        if u == 0:
            continue
        index = ((mask == u).nonzero()).float().mean(dim=0)

        # Set between 0 and 1
        index[2] = index[2] / mask.shape[2]
        index[3] = index[3] / mask.shape[3]
        index[4] = index[4] / mask.shape[4]

        vector[0, 0, :, :, :][mask[0, 0, :, :, :] == u] = -xv[mask[0, 0, :, :, :] == u] + index[2]
        vector[0, 1, :, :, :][mask[0, 0, :, :, :] == u] = -yv[mask[0, 0, :, :, :] == u] + index[3]
        vector[0, 2, :, :, :][mask[0, 0, :, :, :] == u] = -zv[mask[0, 0, :, :, :] == u] + index[4]

    return vector


@torch.jit.script
def vector_to_embedding(vector: torch.Tensor) -> torch.Tensor:
    """
    Constructs a mesh grid and adds the vector matrix to it

    :param vector:
    :return:
    """
    x_factor = 1 / 512
    y_factor = 1 / 512
    z_factor = 1 / 40

    # xv, yv, zv = torch.meshgrid([torch.linspace(0, 1, vector.shape[2]),
    #                              torch.linspace(0, 1, vector.shape[3]),
    #                              torch.linspace(0, 1, vector.shape[4])])

    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * vector.shape[2], vector.shape[2]),
                                 torch.linspace(0, y_factor * vector.shape[3], vector.shape[3]),
                                 torch.linspace(0, z_factor * vector.shape[4], vector.shape[4])])

    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1).to(vector.device)

    return mesh + vector


# @torch.jit.script
def embedding_to_probability(embedding: torch.Tensor, centroids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Vectorizing this is slower than the loop!!!

    #                     /    (e_ix - C_kx)^2       (e_iy - C_ky)^2        (e_iz - C_kz)^2   \
    #  prob_k(e_i) = exp |-1 * ----------------  -  -----------------   -  ------------------  |
    #                    \     2*sigma_kx ^2         2*sigma_ky ^2          2 * sigma_kz ^2  /

    :param embedding: [B, K=3, X, Y, Z] torch.Tensor where K is the likely centroid component: {X, Y, Z}
    :param centroids: [B, I, K_true=3] torch.Tensor where I is the number of instances in the image and K_true is centroid
                        {x, y, z}
    :param sigma: torch.Tensor of shape = (1) or (embedding.shape)
    :return: [B, I, X, Y, Z] of probabilities for instance I
    """

    centroids[:, :, 0] /= 512
    centroids[:, :, 1] /= 512
    centroids[:, :, 2] /= 40

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

    prob = torch.zeros((embedding.shape[0],
                        centroids.shape[1],
                        embedding.shape[2],
                        embedding.shape[3],
                        embedding.shape[4])).to(embedding.device)


    if len(sigma) != 3:
        sigma = (2 * sigma.to(embedding.device) ** 2)
        for i in range(centroids.shape[1]):

            # Calculate euclidean distance between centroid and embedding for each pixel
            euclidean_norm = (embedding - centroids[:, i, :].reshape(centroids.shape[0], 3, 1, 1, 1)).pow(2).sum(
                dim=1).unsqueeze(1)

            print(embedding.max(), centroids.max(), euclidean_norm.sqrt().max())
            # Turn distance to probability and put it in preallocated matrix
            prob[:, i, :, :, :] = torch.exp(-1 * euclidean_norm / sigma).squeeze(1)

    else:
        sigma = (2 * sigma.to(embedding.device) ** 2).reshape(centroids.shape[0], 3, 1, 1, 1)

        for i in range(centroids.shape[1]):

            euclidean_norm = (embedding - centroids[:, i, :].reshape(centroids.shape[0], 3, 1, 1, 1)).pow(2)
            prob[:, i, :, :, :] = torch.exp((-1 * (euclidean_norm / sigma)).sum(1))

    return prob


def estimate_centroids(embedding: torch.Tensor, eps: float = 0.2, min_samples: int = 100) -> torch.Tensor:
    """
    Assume [B, 3, X, Y, Z]
    Warning moves everything to cpu!

    :param embedding:
    :return:
    """
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

    X = np.stack((x, y, z)).T
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    # db = HDBSCAN(min_samples=1000).fit(X)
    labels = db.labels_

    centroids = []
    for u in np.unique(labels):
        if u == -1:
            continue

        index = labels == u
        c = X[index, :].mean(axis=0)
        centroids.append(c)

    if len(centroids) == 0:
        centroids = torch.empty((1, 0, 3)).to(device)
    else:
        centroids = torch.tensor(centroids).to(device).unsqueeze(0)
        centroids[:, :, 0] *= 512
        centroids[:, :, 1] *= 512
        centroids[:, :, 2] *= 40

    return centroids


if __name__ == '__main__':
    print('Loading Val...')
    transforms = torchvision.transforms.Compose([
        # t.nul_crop(),
        # t.random_crop(shape=(256, 256, 23)),
    ])

    val = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test',
                                 transforms=transforms)
    val = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    print('Done')

    for dd in val:
        mask = dd['masks']
        centroids = dd['centroids']

    plt.plot(centroids[0, :, 0] * 1 / 512, centroids[0, :, 1] * 1 / 512, 'ko')
    plt.show()
