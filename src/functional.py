import torch
import torchvision
import src.dataloader
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

import torch
import torchvision.ops

import numpy as np
from numba import njit
from numba import prange

import pandas as pd

import skimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage.transform
import skimage.feature

import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev

import pickle
import glob

import GPy

import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

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
    z_factor = 1 / 512

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
    z_factor = 1 / 512

    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * vector.shape[2], vector.shape[2]),
                                 torch.linspace(0, y_factor * vector.shape[3], vector.shape[3]),
                                 torch.linspace(0, z_factor * vector.shape[4], vector.shape[4])])

    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1).to(vector.device)

    return mesh + vector


@torch.jit.script
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

    sigma = sigma + 1e-10  # when sigma goes to zero, shit hits the fan

    centroids /= 512

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

    prob = torch.zeros((embedding.shape[0],
                        centroids.shape[1],
                        embedding.shape[2],
                        embedding.shape[3],
                        embedding.shape[4])).to(embedding.device)

    sigma = (2 * sigma.to(embedding.device) ** 2)
    for i in range(centroids.shape[1]):
        # Calculate euclidean distance between centroid and embedding for each pixel
        euclidean_norm = (embedding - centroids[:, i, :].reshape(centroids.shape[0], 3, 1, 1, 1)).pow(2)

        # Turn distance to probability and put it in preallocated matrix
        prob[:, i, :, :, :] = torch.exp((euclidean_norm / sigma).mul(-1).sum(dim=1)).squeeze(1)

    # else:
    #     sigma = (2 * sigma.to(embedding.device) ** 2).reshape(centroids.shape[0], 3, 1, 1, 1)
    #
    #     for i in range(centroids.shape[1]):
    #
    #         euclidean_norm = (embedding - centroids[:, i, :].reshape(centroids.shape[0], 3, 1, 1, 1)).pow(2)
    #         euclidean_norm = euclidean_norm/sigma
    #         prob[:, i, :, :, :] = torch.exp((-1 * (euclidean_norm / sigma)).sum(dim=1))

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
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
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
        centroids[:, :, 2] *= 512

    return centroids


def get_cochlear_length(image: torch.Tensor, equal_spaced_distance: int, diagnostics=False) -> torch.Tensor:
    """
    Input an image ->
    max project ->
    reduce image size ->
    run b-spline fit of resulting data on myosin channel ->
    return array of shape [2, X] where [0,:] is x and [1,:] is y
    and X is ever mm of image

    IMAGE: numpy image
    CALIBRATION: calibration info
    :parameter image: [C, X, Y, Z] bool tensor
    :return: Array
    """

    image = image[0, ...].sum(-1).gt(3)
    assert image.max() > 0


    image = skimage.transform.downscale_local_mean(image.numpy(), (10, 10)) > 0
    image = skimage.morphology.binary_closing(image)

    image = skimage.morphology.diameter_closing(image, 10)



    for i in range(2):
        image = skimage.morphology.binary_erosion(image)

    image = skimage.morphology.skeletonize(image)

    # first reshape to a logical image format and do a max project
    if image.ndim > 2:
        image = image.transpose((1, 2, 3, 0)).mean(axis=3) / 2 ** 16
        image = skimage.exposure.adjust_gamma(image[:, :, 2], .2)
        image = skimage.filters.gaussian(image, sigma=2) > .5
        image = skimage.morphology.binary_erosion(image)

    # Sometimes there are NaN or inf we have to take care of
    image[np.isnan(image)] = 0
    try:
        center_of_mass = np.array(scipy.ndimage.center_of_mass(image))
        while image[int(center_of_mass[0]), int(center_of_mass[1])] > 0:
            center_of_mass += 1
    except ValueError:
        center_of_mass = [image.shape[0], image.shape[1]]

    # Turn the binary image into a list of points for each pixel that isnt black

    x, y = image.nonzero()
    x += -int(center_of_mass[0])
    y += -int(center_of_mass[1])

    # Transform into spherical space
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)

    # sort by theta
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]

    # there will be a break somewhere because the cochlea isnt a full circle
    # Find the break and subtract 2pi to make the fun continuous
    loc = np.abs(theta[0:-2:1] - theta[1:-1:1])

    theta[loc.argmax()::] += -2 * np.pi
    ind = theta.argsort()[1:-1:1]
    theta = theta[ind]
    r = r[ind]

    # seems to work better if we downsample by interpolation
    # theta_i =np.linspace(theta.min(), theta.max(), 100)
    # r_i= np.interp(theta_i, theta, r)
    # theta = theta_i
    # r = r_i

    # r_i = np.linspace(r.min(), r.max(), 200)
    # theta_i = np.interp(r_i, r, theta)
    # theta = theta_i
    # r = r_i

    # run a spline in spherical space after sorting to get a best approximated fit
    tck, u = splprep([theta, r], w=np.ones(len(r)) / len(r), s=1.5e-6, k=3)
    u_new = np.arange(0, 1, 1e-4)

    # get new values of theta and r for the fitted line
    theta_, r_ = splev(u_new, tck)

    # plt.plot(theta, r, 'k.')
    # plt.xlabel('$\Theta$ (Radians)')
    # plt.ylabel('Radius')
    # plt.plot(theta_, r_)
    # plt.show()

    kernel = GPy.kern.RBF(input_dim=1, variance=100., lengthscale=5.)
    m = GPy.models.GPRegression(theta[:, np.newaxis], r[:, np.newaxis], kernel)
    m.optimize()
    r_, _ = m.predict(theta[:, np.newaxis])
    r_ = r_[:, 0]
    theta_ = theta

    x_spline = r_ * np.cos(theta_) + center_of_mass[1]
    y_spline = r_ * np.sin(theta_) + center_of_mass[0]

    # x_spline and y_spline have tons and tons of data points.
    # We want equally spaced points corresponding to a certain distance along the cochlea
    # i.e. we want a point ever mm which is not guaranteed by x_spline and y_spline
    equal_spaced_points = []
    for i, coord in enumerate(zip(x_spline, y_spline)):
        if i == 0:
            base = coord
            equal_spaced_points.append(base)
        if np.sqrt((base[0] - coord[0]) ** 2 + (base[1] - coord[1]) ** 2) > equal_spaced_distance:
            equal_spaced_points.append(coord)
            base = coord

    equal_spaced_points = np.array(equal_spaced_points) * 10  # <-- Scale factor from above
    equal_spaced_points = equal_spaced_points.T

    curve = tck[1][0]
    if curve[0] > curve[-1]:
        apex = equal_spaced_points[:, -1]
        percentage = np.linspace(1, 0, len(equal_spaced_points[0, :]))
    else:
        apex = equal_spaced_points[:, 0]
        percentage = np.linspace(0, 1, len(equal_spaced_points[0, :]))

    if not diagnostics:
        return equal_spaced_points, percentage, apex
    else:
        return equal_spaced_points, x_spline, y_spline, image, tck, u


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
