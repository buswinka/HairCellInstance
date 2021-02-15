from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
import torch
import numpy as np
import matplotlib.pyplot as plt
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
import GPy

import torchvision.ops


@torch.jit.script
def vector_to_embedding(vector: torch.Tensor) -> torch.Tensor:
    """
    Constructs a mesh grid and adds the vector matrix to it

    :param vector:
    :return:
    """
    x_factor = 1 / 256
    y_factor = 1 / 256
    z_factor = 1 / 256

    xv, yv, zv = torch.meshgrid([torch.linspace(0, x_factor * vector.shape[2], vector.shape[2], device=vector.device),
                                 torch.linspace(0, y_factor * vector.shape[3], vector.shape[3], device=vector.device),
                                 torch.linspace(0, z_factor * vector.shape[4], vector.shape[4], device=vector.device)])

    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1)

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

    sigma = sigma + 1e-10  # when sigma goes to zero, things tend to break

    centroids /= 256

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)

    prob = torch.zeros((embedding.shape[0],
                        centroids.shape[1],
                        embedding.shape[2],
                        embedding.shape[3],
                        embedding.shape[4]), device=embedding.device)

    sigma.pow_(2).mul_(2)

    for i in range(centroids.shape[1]):

        # Calculate euclidean distance between centroid and embedding for each pixel
        euclidean_norm = (embedding - centroids[:, i, :].view(centroids.shape[0], 3, 1, 1, 1)).pow(2)

        # Turn distance to probability and put it in preallocated matrix
        if sigma.shape[0] == 3:
            prob[:, i, :, :, :] = torch.exp((euclidean_norm / sigma.view(centroids.shape[0], 3, 1, 1, 1)).mul(-1).sum(dim=1)).squeeze(1)
        else:
            prob[:, i, :, :, :] = torch.exp((euclidean_norm / sigma).mul(-1).sum(dim=1)).squeeze(1)

    return prob


def estimate_centroids(embedding: torch.Tensor, eps: float = 0.2, min_samples: int = 100,
                       p: float = 2.0, leaf_size: int = 30) -> torch.Tensor:
    """
    Assume [B, 3, X, Y, Z]
    Warning moves everything to cpu!

    :param embedding:
    :param eps:
    :param min_samples:
    :param p:
    :param leaf_size:
    :return:
    """

    device = embedding.device
    embed_shape = embedding.shape
    embedding = embedding.detach().cpu().squeeze(0).reshape((3, -1))

    x = embedding[0, :]
    y = embedding[1, :]
    z = embedding[2, :]

    ind_x = torch.logical_and(x > 0, x < embed_shape[2]/512)
    ind_y = torch.logical_and(y > 0, y < embed_shape[3]/512)
    ind_z = torch.logical_and(z > 0, z < embed_shape[4]/512)
    ind = torch.logical_and(ind_x, ind_y)
    ind = torch.logical_and(ind, ind_z)

    x = x[ind]
    y = y[ind]
    z = z[ind]

    # ind = torch.randperm(len(x))
    # n_samples = 500000
    # ind = ind[0:n_samples:1]

    x = x[0:-1:5]
    y = y[0:-1:5]
    z = z[0:-1:5]

    # x = x[ind].mul(512).round().numpy()
    # y = y[ind].mul(512).round().numpy()
    # z = z[ind].mul(512).round().numpy()

    x = x.mul(512).round().numpy()
    y = y.mul(512).round().numpy()
    z = z.mul(512).round().numpy()


    X = np.stack((x, y, z)).T
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, p=p, leaf_size=leaf_size).fit(X)
    # db = HDBSCAN(min_samples=min_samples).fit(X)
    labels = db.labels_

    unique, counts = np.unique(labels, return_counts=True)

    centroids = []
    scores = []

    for u, s in zip(unique, counts):
        if u == -1:
            continue

        index = labels == u
        c = X[index, :].mean(axis=0)
        centroids.append(c)
        scores.append(s)

    if len(centroids) == 0:
        centroids = torch.empty((1, 0, 3)).to(device)
    else:
        centroids = torch.tensor(centroids).to(device).unsqueeze(0)
        # centroids[:, :, 0] *= 10
        # centroids[:, :, 1] *= 10
        # centroids[:, :, 2] *= 10

    # Works best with non maximum supression
    centroids_xy = centroids[:, :, [0, 1]]
    wh = torch.ones(centroids_xy.shape) * 12 # <- I dont know why this works but it does so deal with it????
    boxes = torch.cat((centroids_xy, wh.to(centroids_xy.device)), dim=-1)
    boxes = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy')
    keep = torchvision.ops.nms(boxes.squeeze(0), torch.tensor(scores).float().to(centroids_xy.device), 0.075)

    return centroids[:, keep, :]


def get_cochlear_length(image: torch.Tensor,
                        equal_spaced_distance: float = 0.1,
                        diagnostics=False) -> torch.Tensor:
    """
    Input an image ->
    max project ->
    reduce image size ->
    run b-spline fit of resulting data on myosin channel ->
    return array of shape [2, X] where [0,:] is x and [1,:] is y
    and X is ever mm of image

    IMAGE: torch image
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

    plt.figure(figsize=(3,3))
    plt.title('Downsample, binary preprocessing')
    plt.imshow(image)
    plt.savefig('curve_downsample.svg' )
    plt.show()

    image = skimage.morphology.skeletonize(image, method='lee')

    plt.figure(figsize=(3,3))
    plt.title('skeletonize')
    plt.imshow(image)
    plt.savefig('skeletonized.svg')
    plt.show()

    # first reshape to a logical image format and do a max project
    # for development purposes only, might want to predict curve from base not mask
    # if False: #  image.ndim > 2:
    #     image = image.transpose((1, 2, 3, 0)).mean(axis=3) / 2 ** 16
    #     image = skimage.exposure.adjust_gamma(image[:, :, 2], .2)
    #     image = skimage.filters.gaussian(image, sigma=2) > .5
    #     image = skimage.morphology.binary_erosion(image)

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


    plt.figure(figsize=(5,5))
    plt.plot(y, x, 'k.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('X_Y.svg')
    plt.show()

    # Transform into spherical space
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)


    # sort by theta
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]

    plt.figure(figsize=(3,3))
    plt.plot(theta, r, 'k.')
    plt.ylabel('Radius')
    plt.xlabel('$\theta$ (Radians)')
    plt.savefig('polar_without_corrections.svg')
    plt.show()

    # there will be a break somewhere because the cochlea isnt a full circle
    # Find the break and subtract 2pi to make the fun continuous
    loc = np.abs(theta[0:-2:1] - theta[1:-1:1])

    theta[loc.argmax()::] += -2 * np.pi
    ind = theta.argsort()[1:-1:1]
    theta = theta[ind]
    r = r[ind]

    plt.figure(figsize=(3,3))
    plt.plot(theta, r, 'k.')
    plt.ylabel('Radius')
    plt.xlabel('$\Theta$ (Radians)')
    plt.savefig('polar_with_corrections.svg')
    plt.show()

    # run a spline in spherical space after sorting to get a best approximated fit
    tck, u = splprep([theta, r], w=np.ones(len(r)) / len(r), s=1.5e-6, k=3)
    u_new = np.arange(0, 1, 1e-4)

    # get new values of theta and r for the fitted line
    theta_, r_ = splev(u_new, tck)

    plt.figure(figsize=(3,3))
    plt.plot(theta, r, 'k.')
    plt.xlabel('$\Theta$ (Radians)')
    plt.ylabel('Radius')
    plt.plot(theta_, r_, 'r-')
    plt.savefig('fit_overlay.svg')
    plt.show()

    kernel = GPy.kern.RBF(input_dim=1, variance=80., lengthscale=5.)  # 100 before
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
    dd = torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/embed_data.trch')
    embed = dd['embed']
    centroids = dd['centroids']
    cent = estimate_centroids(embed, 0.003, 20)  # 0.0081, 160
    x = embed.detach().cpu().numpy()[0, 0, ...].flatten()
    y = embed.detach().cpu().numpy()[0, 1, ...].flatten()
    plt.figure(figsize=(10, 10))
    plt.hist2d(x, y, bins=256, range=((0, 1), (0, 1)))
    plt.plot(cent[0, :, 0].div(512).detach().cpu().numpy(), cent[0, :, 1].div(512).detach().cpu().numpy(), 'ro')
    plt.plot(centroids[0, :, 0].cpu() / 256, centroids[0, :, 1].cpu() / 256, 'bo')
    plt.show()

