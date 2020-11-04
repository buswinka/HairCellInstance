import torch
from torch.utils.data import DataLoader
import numpy as np
import glob
import os.path
import skimage.io as io


class dataset(DataLoader):
    def __init__(self, path):
        super(DataLoader, self).__init__()

        # Find only files
        files = glob.glob(os.path.join(path, '*.labels.tif'))

        self.mask = []
        self.image = []
        self.centroids = []

        for f in files:
            image_path = os.path.splitext(f)[0]
            image_path = os.path.splitext(image_path)[0] + '.tif'

            image = torch.from_numpy(io.imread(image_path).astype(np.int16)).unsqueeze(0)

            image = image.transpose(1,3).transpose(0,-1).squeeze()

            mask = torch.from_numpy(io.imread(f)).transpose(0,2).unsqueeze(0)

            self.mask.append(colormask_to_torch_mask(mask).float())
            self.image.append(image.float())
            self.centroids.append(colormask_to_centroids(mask).float())

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, item):
        return self.image[item], self.mask[item], self.centroids[item]



def colormask_to_torch_mask(colormask: torch.tensor) -> torch.Tensor:
    """

    :param colormask: [C=1, X, Y, Z]
    :return:
    """
    num_cells = torch.max(colormask) # cells are denoted by integers 1->max_cell
    shape = (num_cells, colormask.shape[1], colormask.shape[2], colormask.shape[3])
    mask = torch.zeros(shape)

    for i in range(num_cells):
        mask[i, :, :, :] = (colormask[0, :, :, :] == i)

    return mask


def colormask_to_centroids(colormask: torch.Tensor) -> torch.Tensor:
    num_cells = torch.max(colormask) # cells are denoted by integers 1->max_cell
    shape = (num_cells, 3)
    centroid = torch.zeros(shape)

    for i in range(num_cells):
        if i == 0:
            continue
            print(torch.nonzero(colormask == i).shape)
            centroid[i,:] = torch.mean(torch.nonzero(colormask == i), dim=1)
    return centroid
