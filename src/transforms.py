import torch
import torchvision.transforms.functional
from PIL.Image import Image
import numpy as np
from typing import Dict, Tuple, Union, List
import elasticdeform


# ----------------- Assumtions -------------------#
# Every image is expected to be [C, X, Y, Z]
# Every transform's input has to be Dict[str, torch.Tensor]
# Every transform's output has to be Dict[str, torch.Tensor]


class nul_crop:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ind = torch.nonzero(data_dict['masks'])  # -> [I, 4] where 4 is ndims

        x_max = ind[:, 1].max().int().item()
        y_max = ind[:, 2].max().int().item()
        z_max = ind[:, 3].max().int().item()

        x = ind[:, 1].min().int().item()
        y = ind[:, 2].min().int().item()
        # z = ind[:, 3].min().int().item()

        w = x_max - x
        h = y_max - y
        # d = z_max-z

        data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=0, w=w, h=h, d=z_max)
        data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=0, w=w, h=h, d=z_max)

        return data_dict


class random_crop:
    def __init__(self, shape: Tuple[int, int, int] = (256, 256, 26)) -> None:
        self.w = shape[0]
        self.h = shape[1]
        self.d = shape[2]

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        For example

        :param data_dict:
        :return:
        """
        shape = data_dict['image'].shape

        x_max = shape[1] - self.w if shape[1] - self.w > 0 else 1
        y_max = shape[2] - self.h if shape[2] - self.h > 0 else 1
        z_max = shape[3] - self.d if shape[3] - self.d > 0 else 1

        x = torch.randint(x_max, (1, 1)).item()
        y = torch.randint(y_max, (1, 1)).item()
        z = torch.randint(z_max, (1, 1)).item()

        data_dict['image'] = _crop(data_dict['image'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)
        data_dict['masks'] = _crop(data_dict['masks'], x=x, y=y, z=z, w=self.w, h=self.h, d=self.d)

        return data_dict


class random_v_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.vflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class random_h_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.hflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image'])))
            data_dict['masks'] = _reshape(self.fun(_shape(data_dict['masks'])))

        return data_dict


class normalize:
    def __init__(self, mean: List[float] = [0.5], std: List[float] = [0.5]) -> None:
        self.mean = mean
        self.std = std
        self.fun = torch.jit.script(torchvision.transforms.functional.normalize)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        data_dict['image'] = self.fun(data_dict['image'], self.mean, self.std)
        return data_dict


class gaussian_blur:
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5) -> None:
        self.kernel_targets = kernel_targets
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.gaussian_blur)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.rand(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            data_dict['image'] = _reshape(self.fun(_shape(data_dict['image']), [kern, kern]))
        return data_dict


class adjust_brightness:
    def __init__(self, rate: float = 0.5, range_brightness: Tuple[float, float] = (-0.5, 0.5)) -> None:
        self.rate = rate
        self.range = range_brightness
        self.fun = _adjust_brightness

    def __call__(self, data_dict):

        if torch.rand(1) < self.rate:
            val = torch.FloatTensor(data_dict['image'].shape[0]).uniform_(self.range[0], self.range[1])
            data_dict['image'] = self.fun(data_dict['image'], val)

        return data_dict


# needs docstring
class adjust_contrast:
    def __init__(self, rate: float = 0.5, range_contrast: Tuple[float, float] = (.3, 1.7)) -> None:
        self.rate = rate
        self.range = range_contrast
        self.fun = torch.jit.script(torchvision.transforms.functional.adjust_brightness)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        for i in range(data_dict['image'].shape[0]):
            if torch.rand(1) < self.rate:
                val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1])  # .to(image.device)
                data_dict['image'][i, :, :, :] = torchvision.transforms.functional.adjust_contrast(
                    data_dict['image'][i, :, :, :],
                    val.to(
                        data_dict['image'].device))

        return data_dict


class elastic_deformation:
    def __init__(self, grid_shape: Tuple[int, int, int] = (2, 2, 2), scale: int = 2):
        self.x_grid = grid_shape[0]
        self.y_grid = grid_shape[1]
        self.z_grid = grid_shape[2] if len(grid_shape) > 2 else None
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        device = data_dict['image'].device
        image = data_dict['image'].cpu().numpy()
        mask  = data_dict['masks'].cpu().numpy()
        dtype = image.dtype

        displacement = np.random.randn(3, self.x_grid, self.y_grid, self.z_grid) * self.scale
        image = elasticdeform.deform_grid(image, displacement, axis=(1, 2, 3))
        mask = elasticdeform.deform_grid(mask, displacement, axis=(1, 2, 3), order=0)

        image[image < 0] = 0.0
        image[image > 1] = 1.0
        image.astype(dtype)

        data_dict['image'] = torch.from_numpy(image).to(device)
        data_dict['masks'] = torch.from_numpy(mask).to(device)

        return data_dict


# needs docstring
class random_affine:
    def __init__(self, rate: float = 0.5, angle: Tuple[int, int] = (-180, 180),
                 shear: Tuple[int, int] = (-5, 5), scale: Tuple[float, float] = (0.9, 1.1)) -> None:
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1) < self.rate:
            angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1])
            shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1])
            scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            translate = torch.tensor([0, 0])

            data_dict['image'] = _reshape(_affine(_shape(data_dict['image']), angle, translate, scale, shear))
            data_dict['masks'] = _reshape(_affine(_shape(data_dict['masks']), angle, translate, scale, shear))

        return data_dict


class to_cuda:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move every element in a dict containing torch tensor to cuda.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader
        :return: Dict[str, torch.Tensor]
        """
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        return data_dict


class to_tensor:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, Union[torch.Tensor, Image, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Convert a PIL image or numpy.ndarray to a torch.Tensor

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader
        :return: Dict[str, torch.Tensor]
        """
        data_dict['image'] = torchvision.transforms.functional.to_tensor(data_dict['image'])
        return data_dict


class adjust_centroids:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        shape = data_dict['masks'].shape
        device = data_dict['masks'].device
        centroid = torch.zeros(shape[0], 3, dtype=torch.float)
        ind = torch.ones(shape[0], dtype=torch.long)

        for i in range(shape[0]):  # num of instances
            indexes = torch.nonzero(data_dict['masks'][i, :, :, :] > 0).float()

            if indexes.shape[0] == 0:
                centroid[i, :] = torch.tensor([-1, -1, -1])
                ind[i] = 0
            else:
                centroid[i, :] = torch.mean(indexes, dim=0)

        # centroid[:, 0] /= shape[1]
        # centroid[:, 1] /= shape[2]
        # centroid[:, 2] /= shape[3]

        data_dict['centroids'] = centroid[ind.bool()].to(device)
        data_dict['masks'] = data_dict['masks'][ind.bool(), :, :, :]

        return data_dict


class debug:
    def __init__(self, ind: int = 0):
        self.ind = ind

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = data_dict['image']
        mask = data_dict['masks']
        try:
            assert image.shape[-1] == mask.shape[-1]
            assert image.shape[-2] == mask.shape[-2]
            assert image.shape[-3] == mask.shape[-3]

            assert image.max() <= 1
            assert mask.max() <= 1
            assert image.min() >= 0
            assert mask.min() >= 0
        except Exception as ex:
            print(self.ind)
            raise ex

        return data_dict


@torch.jit.script
def _affine(img: torch.Tensor, angle: torch.Tensor, translate: torch.Tensor, scale: torch.Tensor,
            shear: torch.Tensor) -> torch.Tensor:
    """

    :param img:
    :param angle:
    :param translate:
    :param scale:
    :param shear:
    :return:
    """
    angle = float(angle.item())
    scale = float(scale.item())
    shear = [float(shear.item())]
    translate_list = [int(translate[0].item()), int(translate[1].item())]
    return torchvision.transforms.functional.affine(img, angle, translate_list, scale, shear)


@torch.jit.script
def _shape(img: torch.Tensor) -> torch.Tensor:
    # [C, X, Y, Z] -> [C, 1, X, Y, Z] ->  [C, Z, X, Y, 1] -> [C, Z, X, Y]
    return img.unsqueeze(1).transpose(1, -1).squeeze(-1)


@torch.jit.script
def _reshape(img: torch.Tensor) -> torch.Tensor:
    # [C, Z, X, Y] -> [C, Z, X, Y, 1] ->  [C, 1, X, Y, Z] -> [C, Z, X, Y]
    return img.unsqueeze(-1).transpose(1, -1).squeeze(1)


@torch.jit.script
def _crop(img: torch.Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> torch.Tensor:
    if img.ndim == 4:
        img = img[:, x:x + w, y:y + h, z:z + d]
    elif img.ndim == 5:
        img = img[:, :, x:x + w, y:y + h, z:z + d]
    else:
        raise IndexError('Unsupported number of dimensions')

    return img


@torch.jit.script
def _adjust_brightness(img: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    img = img.add_(val.reshape(img.shape[0], 1, 1, 1).to(img.device))
    img[img < 0] = 0
    img[img > 0] = 1
    return img


# @torch.jit.script
# def _adjust_contrast(img: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
#
#     return
