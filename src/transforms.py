import torch
import torchvision.transforms.functional
from PIL.Image import Image
import numpy as np
from typing import Dict, Tuple, Union, List

# ----------------- Assumtions -------------------#
# Every image is expected to be [B, C, X, Y, Z]
# Every transform's input has to be Dict[str, torch.Tensor]
# Every transform's output has to be Dict[str, torch.Tensor]



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
            data_dict['image'] = self.fun(data_dict['image'])
            data_dict['masks'] = self.fun(data_dict['masks'])

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
            data_dict['image'] = self.fun(data_dict['image'])
            data_dict['masks'] = self.fun(data_dict['masks'])

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
            data_dict['image'] = self.fun(data_dict['image'], [kern, kern])
        return data_dict


class random_resize:
    def __init__(self, rate: float = 0.5, scale: tuple = (300, 1440)) -> None:
        self.rate = rate
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly resizes an mask

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.rand(1) < self.rate:
            size = torch.randint(self.scale[0], self.scale[1], (1, 1)).item()
            data_dict['image'] = torchvision.transforms.functional.resize(data_dict['image'], size)
            data_dict['masks'] = torchvision.transforms.functional.resize(data_dict['masks'], size)

        return data_dict


class adjust_brightness:
    def __init__(self, rate: float = 0.5, range_brightness: Tuple[float, float] = (0.3, 1.7)) -> None:
        self.rate = rate
        self.range = range_brightness
        self.fun = torch.jit.script(torchvision.transforms.functional.adjust_brightness)

    def __call__(self, data_dict):

        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1])
        if torch.rand(1) < self.rate:
            data_dict['image'] = self.fun(data_dict['image'], val.item())

        return data_dict


# needs docstring
class adjust_contrast:
    def __init__(self, rate: float = 0.5, range_contrast: Tuple[float, float] = (.3, 1.7)) -> None:
        self.rate = rate
        self.range = range_contrast
        self.fun = torch.jit.script(torchvision.transforms.functional.adjust_brightness)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if torch.rand(1) < self.rate:
            val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1])  # .to(image.device)
            data_dict['image'] = torchvision.transforms.functional.adjust_contrast(data_dict['image'], val.to(data_dict['image'].device))

        return data_dict


# needs docstring
class random_affine:
    def __init__(self, rate: float = 0.5, angle: Tuple[int, int] = (-180, 180),
                 shear: Tuple[int, int] = (-45, 45), scale: Tuple[float, float] = (0.9, 1.5)) -> None:
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

            data_dict['image'] = _affine(data_dict['image'], angle, translate, scale, shear)
            data_dict['masks'] = _affine(data_dict['masks'], angle, translate, scale, shear)

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


class stack_image:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_dict['image'] = torch.cat((data_dict['image'], data_dict['image'], data_dict['image']), dim=0)
        return data_dict


@torch.jit.script
def _affine(img: torch.Tensor, angle: torch.Tensor, translate: torch.Tensor, scale: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
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
    # [B, C, X, Y, Z] -> [B, C, 1, X, Y, Z] ->  [B, C, Z, X, Y, 1] -> [B, C, Z, X, Y]
    return img.unsqueeze(2).transpose(2, -1).squeeze(-1)


@torch.jit.script
def _reshape(img: torch.Tensor) -> torch.Tensor:
    # [B, C, Z, X, Y] -> [B, C, Z, X, Y, 1] ->  [B, C, 1, X, Y, Z] -> [B, C, Z, X, Y]
    return img.unsqueeze(-1).transpose(2, -1).squeeze(2)
