import torch
import torch.nn as nn

class jaccard_loss:
    # def __call__(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns intersection over union
    #
    #     :param predicted: [B, I, X, Y, Z] torch.Tensor of probabilities calculated from src.utils.embedding_to_probability
    #                       where B: is batch size, I: instances in image
    #     :param ground_truth: [B, I, X, Y, Z] segmentation mask for each instance (I).
    #     :return:
    #     """
    #     intersection = (predicted + ground_truth) > 1
    #     union = (predicted + ground_truth) > 0
    #
    #     loss = 1 - intersection/union + 1e-7
    #
    #     return loss
    def __call__(self, pred: torch.Tensor, mask: torch.Tensor):
        """
        Calculates the dice loss between pred and mask

        :param pred: torch.Tensor | probability map of shape [B,C,X,Y,Z] predicted by hcat.unet
        :param mask: torch.Tensor | ground truth probability map of shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped
                     to identical size of pred
        :return: torch.float | calculated dice loss
        """
        # loss = nn.SmoothL1Loss()
        return 1 - (2 * (pred * mask).sum() + 1e-10) / ((pred + mask).sum() + 1e-10)
        # return loss(pred, mask)
