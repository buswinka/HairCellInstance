import torch

class jaccard_loss:
    def __call__(self, predicted: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Returns intersection over union

        :param predicted: [B, I, X, Y, Z] torch.Tensor of probabilities calculated from src.utils.embedding_to_probability
                          where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] segmentation mask for each instance (I).
        :return:
        """

        intersection = torch.logical_and(predicted, ground_truth)
        union = torch.logical_or(predicted, ground_truth)

        loss = 1 - intersection.sum()/union.sum()

        return loss




