from src.models.HCNet import HCNet
from src.utils import calculate_indexes, remove_edge_cells
import src.functional
from src.cell import cell
import skimage.io as io
import torch
import click
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt

default_path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/' \
               'Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged-test_part.tif'


@click.command()
@click.option('--path', default=default_path, help='Path to image')
def analyze(path: str) -> None:
    print('Loading Model...', end=' ')
    model = torch.jit.script(HCNet(in_channels=3, out_channels=6, complexity=10)).cuda()
    model.load_state_dict(
        torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/overnight_best.hcnet'))
    model.eval()
    print('Done')

    # (Z, Y, X, C)
    print('Loading Image...', end=' ')
    image_base = io.imread(path)
    base_im_shape = image_base.shape
    print('Done')

    out_img = torch.zeros((1, image_base.shape[2], image_base.shape[1], image_base.shape[0]),
                          dtype=torch.long)  # (1, X, Y, Z)

    max_cell = 0

    x_ind = calculate_indexes(100, 513, base_im_shape[2], base_im_shape[2])
    y_ind = calculate_indexes(100, 513, base_im_shape[1], base_im_shape[1])
    total = len(x_ind) * len(y_ind)

    with torch.no_grad():
        for (x, y) in tqdm(product(x_ind, y_ind), total=total):

            image = torch.from_numpy(image_base[:, y[0]:y[1], x[0]:x[1], [0, 2, 3]] / 2 ** 16).unsqueeze(0)
            image = image.transpose(1, 3).transpose(0, -1).squeeze().unsqueeze(0).sub(0.5).div(0.5).cuda()

            if image.sum() == 0:
                continue

            out = model(image.float().cuda(), 5)
            sigma = torch.sigmoid(out[:, -3::1, ...])
            out = src.functional.vector_to_embedding(out[:, 0:3:1, ...])
            try:
                cent = src.functional.estimate_centroids(out, 0.009, 60)

            except ValueError:
                # DBSCAN throws an error if it doesnst detect sample
                continue

            if cent.nelement() == 0:
                del image, out, cent
                continue

            out = src.functional.embedding_to_probability(out, cent, sigma)

            value, out = out.max(1)
            out[value < 0.5] = 0
            out[out != 0] += max_cell
            max_cell = out.max()
            # out = out[..., :image.shape[-1]:].cpu().to(out_img.dtype)

            # post processing
            out = remove_edge_cells(out)
            # out = remove_small_cells(out)

            out_img[:, x[0]:x[1]-1, y[0]:y[1]-1:, 0:out.shape[-1]][out != 0] = out[out != 0].cpu()

            del image, out, cent, value

    # curve, percent, apex = src.functional.get_cochlear_length(out_img > 0, 10)
    # print(curve.shape)
    # plt.imshow(out_img[0, ...].gt(0).sum(-1).gt(3))
    # plt.plot(curve[0, :], curve[1, :], 'r')
    # plt.show()

    cells = []
    print(image_base.shape, type(image_base))
    image_base = torch.from_numpy(image_base / 2 ** 16).transpose(0, -1).transpose(1, 2)

    # for u in tqdm(torch.unique(out_img)):
    #     cells.append(cell(image_base.unsqueeze(0), (out_img == u).unsqueeze(0)))
    #
    # torch.save(cells, 'cells.trch')
    torch.save(out_img, 'out_image.trch')
    out_img = out_img.squeeze(0).int().numpy().transpose((2, 1, 0))
    io.imsave('big_img.tif', out_img)


if __name__ == '__main__':
    analyze()
