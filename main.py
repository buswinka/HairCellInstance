from src.models.HCNet import HCNet
from src.utils import calculate_indexes, remove_edge_cells, remove_small_cells
import src.functional
from src.cell import cell
import skimage.io as io
import torch
import click
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from src.explore_lif import Reader

import glob
import os.path

# default_path = '/media/chris/Padlock_3/Dose-adjusted injection data/**/**/'
# default_path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/shit'
default_path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/'


@click.command()
@click.option('--path', default=default_path, help='Path to image')
def analyze(path: str) -> None:
    print('Loading Model...', end=' ')
    model = torch.jit.script(HCNet(in_channels=3, out_channels=4, complexity=15)).cuda()
    # model.load_state_dict(torch.load('trained_models/Jan2_REALLY_GOOD.hcnet'))
    model.load_state_dict(torch.load('Jan15_1_background_reject.hcnet'))
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
    print('Done')

    files = glob.glob(os.path.join(path, '*.lif'))
    print(f'Analyzing {len(files)} files...')
    for f in files:
        # (Z, Y, X, C)
        # if not os.path.exists(f + '.cells.trch'):
        #     continue

        print('Loading ' + f + '... ', end=' ')
        reader = Reader(f)
        series = reader.getSeries()
        for i, header in enumerate(reader.getSeriesHeaders()):
            print(header.getName())
            if header.getName() == 'Myo7a G10.0 La1.0':###'TileScan 1 Merged':
                chosen = series[i]
                for c in range(4):
                    if c == 0:
                        image_base = torch.from_numpy(chosen.getXYZ(T=0, channel=c)).unsqueeze(0)
                    else:
                        image_base = torch.cat((image_base, torch.from_numpy(
                            chosen.getXYZ(T=0, channel=c, dtype=np.uint16).astype(np.int16)).unsqueeze(0)), dim=0)
                break
        # [C, X, Y, Z]
        # c needs to be [Blue, Green, Yellow, Red]
        image_base = image_base.numpy()
        image_base = image_base[:, 0:-1:2, 0:-1:2, :]
        base_im_shape = image_base.shape

        print('Done ', image_base.shape)

        out_img = torch.zeros((1, image_base.shape[1], image_base.shape[2], image_base.shape[3]),
                              dtype=torch.long)  # (1, X, Y, Z)

        max_cell = 0

        x_ind = calculate_indexes(20, 257, base_im_shape[1], base_im_shape[1])
        y_ind = calculate_indexes(20, 257, base_im_shape[2], base_im_shape[2])
        total = len(x_ind) * len(y_ind)

        cells = []

        with torch.no_grad():
            for (x, y) in tqdm(product(x_ind, y_ind), total=total):

                image = torch.from_numpy(image_base[:, x[0]:x[1], y[0]:y[1], :] / 2 ** 16).unsqueeze(0).float().sub(
                    0.5).div(0.5).cuda()

                if image.max() == -1:
                    continue

                # if image is all zeros or doesnt have a myo signal of 20% of max intensity
                if torch.sum(image[:, 2, ...] > ((20000 / 2 ** 16) - 0.5) / 0.5) < 100:
                    continue

                # Eval model
                out = model(image[:, [0, 2, 3], ...].cuda(), 5)
                prob_map = torch.sigmoid(out[:, -1, ...]).unsqueeze(1)
                out = out[:, 0:3:1, ...]
                embed = src.functional.vector_to_embedding(out)

                # Remove obviously bad embeddings
                for i in range(3):
                    embed[:, i, ...][prob_map.squeeze(0) < 0.9] = -10

                # Skip if all embeddings are bad
                if embed.max() == -10:
                    continue

                cent = src.functional.estimate_centroids(embed, 1, 80)

                if cent.nelement() == 0:
                    continue

                out = src.functional.vector_to_embedding(out)
                out = src.functional.embedding_to_probability(out, cent, torch.tensor([0.01]))

                value, out = out.max(1)
                out[value < 0.25] = 0
                out[prob_map.squeeze(1) < 0.5] = 0

                # Post Processing
                out = remove_edge_cells(out)
                out = remove_small_cells(out)
                out[out > 0] = out[out > 0] + max_cell

                if out.max() > 0:
                    max_cell = out.max()

                out_img[:, x[0]:x[1] - 1, y[0]:y[1] - 1:, 0:out.shape[-1]][out != 0] = out[out != 0].cpu()

                for u in torch.unique(out):
                    if u == 0:
                        continue
                    center = (out == u).nonzero().float().mean(0)
                    center[1] += x[0]
                    center[2] += y[0]
                    cells.append(cell(image=image[:, :, :-1:, :-1:, 0:out.shape[-1]].cpu(),
                                      mask=(out == u).cpu().unsqueeze(0),
                                      place=center))

        volumes = []
        for c in cells:
            volumes.append(c.volume.item())
        plt.hist(torch.tensor(volumes).numpy(), bins=40)
        plt.savefig(f + '.figure.png', dpi=400)
        plt.show()

        # Get cochlear length
        equal_spaced_points, percentage, apex = src.functional.get_cochlear_length(out_img, equal_spaced_distance=0.1)
        torch.save(equal_spaced_points, f + '.curvature.trch')
        torch.save(percentage, f + '.percentage.trch')
        torch.save(apex, f + '.apex.trch')


        plt.imshow(image[0,[0,2,3],:,:,:].cpu().numpy().max(-1).transpose((1,2,0)))
        plt.plot(equal_spaced_points[1, :], equal_spaced_points[0, :], 'r-')
        plt.savefig('cochlea_with_curve.svg')
        plt.show()

        torch.save(cells, f + '.cells.trch')
        out_img = out_img.int().squeeze(0).numpy().transpose((2, 1, 0))
        print(out_img.shape)
        print(out_img.max())
        io.imsave(f + '.analyzed.tif', out_img)


if __name__ == '__main__':
    analyze()
