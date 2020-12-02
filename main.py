from src.models.HCNet import HCNet
from src.utils import calculate_indexes, remove_edge_cells
import src.functional
import skimage.io as io
import torch
import numpy as np
import matplotlib.pyplot as plt


model = torch.jit.script(HCNet(in_channels=3, out_channels=3, complexity=30)).cuda()
model.load_state_dict(torch.load('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/fully_trained.hcnet'))
model.eval()

# (Z, Y, X, C)
image_base = io.imread('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged-test_part.tif')
base_im_shape = image_base.shape
print(image_base.shape)

out_img = torch.zeros((1, image_base.shape[2], image_base.shape[1], image_base.shape[0]), dtype=torch.int16) # (1, X, Y, Z)

max_cell = 0

x_ind = calculate_indexes(100, 513, base_im_shape[2], base_im_shape[2])
y_ind = calculate_indexes(100, 513, base_im_shape[1], base_im_shape[1])

i = 0
imax = len(x_ind) * len(y_ind)
with torch.no_grad():
    for x in x_ind:
        for y in y_ind:
            print(i, imax)
            i += 1
            image = torch.from_numpy(image_base[:, y[0]:y[1], x[0]:x[1], [0, 2, 3]] / 2 ** 16).unsqueeze(0)
            image = image.transpose(1, 3).transpose(0, -1).squeeze().unsqueeze(0).sub(0.5).div(0.5).cuda()

            if image.sum() == 0:
                continue

            out = model(image.float().cuda())
            out = src.functional.vector_to_embedding(out)
            cent = src.functional.estimate_centroids(out, 0.01, 80)


            if cent.nelement() == 0:
                del image, out, cent
                continue

            out = src.functional.embedding_to_probability(out, cent, torch.tensor([0.0231/2, 0.0231/2, 0.0231*2]))

            value, out = out.max(1)
            out[value < 0.5] = 0

            out[out != 0] += max_cell
            max_cell = out.max()
            out = out[..., 0:image.shape[-1]].cpu().to(out_img.dtype)

            # post processing
            out = remove_edge_cells(out)
            # out = remove_small_cells(out)

            out_img[:, x[0]:x[1], y[0]:y[1]:, 0:image.shape[-1]][out != 0] = out[out != 0]

            del image, out, cent, value

# u, counts = torch.unique(out_img, return_counts=True)
#
# for small in u[counts < 500]:
#     out_img[out_img == small] = 0
#
# u, counts = torch.unique(out_img, return_counts=True)
# counts = counts[u!=0]
# plt.hist(counts.numpy(),bins=30)
# plt.show()


torch.save(out_img, 'out_image.trch')
out_img = out_img.squeeze(0).int().numpy().transpose((2,1,0))
io.imsave('big_img.tif', out_img)

