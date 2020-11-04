import src.utils
import torch
import matplotlib.pyplot as plt
import skimage.io as io

def test_vector():
    return True
    data = io.imread('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/test.labels.tif')
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).transpose(2,-1)
    out = src.utils.calculate_vector(data)
    assert out.shape[2] == data.shape[2]
    assert out.shape[3] == data.shape[3]
    assert out.shape[4] == data.shape[4]
    assert out.max() < 1
    assert out.min() > -1
    assert out[0,0,0,0,0] == 0
    return out

