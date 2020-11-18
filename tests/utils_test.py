import src.functional
import torch
import matplotlib.pyplot as plt
import skimage.io as io

def test_vector():
    return True
    data = io.imread('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/test.labels.tif')
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).transpose(2,-1)
    out = src.functional.calculate_vector(data)
    assert out.shape[2] == data.shape[2]
    assert out.shape[3] == data.shape[3]
    assert out.shape[4] == data.shape[4]
    assert out.max() < 1
    assert out.min() > -1
    assert out[0,0,0,0,0] == 0
    return out

def test_vector_to_embedding():
    x = torch.randn((1, 3, 100, 100, 10))
    out = src.functional.vector_to_embedding(x)


def test_convert_embedding_to_probability():
    embed = torch.randn((1,3,100,100,10)).cuda()
    centroid = torch.randn((1, 90, 3)).cuda()

    sigma = torch.Tensor([2]).cuda()
    out = src.functional.embedding_to_probability(embed, centroid, sigma)
    #
    # assert out.shape[1] == centroid.shape[1]
    # assert out.shape[2] == embed.shape[2]
    # assert out.shape[3] == embed.shape[3]
    # assert out.shape[4] == embed.shape[4]
    # assert out.max() < 1
    # assert out.min() > 0

def test_convert_embedding_to_probability_vector():
    embed = torch.randn((1,3,100,100,10)).cuda()
    centroid = torch.randn((1, 90, 3)).cuda()

    sigma = torch.Tensor([2]).cuda()
    out = src.functional.embedding_to_probability_vector(embed, centroid, sigma).cuda()
    #
    # assert out.shape[1] == centroid.shape[1]
    # assert out.shape[2] == embed.shape[2]
    # assert out.shape[3] == embed.shape[3]
    # assert out.shape[4] == embed.shape[4]
    # assert out.max() < 1
    # assert out.min() > 0

def test_compare_convert_embedding_to_probability_vectorized():
    embed = torch.randn((1,3,100,100,10))
    centroid = torch.randn((1, 90, 3))

    sigma = torch.Tensor([2])
    out = src.functional.embedding_to_probability(embed, centroid, sigma)
    out2 = src.functional.embedding_to_probability_vector(embed, centroid, sigma)

    assert out[0,34,66,66,5] == out2[0,34,66,66,5]
    assert out[0,20,66,66,5] == out2[0,20,66,66,5]

    assert out2.shape[1] == centroid.shape[1]
    assert out2.shape[2] == embed.shape[2]
    assert out2.shape[3] == embed.shape[3]
    assert out2.shape[4] == embed.shape[4]
    assert out2.max() < 1
    assert out2.min() > 0