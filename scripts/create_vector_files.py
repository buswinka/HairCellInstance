import src.utils
import skimage.io
import torch
import pickle

path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test/tiny.labels.tif'

mask = skimage.io.imread(path)
mask = torch.from_numpy(mask).transpose(0,2).unsqueeze(0).unsqueeze(0)
vector = src.utils.calculate_vector(mask)

pickle.dump(vector, open('../data/test/tiny.vector.pkl', 'wb'))