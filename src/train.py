import src.dataloader
import src.loss
import src.utils
import torch
from src.models.RDCNet import RDCNet
from src.models.RecurrentUnet import RecurrentUnet
import torch
import torch.optim
import torch.nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

epochs = 100

model = RDCNet(4, 3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fun = src.loss.jaccard_loss()

data = src.dataloader.dataset('/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test')
data = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

for e in range(epochs):
    print(e)
    for image, mask, centroids in data:

        print(image.shape, mask.shape, centroids.shape)

        with autocast():
            optimizer.zero_grad()
            out = model(image.cuda())

        out = src.utils.vector_to_embedding(out)
        out = src.utils.embedding_to_probability(out, centroids.cuda(), torch.tensor([2]).cuda())
        loss = loss_fun(out, mask.cuda())

        loss.backward()
        optimizer.step()


