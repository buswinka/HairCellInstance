import src.model
import torch
import torch.optim
import torch.nn



def test_RDCNet_function():
    """
    Test if RDCNet can evaluate a synthetic input
    :return: None
    """
    x = torch.ones((1,3,512,512,35)).cuda()
    model = src.model.RDCNet(3, 10).cuda()
    out = model(x)
    assert True

def test_RDCNet_train():
    """
    Test if backprop works with RDCNet on synthetic inputs
    :return: None
    """
    x = torch.randn((1, 3, 512, 512, 35)).cuda()
    model = src.model.RDCNet(3, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
    optimizer.zero_grad()
    loss_fun = torch.nn.L1Loss()
    out = model(x)
    x_true = torch.ones(out.shape).cuda()
    loss = loss_fun(out,x_true)
    loss.backward()
    optimizer.step()

