import torch


@torch.jit.script
def calculate_vector(mask:torch.Tensor) -> torch.Tensor:
    """
    mask: [1,1,x,y,z] mask of ints

    :param mask:
    :return: [1,1,x,y,z] vector to the center of mask
    """

    # com = torch.zeros(mask.shape)
    vector = torch.zeros((1,3,mask.shape[2],mask.shape[3],mask.shape[4]))
    xv,yv,zv = torch.meshgrid([torch.linspace(0,1,mask.shape[2]), torch.linspace(0,1,mask.shape[3]), torch.linspace(0,1,mask.shape[4])])

    for u in torch.unique(mask):
        if u == 0:
            continue
        index = ((mask==u).nonzero()).float().mean(dim=0)

        # Set between 0 and 1
        index[2] = index[2] / mask.shape[2]
        index[3] = index[3] / mask.shape[3]
        index[4] = index[4] / mask.shape[4]

        vector[0,0,:,:,:][mask[0,0,:,:,:]==u] = xv[mask[0,0,:,:,:]==u]-index[2]
        vector[0,1,:,:,:][mask[0,0,:,:,:]==u] = yv[mask[0,0,:,:,:]==u]-index[3]
        vector[0,2,:,:,:][mask[0,0,:,:,:]==u] = zv[mask[0,0,:,:,:]==u]-index[4]

    return vector