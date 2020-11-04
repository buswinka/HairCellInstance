import torch


@torch.jit.script
def calculate_vector(mask: torch.Tensor) -> torch.Tensor:
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


def vector_to_embedding(vector: torch.Tensor) -> torch.Tensor:
    """

    :param embedding: [B, K=3, X, Y, Z] torch.Tensor where K is the vector component {X, Y, Z}
    :param sigma: torch.Tensor of shape = (1) or (embedding.shape)
    :return:
    """

    xv, yv, zv = torch.meshgrid([torch.linspace(0, 1, vector.shape[2]),
                                 torch.linspace(0, 1, vector.shape[3]),
                                 torch.linspace(0, 1, vector.shape[4])])
    mesh = torch.cat((xv.unsqueeze(0).unsqueeze(0),
                      yv.unsqueeze(0).unsqueeze(0),
                      zv.unsqueeze(0).unsqueeze(0)), dim=1).to(vector.device)

    return mesh + vector


def embedding_to_probability(embedding: torch.Tensor, centroids: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """

    :param embedding: [B, K=3, X, Y, Z] torch.Tensor where K is the likely centroid component: {X, Y, Z}
    :param centroids: [B, I, K_true=3] torch.Tensor where I is the number of instances in the image and K_true is centroid
                        {x, y, z}
    :param sigma: torch.Tensor of shape = (1) or (embedding.shape)
    :return: [B, I, X, Y, Z] of probabilities for instance I
    """

    # Calculates the euclidean distance between the centroid and the embedding
    # embedding [B, 3, X, Y, Z] -> euclidean_norm[B, 1, X, Y, Z]
    # euclidean_norm = sqrt(Δx^2 + Δy^2 + Δz^2) where Δx = (x_embed - x_centroid_i)
    for i in range(centroids.shape[1]):
        euclidean_norm = torch.cat((
                            (embedding[:, 0, :, :, :] - centroids[:, i, 0]).unsqueeze(1),
                            (embedding[:, 1, :, :, :] - centroids[:, i, 1]).unsqueeze(1),
                            (embedding[:, 2, :, :, :] - centroids[:, i, 2]).unsqueeze(1),
        ), dim=1).pow(2).sum(dim=1).sqrt().unsqueeze(1)

        instance_prob = torch.exp(-1 * (euclidean_norm ** 2) / (2 * sigma ** 2))
        prob = instance_prob if i == 0 else torch.cat((prob, instance_prob), dim=1)

    return prob
