import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu


def GaussianKernel(x, y, bs):

    sigma = 1
    I = torch.ones([bs, 1])
    x_tmp = torch.sum(torch.square(x), dim=1)
    y_tmp = torch.sum(torch.square(y), dim=1)
    x2 = torch.matmul(I, x_tmp.T)
    y2 = torch.matmul(I, y_tmp.T)
    normTmp = x2 - 2 * torch.matmul(y, x.T) + y2.T
    norm = torch.sum(normTmp, dim=1)
    K = torch.exp(-1 * norm / sigma)

    return K


def MMD(x, y, bs):
    Kxx = GaussianKernel(x, x, bs) / (bs ** 2)
    Kxy = GaussianKernel(x, y, bs) / (bs ** 2)
    Kyy = GaussianKernel(y, y, bs) / (bs ** 2)

    MMD = Kxx - 2 * Kxy + Kyy
    return MMD.sum()


class MMDEstimator(object):

    def __init__(self, input_dim, D=512, gamma=1):
        self.gamma = gamma
        self.D = D
        self.W = ptu.from_numpy(np.random.randn(input_dim, D))
        self.b = ptu.from_numpy(np.random.uniform(size=(1, D), low=0, high=2 * np.pi))

    def _forward(self, x, y):
        psi_x = np.sqrt(2.0 / self.D) * torch.cos(np.sqrt(2 / self.gamma) *
                                                  torch.matmul(x, self.W) + self.b)
        psi_y = np.sqrt(2.0 / self.D) * torch.cos(np.sqrt(2 / self.gamma) *
                                                  torch.matmul(y, self.W) + self.b)
        MMD = torch.mean(psi_x, dim=0) - torch.mean(psi_y, dim=0)

        return MMD.norm(2)

    def forward(self, latents):
        loss = 0.
        n = len(latents)
        for i in range(n):
            x, y = latents[i], latents[(i+1) % n]
            loss += self._forward(x, y)
        return loss / n


