import torch
import torch.nn as nn


class GaussianKernel(nn.Module):

    def __init__(self, sigma=None, track_running_stats=True, alpha=1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X):
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class MMD_loss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5,eplison=0.001, fix_sigma=None, **kwargs):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.eplison = eplison

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def rbf_cmmd(self, sX, tX, sY, tY):
        '''
        Return CMMD score based on guassian kernel.
        '''
        n_sample1 = sX.size(0)
        n_sample2 = tX.size(0)
        device = sX.device
        batch_size = sX.size(0)
        xkernels = self.guassian_kernel(sX, tX, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)
        ykernels = self.guassian_kernel(sY, tY, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)

        X11 = xkernels[:batch_size, :batch_size]
        X21 = xkernels[batch_size:, :batch_size]
        X22 = xkernels[batch_size:, batch_size:]

        Y11 = ykernels[:batch_size, :batch_size]
        Y12 = ykernels[:batch_size, batch_size:]
        Y22 = ykernels[batch_size:, batch_size:]
        X11_inver = torch.inverse(X11 + self.eplison * n_sample1 * torch.eye(n_sample1).to(device))
        X22_inver = torch.inverse(X22 + self.eplison * n_sample2 * torch.eye(n_sample2).to(device))

        cmmd1 = -2.0 / (n_sample1 * n_sample2) * torch.trace(X21.mm(X11_inver).mm(Y12).mm(X22_inver))
        cmmd2 = 1.0 / (n_sample1 * n_sample1) * torch.trace(Y11.mm(X11_inver))
        cmmd3 = 1.0 / (n_sample2 * n_sample2) * torch.trace(Y22.mm(X22_inver))

        loss = cmmd1 + cmmd2 + cmmd3
        loss = torch.mean(loss)
        # return torch.sqrt(loss)
        return loss

    def rbf_mmd(self, source, target):
        '''
        Return MMD score based on guassian kernel.
        '''
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def forward(self, source, target,sourceY=None, targetY=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'mmd':
            return self.rbf_mmd(source,target)
        elif self.kernel_type== 'cmmd':
            return self.rbf_cmmd(source,target,sourceY,targetY)





