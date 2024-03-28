import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

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

    def convert_to_onehot(self, sca_label, class_num=7):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=64, class_num=10):
        batch_size = s_label.size()[0]

        # s_sca_label = s_label.cpu().data.numpy()
        # s_vec_label = self.convert_to_onehot(s_sca_label, class_num=class_num)
        # s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        # s_sum[s_sum == 0] = 100
        # s_vec_label = s_vec_label / s_sum
        s_sca_label = s_label.cpu().data.max(1)[1].numpy()
        s_vec_label = s_label.cpu().data.numpy()
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def rbf_cwmmd(self, sX, tX, sY, tY):
        '''
        Return CWMMD score based on guassian kernel.
        '''
        n_sample1 = sX.size(0)
        n_sample2 = tX.size(0)
        device = sX.device
        batch_size = sX.size(0)
        xkernels = self.guassian_kernel(sX, tX, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)
        ykernels = self.guassian_kernel(sY, tY, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)
        weight_ss, weight_tt, weight_st = self.cal_weight(
            sY, tY, batch_size=batch_size, class_num=10)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

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

        # loss = cmmd1 + cmmd2 + cmmd3
        loss = torch.sum(weight_ss * cmmd2 + weight_tt * cmmd3 + weight_st * cmmd1)
        # loss = torch.mean(loss)
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
        elif self.kernel_type== 'cwmmd':
            return self.rbf_cwmmd(source,target,sourceY,targetY)
