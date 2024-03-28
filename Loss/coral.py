import torch


# classical coral implementation
def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


# coral difference method
def coral_1(source,target):
    m,n = source.size()[0],target.size()[0]
    eps = 1e-8
    d = source.size()[1]
    ns, nt = source.size()[0], target.size()[0]

    # source covariance
    tmp_s = torch.matmul(torch.ones(size=(1, ns)).cuda(), source)
    cs = (torch.matmul(torch.t(source), source) - torch.matmul(torch.t(tmp_s), tmp_s) / (ns + eps)) / (ns - 1 + eps) * (
            ns / m)

    # target covariance
    tmp_t = torch.matmul(torch.ones(size=(1, nt)).cuda(), target)
    ct = (torch.matmul(torch.t(target), target) - torch.matmul(torch.t(tmp_t), tmp_t) / (nt + eps)) / (
            nt - 1 + eps) * (
                 nt / n)
    # frobenius norm
    # loss = torch.sqrt(torch.sum(torch.pow((cs - ct), 2)))
    loss = torch.norm((cs - ct))
    loss = loss / (4 * d * d)

    return loss


