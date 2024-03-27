import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from torch.autograd import Function
import scipy


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)




def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1



def get_accuracy(preds, targets):
    assert preds.shape[0] == targets.shape[0]
    correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
    accuracy = correct / preds.shape[0]

    return accuracy

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*((iter_num-1) / (max_iter-1)))) - (high - low) + low)

def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1. / len(a))


def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True


def get_next_batch(loaders, iters, src, device):
    inputs, labels = None, None
    if type(src) == list:
        for key in src:
            try:
                inputs, labels = next(iters[key])
                break
            except StopIteration:
                continue
        if inputs == None:
            for key in src:
                iters[key] = iter(loaders[key])
            inputs, labels = next(iters[src[0]])
    else:
        try:
            inputs, labels = next(iters[src])
        except StopIteration:
            iters[src] = iter(loaders[src])
            inputs, labels = next(iters[src])

    return inputs.to(device), labels.to(device)


def get_gate_label(gate_out, idx, device):
    labels = torch.full(gate_out.size()[:-1], idx, dtype=torch.long)
    labels = labels.to(device)
    return labels


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def mfsan_guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def MFSAN_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, cb=None):
    batch_size = int(source.size()[0])
    kernels = mfsan_guassian_kernel(source, target, kernel_mul=kernel_mul,
                                    kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    if cb != None:
        loss = torch.mean(XX + cb * cb.T * YY - cb * XY - cb.T * YX)
    else:
        loss = torch.mean(XX + YY - XY - YX)

    return loss


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha=1.0, lo=0.0, hi=1.,
                 max_iters=1000., auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()

        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


class ReSize(object):
    def __init__(self, size=1):
        self.size = size
    def __call__(self, seq):
        seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)
        seq = seq / 255
        return seq


def entropy(predictions: torch.Tensor, reduction='mean') -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


def binary_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


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

# 域对抗损失函数
class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, reduction='mean', grl=None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                                 max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    def forward(self, f_s, f_t, w_s=None, w_t=None):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)

        domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) \
                            + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        loss = 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + \
                      self.bce(d_t, d_label_t, w_t.view_as(d_t)))
        return loss

from torch.utils.data import DataLoader, Dataset

def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to
    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1-torch.sum(target[i])/target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]))
            y = torch.max(torch.log(1-input[i]), torch.tensor([-100.0]))
            l = -(beta*target[i] * x + (1-beta)*(1 - target[i]) * y)
            loss += torch.sum(l)
        return loss

def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class CB_Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
        labels_one_hot=None,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(CB_Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced
        self.labels_one_hot = labels_one_hot

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        if self.labels_one_hot is None:
            labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()


            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

from typing import Optional
import torch
import torch.nn as nn

class BatchSpectralPenalizationLoss(nn.Module):
    r"""Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
    Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
    <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>`_.
    Given source features :math:`f_s` and target features :math:`f_t` in current mini batch, singular value
    decomposition is first performed
    .. math::
        f_s = U_s\Sigma_sV_s^T
    .. math::
        f_t = U_t\Sigma_tV_t^T
    Then batch spectral penalization loss is calculated as
    .. math::
        loss=\sum_{i=1}^k(\sigma_{s,i}^2+\sigma_{t,i}^2)
    where :math:`\sigma_{s,i},\sigma_{t,i}` refer to the :math:`i-th` largest singular value of source features
    and target features respectively. We empirically set :math:`k=1`.
    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.
    """

    def __init__(self):
        super(BatchSpectralPenalizationLoss, self).__init__()

    def forward(self, f_s, f_t):
        _, s_s, _ = torch.svd(f_s)
        _, s_t, _ = torch.svd(f_t)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss


def marginloss(yHat, y, classes=10, alpha=1, weight=None):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))  # .detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1) / np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    if weight is not None:
        weight *= (Yg_.view(len(yHat), ) / Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), ) / Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss

class DoaminAdversarial(nn.Module):
    def __init__(self,domain_discriminator,reduction='mean',grl=None):
        super(DoaminAdversarial, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                                 max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    def forward(self,f_s,f_t,f_m,l_s,l_t,l_m,w_s,w_t,w_m,coeff=None,len_share=0):
        train_bs = f_s.size(0)

        f_s,f_t,f_m = self.grl(f_s),self.grl(f_t),self.grl(f_m)
        d_s,d_t,d_m = self.domain_discriminator(f_s),self.domain_discriminator(f_t),self.domain_discriminator(f_m)
        d_label_s = torch.ones((f_s.size(0),1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0),1)).to(f_t.device)
        d_label_m = torch.zeros((f_m.size(0),1)).to(f_m.device)

        entropy_source,entropy_target,entropy_middle = Entropy(F.softmax(l_s,dim=1)),Entropy(F.softmax(l_t,dim=1)),Entropy(F.softmax(l_m,dim=1))
        entropy_source.register_hook(grl_hook(coeff))
        entropy_source = 1. + torch.exp(-entropy_source)
        # entropy_target.register_hook(grl_hook(coeff))
        # entropy_target = 1. + torch.exp(-entropy_target)
        entropy_middle.register_hook(grl_hook(coeff))
        entropy_middle = 1. + torch.exp(-entropy_middle)

        w_s = w_s * entropy_source
        w_m = w_m * entropy_middle
        a = self.bce(d_s,d_label_s,w_s.view_as(d_s).detach())

        loss = 0.5 * (self.bce(d_s,d_label_s,w_s.view_as(d_s).detach()) + self.bce(d_t,d_label_t,w_t.view_as(d_t).detach()))
        loss = loss + (len_share / train_bs) * self.bce(d_m,d_label_m,w_m.view_as(d_m).detach()) * 0.5

        return loss


