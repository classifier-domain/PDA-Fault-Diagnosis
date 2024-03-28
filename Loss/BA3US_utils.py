from utils.UTILS import *


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*((iter_num-1) / (max_iter-1)))) - (high - low) + low)


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def DANN(features, ad_net, entropy=None, coeff=None, cls_weight=None, len_share=0,grl=None):
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                        max_iters=1000, auto_step=True) if grl is None else grl
    features = grl(features)
    ad_out = ad_net(features)
    train_bs = (ad_out.size(0) - len_share) // 2
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * (train_bs + len_share))).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
    else:
        entropy = torch.ones(ad_out.size(0)).cuda()

    source_mask = torch.ones_like(entropy)
    source_mask[train_bs: 2 * train_bs] = 0
    source_weight = entropy * source_mask
    source_weight = source_weight * cls_weight

    target_mask = torch.ones_like(entropy)
    target_mask[0: train_bs] = 0
    target_mask[2 * train_bs::] = 0
    target_weight = entropy * target_mask
    target_weight = target_weight * cls_weight

    weight = (1.0 + len_share / train_bs) * source_weight / (torch.sum(source_weight).detach().item()) + \
             target_weight / torch.sum(target_weight).detach().item()

    weight = weight.view(-1, 1)
    return torch.sum(weight * nn.BCELoss(reduction='none')(ad_out, dc_target)) / (
                1e-8 + torch.sum(weight).detach().item())


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