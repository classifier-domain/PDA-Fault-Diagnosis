from tqdm import tqdm
import time
import logging
import random
from sklearn.metrics import confusion_matrix
from torch import nn,optim
import argparse
from datetime import datetime
from torchmetrics import MeanMetric, Accuracy
from pytorch_lightning.utilities.seed import seed_everything
from Loss.BA3US_utils import *
from Loss.CWMMD import *
from model.model_cwru import *
from utils.UTILS import *
from utils.t_sne import *
from data_loader.cwru_fft import source_loader, target_loader,target_loader_test,source_dataset
from Loss.weight import CLASSWeightModule
from Loss.LMMD import *



def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--model_name', type=str, default='cwru', help='the name of the Models')
    parser.add_argument('--signal_size', type=int, default=512, help='the signal of the data')
    parser.add_argument('--stride', type=int, default=256, help='')

    parser.add_argument('--max_iterations', type=int, default=61, help="max iterations")
    parser.add_argument('--test_interval', type=int, default=7, help="interval of two continuous test phase")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--source_name', type=str, default='cwru_12_0', help='the name of the source data')
    parser.add_argument('--target_name', type=str, default='cwru_12_1', help='the name of the target data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/',
                        help='the directory to save the Models')
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--nepoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--mu', type=int, default=4, help="init augmentation size = batch_size//mu")
    parser.add_argument('--cot_weight', type=float, default=1.0, choices=[0, 1, 5, 10])
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--class_num', type=int, default=10, help='')
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_aug', type=bool, default=True)
    parser.add_argument('--weight_cls', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='initialization list')
    args = parser.parse_args()
    return args

args = parse_args()
sub_dir = args.model_name+ '_0_1' + '_' + 'To' + '_' + datetime.strftime(datetime.now(), '%m%d-%H：%M：%S')
save_dir = os.path.join(args.checkpoint_dir, args.model_name, sub_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# set the logger
setlogger(os.path.join(save_dir, 'training.log'))
for k, v in args.__dict__.items():
     logging.info("{}: {}".format(k, v))

# ################################################################################################################

def train(feature_extractor,label_predictor,domain_classifier,classifier,grl,source_loader,target_loader,optimizer,current):

    start = time.time()
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()
    classifier.train()
    global  D_M,D_C,MU

    d_m,d_c = 0.,0.
    class_weight = None
    train_bs = args.batch_size
    total_epochs = args.max_iterations // args.test_interval   #61/7--8
    dset_loaders = {}
    dset_loaders['source'] = source_loader
    dset_loaders['target'] = target_loader


    iter_source, iter_target = ForeverDataIterator(source_loader), ForeverDataIterator(target_loader)
    num_iter = len(source_loader)
    if D_M == 0 and D_C == 0 and MU == 0:
        MU = 0.5
    else:
        D_M = D_M / num_iter
        D_C = D_C / num_iter
        MU =  D_M / (D_M + D_C)
    logging.info('MU is :{:.6f}'.format(MU))

    for i in tqdm(range(0, num_iter)):
        loss_s, loss_t, loss_st = 0.0, 0.0, 0.
        lamb = calc_coeff(current, 1, 0, 10, args.nepoch)
        total_hit_S, total_num_S = 0.0, 0.0
        tmpd_c = 0


        if i % args.test_interval == 0:
            if args.mu > 0:
                epoch = i // args.test_interval
                len_share = int(max(2, (train_bs // args.mu) * (1 - epoch / total_epochs)))
            elif args.mu == 0:
                len_share = 0  # no augmentation
            else:
                len_share = int(train_bs // abs(args.mu))

            dset_loaders["middle"] = None
            if not len_share == 0:
                dset_loaders["middle"] = DataLoader(source_dataset, batch_size=len_share, shuffle=True, drop_last=True)
                iter_middle = iter(dset_loaders["middle"])

        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data = source_data.type(torch.FloatTensor).cuda()
        target_data = target_data.type(torch.FloatTensor).cuda()
        source_label = source_label.long().cuda()

        optimizer.zero_grad()

        if class_weight is not None:
            loss ,loss_cls = torch.tensor(0.0),torch.tensor(0.0)

        s_out, t_out, m_out = [], [], []
        c_loss = 0.0
        coral_loss = 0.
        # #############################################################
        # 第一个特征提取器
        f_s, f_t = feature_extractor(source_data), feature_extractor(target_data)

        if dset_loaders["middle"] is not None:
            source_middle, middle_lable = next(iter_middle)
            inputs_middle = source_middle.type(torch.FloatTensor).cuda()
            middle_lable = middle_lable.long().cuda()
            f_m = feature_extractor(inputs_middle)
            outputs = torch.cat((f_s,f_t,f_m),dim=0)
        else:
            outputs = torch.cat((f_s, f_t), dim=0)

        # ##################################################
        # “
        # ajda 第一个对抗模块，固定训练第一个域鉴别器
        # ”
        w_s, w_t = class_weight_module_source.get_class_weight_for_adversarial_loss(source_label)
        w_m,_ = class_weight_module_source.get_class_weight_for_adversarial_loss(middle_lable)
        class_weight = [w_s,w_t,w_m]


        cls_weight = torch.ones(outputs.size(0)).cuda()
        if class_weight is not None and args.weight_aug:
            cls_weight[0:train_bs] = w_s
            # cls_weight[train_bs:2*train_bs] = w_t
            if dset_loaders["middle"] is not None:
                cls_weight[2 * train_bs::] = w_m


        f_grl_s,f_gel_t,f_grl_m = grl(f_s),grl(f_t),grl(f_m)
        dg_s,dg_t,dg_m = domain_classifier(f_grl_s),domain_classifier(f_gel_t),domain_classifier(f_grl_m)
        wsks = entropy(F.softmax(dg_s,dim=1),reduction=None)
        wsks.register_hook(grl_hook(lamb))
        wsks = 1. + torch.exp(-wsks)
        wsks = (wsks / torch.mean(wsks)).detach()

        wskt = entropy(F.softmax(dg_t,dim=1),reduction=None)
        wskt.register_hook(grl_hook(lamb))
        wskt = 1. + torch.exp(-wskt)
        wskt = (wskt / torch.mean(wskt)).detach()

        wskm = entropy(F.softmax(dg_m,dim=1),reduction=None)
        wskm.register_hook(grl_hook(lamb))
        wskm = 1. + torch.exp(-wskm)
        wskm = (wskm / torch.mean(wskm)).detach()


        dg1 = torch.cat((dg_s,dg_t,dg_m),dim=0)
        train_b = (dg1.size(0) - len_share) // 2
        d_label_1 = torch.from_numpy(np.array([[1]] * train_b + [[0]] * (train_b + len_share))).float().cuda()
        wsk = entropy(F.softmax(dg1,dim=1),reduction=None)
        wsk.register_hook(grl_hook(lamb))
        wsk = 1.0 + torch.exp(-wsk)

        source_mask_1 = torch.ones_like(wsk)
        source_mask_1[train_b:2 * train_b] = 0
        source_weight_1 = wsk * source_mask_1

        target_mask_1 = torch.ones_like(wsk)
        target_mask_1[0:train_b] = 0
        target_mask_1[2*train_b::] = 0
        target_weight_1 = wsk * target_mask_1

        weight_1 = (1.0 + len_share/train_b)*source_weight_1 / (torch.sum(source_weight_1).detach().item()) +\
                   target_weight_1 / (torch.sum(target_weight_1).detach().item())
        weight_1 = weight_1.view(-1,1)


        loss_doamin_1 = F.binary_cross_entropy(dg1,d_label_1,reduction='none')
        loss_doamin_1 = torch.sum(weight_1 * loss_doamin_1) / (1e-10 + torch.sum(weight_1).detach().item())


        f_ws = wsks.view(-1,1) * f_s
        f_wt = wskt.view(-1,1) * f_t
        f_wm = wskm.view(-1,1) * f_m


        f_s1, f_t1 ,f_m1= gf(f_ws), gf(f_wt), gf(f_wm)

        # ###################################################

        l_s1, l_t1 ,l_m1= label_predictor(f_s1), label_predictor(f_t1),label_predictor(f_m1)

        d_s1, d_t1 ,d_m1= domain_classifier(grl(f_s1)), domain_classifier(grl(f_t1)),domain_classifier(grl(f_m1))

        # #####################################################

        for k in range(10):
            ps = l_s1[:, k].reshape((f_s1.shape[0], 1))
            fs = ps * grl(f_s1)
            pt = l_t1[:, k].reshape((f_t1.shape[0], 1))
            ft = pt * grl(f_t1)
            pm = l_m1[:, k].reshape((f_m1.shape[0], 1))
            fm = pm * grl(f_m1)
            outsi = domain_classifier(fs)
            s_out.append(outsi)
            outti = domain_classifier(ft)
            t_out.append(outti)
            outmi = domain_classifier(fm)
            m_out.append(outmi)

        # ##############################################################

        l_outputs = torch.cat((l_s1,l_t1,l_m1),dim=0)
        ml = entropy(F.softmax(l_outputs,dim=1),reduction=None)
        ml.register_hook(grl_hook(lamb))
        ml = 1.0 + torch.exp(-ml)

        source_mask = torch.ones_like(ml)
        source_mask[train_b: 2 * train_b] = 0
        source_weight = ml * source_mask
        msk = source_weight * cls_weight


        target_mask = torch.ones_like(ml)
        target_mask[0: train_b] = 0
        target_mask[2 * train_b::] = 0
        target_weight = ml * target_mask
        mtk = target_weight * cls_weight

        d_label = torch.from_numpy(np.array([[1]] * f_s1.size(0) + [[0]] * (f_t1.size(0)+f_m1.size(0)))).float().cuda()
        ad_out = torch.cat((d_s1,d_t1,d_m1),dim=0)

        mk = (1.0 + len_share / train_b) * msk / (torch.sum(msk).detach().item()) + mtk / (torch.sum(mtk).detach().item())

        mk = mk.view(-1,1)
        global_loss_ = F.binary_cross_entropy(ad_out,d_label,reduction='none')
        global_loss = torch.sum(mk * global_loss_) / (1e-10 + torch.sum(mk).detach().item())
        global_loss = global_loss * 0.5

        for j in range(10):
            outputs_lodal = torch.cat((s_out[j],t_out[j],m_out[j]),dim=0)
            loss_i = F.binary_cross_entropy(outputs_lodal,d_label,reduction='none')
            loss_i = torch.sum(mk * loss_i) / (1e-10 + torch.sum(mk).detach().item())
            loss_st += loss_i

        local_loss = loss_st * 0.05
        join_loss = global_loss + local_loss

        # #############################################################
        pre_pseudo_label = torch.argmax(l_t1,dim=-1)

        one_hot_label = torch.zeros(l_s1.size(0),l_s1.size(1)).scatter_(1,source_label.view(batch_size,1).data.cpu(),1).cuda()
        one_hot_label_t = torch.ones(l_t1.size(0),l_s1.size(1)).scatter_(1,pre_pseudo_label.view(batch_size,1).data.cpu(),0).cuda()
        MMDloss = MMD(f_s1,f_t1,one_hot_label,one_hot_label_t)

        distance_loss = mmd.get_loss(f_s1, f_t1,source_label,l_t1)

        JDA_loss = MMDloss * 400 + distance_loss * 0.05

        d_m += torch.exp(2 * (1 - 2 * JDA_loss.cpu())).item()
        d_c += torch.exp(2 * (1 - 2 * join_loss.cpu())).item()

        transfer_loss = MU * join_loss * lamb + JDA_loss * (1 + MU)

        ajda_loss = loss_doamin_1

        cot_loss = marginloss(l_s1,source_label,10,alpha=1,weight=w_s.detach())

        loss_cls = CB_loss(l_s1,source_label)
        weight = class_weight_module_source.get_class_weight_for_cross_entropy_loss()
        loss_cls = torch.sum(weight * loss_cls) / (1e-8 + torch.sum(weight).item())

        l_tm1 = torch.cat((l_t1,l_m1),dim=0)
        entropy_loss = entropy(F.softmax(l_tm1,dim=1),reduction='mean')

        loss = loss_cls + ajda_loss * lamb + entropy_loss + transfer_loss + cot_loss * 5
        loss.backward(retain_graph=True)
        optimizer.step()
        class_weight_module_source.step()
        lr_scheduler.step()

        # ##############################################################
        y_s = label_predictor(gf(feature_extractor(source_data)))
        metric_accuracy_1.update(y_s.max(1)[1], source_label)
        metric_mean_1.update(loss)
        metric_mean_2.update(loss_cls)
        metric_mean_3.update(join_loss)

    train_acc = metric_accuracy_1.compute()
    train_all_loss = metric_mean_1.compute()
    train_loss_cls = metric_mean_2.compute()
    j_loss = metric_mean_3.compute()

    metric_accuracy_1.reset()
    metric_mean_1.reset()
    metric_mean_2.reset()
    metric_mean_3.reset()

    D_M = np.copy(d_m).item()
    D_C = np.copy(d_c).item()
    end = time.time()
    TIME = end - start
    source_acc_value.append(train_acc.item())
    source_loss_values.append(train_all_loss.item())


    return train_acc, train_all_loss, train_loss_cls, j_loss, TIME



def test(feature_extractor,label_predictor,gf,target_loader_test):
    feature_extractor.eval()
    label_predictor.eval()
    iter_target = iter(target_loader_test)
    num_iter = len(target_loader_test)
    for i in range(0, num_iter):
        target_data, target_label = next(iter_target)
        target_data = target_data.type(torch.FloatTensor).cuda()
        target_label = target_label.long().cuda()
        output2 = label_predictor(gf(feature_extractor(target_data)))

        metric_accuracy_2.update(output2.max(1)[1], target_label)
        metric_mean_3.update(class_criterion(output2, target_label))
    test_acc = metric_accuracy_2.compute()
    test_loss = metric_mean_3.compute()
    metric_accuracy_2.reset()
    metric_mean_3.reset()
    target_acc_value.append(test_acc.item())
    return test_acc, test_loss


# ################################################################################################
if __name__ == '__main__':
    D_M, D_C, MU = 0, 0, 0,

    seed_everything(42)
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metric_accuracy_1 = Accuracy(task="multiclass", num_classes=10).to(device)
    metric_accuracy_2 = Accuracy(task="multiclass", num_classes=10).to(device)
    metric_mean_1 = MeanMetric().to(device)
    metric_mean_2 = MeanMetric().to(device)
    metric_mean_3 = MeanMetric().to(device)
    metric_mean_4 = MeanMetric().to(device)
    metric_mean_5 = MeanMetric().to(device)

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)
    classifier = classifier().to(device)
    gf = GF().to(device)


    bsp_penalty = BatchSpectralPenalizationLoss().to(device)
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                        max_iters=1000, auto_step=True)

    grls = GradientReverseLayer()
    domain_adv = DoaminAdversarial(domain_classifier, grl=grls)
    class_weight_module_source = CLASSWeightModule(1000, source_loader, classifier, 10, device)

    mmd = LMMD_loss(class_num=10)
    MMD = MMD_loss(kernel_type='cwmmd', eplison=1e-5).cuda()

    train_acc_value = []
    target_acc_value = []
    source_loss_values = []
    domain_loss_values = []
    source_acc_value = []
    losses = []
    class_criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size


    ###############################################
    optimizer = optim.Adam([
        {'params': feature_extractor.parameters()},
        {'params': label_predictor.parameters()},
        {'params': domain_classifier.parameters()},
        {'params': gf.parameters()},
    ], lr=0.01, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.2)
    stop = 0

    samples_per_class = [30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    CB_loss = CB_Loss(
        loss_type="cross_entropy",
        samples_per_class=samples_per_class,
        class_balanced=True,
    )

    for epoch in tqdm(range(0, args.nepoch)):
        stop += 1
        train_acc, train_all_loss, train_loss_cls, j_loss, Time = train(
            feature_extractor, label_predictor, domain_classifier, classifier, grl, source_loader, target_loader,optimizer, current=epoch)

        test_acc, test_loss = test(feature_extractor, label_predictor, gf, target_loader_test)
        logging.info(
            'Epoch{}, train_loss is {:.5f}, test_loss is {:.5f},  train_accuracy is {:.5f}%, test_accuracy is {:.5f}%, time is {:.5f}'.format(
                epoch + 1, train_all_loss, test_loss, train_acc * 100, test_acc * 100, Time))

    Epoch_s = args.nepoch * len(source_loader)
    Epoch_t = args.nepoch * len(target_loader)
    plt.plot(range(args.nepoch), source_acc_value, label='Source ACC')
    plt.plot(range(args.nepoch), target_acc_value, label='Target ACC')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(save_dir, 'Accuracy.png'))
    plt.show()

    result = []
    first = 0
    label_predictor.eval()
    feature_extractor.eval()
    total_hit, total_num = 0.0, 0.0
    for i, (test_data, test_label) in enumerate(target_loader_test):
        test_data = test_data.type(torch.FloatTensor)
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        class_logits = label_predictor(gf(feature_extractor(test_data)))
        if first == 0:
            outputSum = class_logits
            targetSum = test_label
            first += 1
        else:
            outputSum = torch.cat([outputSum, class_logits], dim=0)
            targetSum = torch.cat([targetSum, test_label], dim=0)
            first += 1
        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
        total_num += test_data.shape[0]
    print('target domain test accuracy: {:6.4f}%'.format(total_hit / total_num * 100))
    logging.info('target domain test accuracy: {:6.4f}%'.format(total_hit / total_num * 100))

    outputSum_t = outputSum.cpu().detach().numpy()
    targetSum_t = targetSum.cpu().numpy()


    source_names = ['0', '1', '2', '3', '4', '5', '6','7','8']
    target_names = ['0', '1', '2', '3', '4', '5', '6',]

    month = len(target_names)
    pred_classes = torch.argmax(outputSum, dim=1)
    confusion_mtx = confusion_matrix(targetSum.cpu(), pred_classes.cpu())
    fig, ax = plt.subplots()
    plt.rcParams.update({
        'font.size': 10,  # 设置字体大小
        'font.family': 'Times New Roman',  # 设置字体类型
    })
    plt.rcParams['figure.figsize'] = 12, 10
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600
    plot_confusion_matrix(save_dir, ax, confusion_mtx, classes=target_names,
                          source_classes=source_names, normalize=True,
                          title=str(month) + ' confusion matrix')
    plt.show()


    tsne = TSNE(perplexity=30, n_components=2, init='pca')
    CCC_list ,YYY_list = [],[]
    plt_only = 400
    first = 0

    label_predictor.eval()
    feature_extractor.eval()
    gf.eval()
    source_iter = ForeverDataIterator(source_loader)
    target_iter = ForeverDataIterator(target_loader_test)
    CCC_list = []
    YYY_list = []

    for _ in range(len(target_loader_test)):
        source_data, source_label = next(source_iter)
        target_data, target_label = next(target_iter)

        source_data = source_data.type(torch.FloatTensor).cuda()
        source_label = source_label.long().cuda()
        target_data = target_data.type(torch.FloatTensor).cuda()
        target_label = target_label.long().cuda()

        source_features = gf(feature_extractor(source_data))
        label_predictor(source_features)
        source = label_predictor.featuremap.cpu().detach().numpy()[:plt_only, :]

        target_features = gf(feature_extractor(target_data))
        label_predictor(target_features)
        target = label_predictor.featuremap.cpu().detach().numpy()[:plt_only, :]

        CCC = np.vstack((source, target))
        CCC_list.append(CCC)

        labels_source = source_label.cpu().numpy()[:plt_only]
        labels_target = target_label.cpu().numpy()[:plt_only]
        YYY = np.hstack((labels_source, labels_target + 10))
        YYY_list.append(YYY)

    CCC_list = np.concatenate(CCC_list, axis=0)
    YYY_list = np.concatenate(YYY_list, axis=0)

    low_dim_embs = tsne.fit_transform(CCC_list)
    plot_with_labels_tall(save_dir, low_dim_embs, YYY_list, 'merge')
    plt.show()
