import os
from DADDN.utils.sequence_aug import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import itertools

# draw confusion matrix
def plot_confusion_matrix(save_dir,ax, cm, classes, source_classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    # (Blues,cool,Reds,Oranges,Purples,GnBu,YlOrBr) 颜色

    # Remove NaN values from the confusion matrix
    cm = np.nan_to_num(cm)

    # Remove zero rows from the confusion matrix
    cm = cm[~np.all(cm == 0, axis=1)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_title(title,fontsize=15,y=-0.3)
    ax.set_xticks(np.arange(len(source_classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(source_classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.xaxis.set_label_coords(0.5, -0.12)  # x-axis label position
    ax.yaxis.set_label_coords(-0.08, 0.5)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.041,)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    # plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))


def plot_with_labels_tall(save_dir,lowDWeights, labels,text):
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = 10, 8
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['figure.dpi'] = 600

    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    plt.figure()
    type0_x = []
    type0_y = []
    type1_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []
    type11_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type11_y = []
    type12_x = []
    type12_y = []
    type13_x = []
    type13_y = []
    type14_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type14_y = []
    type15_x = []
    type15_y = []
    type16_x = []
    type16_y = []
    type17_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type17_y = []
    type18_x = []
    type18_y = []
    type19_x = []
    type19_y = []


    for i in range(len(labels)):  # 1000组数据，i循环1000次
        if labels[i] == 0:
            type0_x.append(lowDWeights[i][0])
            type0_y.append(lowDWeights[i][1])

        if labels[i] == 1:  # 根据标签进行数据分类,注意标签此时是字符串
            type1_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type1_y.append(lowDWeights[i][1])

        if labels[i] == 2:
            type2_x.append(lowDWeights[i][0])
            type2_y.append(lowDWeights[i][1])

        if labels[i] == 3:
            type3_x.append(lowDWeights[i][0])
            type3_y.append(lowDWeights[i][1])

        if labels[i] == 4:  # 根据标签进行数据分类,注意标签此时是字符串
            type4_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type4_y.append(lowDWeights[i][1])

        if labels[i] == 5:
            type5_x.append(lowDWeights[i][0])
            type5_y.append(lowDWeights[i][1])

        if labels[i] == 6:
            type6_x.append(lowDWeights[i][0])
            type6_y.append(lowDWeights[i][1])

        if labels[i] == 7:  # 根据标签进行数据分类,注意标签此时是字符串
            type7_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type7_y.append(lowDWeights[i][1])

        if labels[i] == 8:
            type8_x.append(lowDWeights[i][0])
            type8_y.append(lowDWeights[i][1])

        if labels[i] == 9:
            type9_x.append(lowDWeights[i][0])
            type9_y.append(lowDWeights[i][1])
        if labels[i] == 10:
            type10_x.append(lowDWeights[i][0])
            type10_y.append(lowDWeights[i][1])

        if labels[i] == 11:  # 根据标签进行数据分类,注意标签此时是字符串
            type11_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type11_y.append(lowDWeights[i][1])

        if labels[i] == 12:
            type12_x.append(lowDWeights[i][0])
            type12_y.append(lowDWeights[i][1])

        if labels[i] == 13:
            type13_x.append(lowDWeights[i][0])
            type13_y.append(lowDWeights[i][1])

        if labels[i] == 14:  # 根据标签进行数据分类,注意标签此时是字符串
            type14_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type14_y.append(lowDWeights[i][1])

        if labels[i] == 15:
            type15_x.append(lowDWeights[i][0])
            type15_y.append(lowDWeights[i][1])

        if labels[i] == 16:
            type16_x.append(lowDWeights[i][0])
            type16_y.append(lowDWeights[i][1])

        if labels[i] == 17:  # 根据标签进行数据分类,注意标签此时是字符串
            type17_x.append(lowDWeights[i][0])  # 取的是样本数据的第一列特征和第二列特征
            type17_y.append(lowDWeights[i][1])

        if labels[i] == 18:
            type18_x.append(lowDWeights[i][0])
            type18_y.append(lowDWeights[i][1])

        if labels[i] == 19:
            type19_x.append(lowDWeights[i][0])
            type19_y.append(lowDWeights[i][1])

    plt.scatter(type0_x, type0_y, s=30, c='r', label='S0')
    plt.scatter(type1_x, type1_y, s=40, c='b', label='S1')
    plt.scatter(type2_x, type2_y, s=50, c='k', label='S2')
    plt.scatter(type3_x, type3_y, s=60, c='g', label='S3')
    plt.scatter(type4_x, type4_y, s=70, c='c', label='S4')
    plt.scatter(type5_x, type5_y, s=80, c='y', label='S5')
    plt.scatter(type6_x, type6_y, s=90, c='darkblue', label='S6')
    plt.scatter(type7_x, type7_y, s=100, c='indigo', label='S7')
    plt.scatter(type8_x, type8_y, s=110, c='orange', label='S8')
    plt.scatter(type9_x, type9_y, s=120, c='lime', label='S9')
    plt.scatter(type10_x, type10_y, s=30, c='r', label='T0',marker='*')
    plt.scatter(type11_x, type11_y, s=40, c='b', label='T1',marker='*')
    plt.scatter(type12_x, type12_y, s=50, c='k', label='T2',marker='*')
    plt.scatter(type13_x, type13_y, s=60, c='g', label='T3',marker='*')
    plt.scatter(type14_x, type14_y, s=70, c='c', label='T4',marker='*')
    plt.scatter(type15_x, type15_y, s=80, c='y', label='T5',marker='*')
    plt.scatter(type16_x, type16_y, s=90, c='darkblue', label='T6',marker='*')
    plt.scatter(type17_x, type17_y, s=100, c='indigo', label='T7',marker='*')
    plt.scatter(type18_x, type18_y, s=110, c='orange', label='T8',marker='*')
    plt.scatter(type19_x, type19_y, s=120, c='lime', label='T9',marker='*')
    plt.legend()
    # plt.xlim(X.min() - 5, X.max() + 5)
    # plt.ylim(Y.min() - 5, Y.max() + 5)
    plt.xticks([])
    plt.yticks([])


    plt.title('Visualized fully connected layer')
    plt.savefig(os.path.join(save_dir, 'TSNE_{}.png'.format(text)))
    plt.show()
    plt.pause(0.01)
    # 绘制所有的图片
