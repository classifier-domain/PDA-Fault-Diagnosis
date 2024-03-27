import os
from scipy.io import loadmat
import torch
from sklearn.model_selection import train_test_split
from utils.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging



signal_size = 512
batch_size = 128
stride = 256
normlizetype = '-1-1'

np.random.seed(2048)
torch.manual_seed(2048)
torch.cuda.manual_seed_all(2048)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.info(torch.__version__)
logging.info(torch.version.cuda)
logging.info(torch.backends.cudnn.version())
logging.info(torch.cuda.get_device_name(0))

normal = ['97.mat','98.mat','99.mat','100.mat']
# For 12k Drive End Bearing Fault Data
dataname1 = {0:['97.mat',"105.mat", "169.mat", "209.mat", "118.mat", "185.mat", "222.mat", "130.mat", "197.mat","234.mat"],# 1797rpm
            1:['98.mat',"106.mat", "170.mat", "210.mat", "119.mat", "186.mat", "223.mat", "131.mat", "198.mat", "235.mat"] , # 1772rpm
            2:['99.mat',"107.mat", "171.mat", "211.mat", "120.mat", "187.mat", "224.mat", "132.mat", "199.mat","236.mat"] , #1750rpm
            3: ['100.mat',"108.mat", "172.mat", "212.mat", "121.mat", "188.mat", "225.mat", "133.mat", "200.mat","237.mat"] }
# For 12k Fan End Bearing Fault Data
dataname2 ={0: ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat","315.mat"],
           1:["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat","316.mat"] ,
            2:["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat","317.mat"] ,
            3: ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat","318.mat"] } # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname3 ={0: ["109.mat", "122.mat", "135.mat", "173.mat", "189.mat", "201.mat", "212.mat", "250.mat","262.mat"],  # 1797rpm
            1:["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat", "263.mat"] ,
            2:["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat", "264.mat"],
            3:["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat", "265.mat"] } # 1730rpm

dataname_s=["97.mat","105.mat", "169.mat", "209.mat", "118.mat", "185.mat", "222.mat", "130.mat", "197.mat","234.mat"]
dataname_t=["98.mat","106.mat", "170.mat", "210.mat", "119.mat", "186.mat", "223.mat",]



datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]

axis = ["_DE_time", "_FE_time", "_BA_time"]
root = 'D:\\Python\\fault_diagnosis\\DADDN\\data\\cwru'

def data_load(filename, axisname, label,axis):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis
    else:
        realaxis = "X" + datanumber[0] + axis
    fl = loadmat(filename)[realaxis]
    fl = fl.reshape(-1,)
    data = []
    lab = []
    n = (int)((fl.size - signal_size) / stride + 1)
    start, end = 0, 0
    # 贴标签，使用重叠采样，采样数量为 (点数-采样窗口大小)/步长+1
    for i in range(n):
        start = i * stride
        end = signal_size + i * stride
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
    return data, lab


def get_files(dataname,axis,labels):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data,lab = [] ,[]
    for n in tqdm(range(len(dataname))):
        path1 = os.path.join(root, dataname[n])
        data1, lab1 = data_load(path1, dataname[n], label=labels[n],axis=axis)
        data += data1
        lab += lab1
    data = np.array(data)
    lab = np.array(lab)

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Normalize(normlize_type),
            Retype()

        ]),
        'test': Compose([
            # Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


class MyDataset(Dataset):
    def __init__(self, data, label,transform):
        super().__init__()
        self.data = data.reshape(data.shape[0], 1, 256)
        self.label = label
        self.length = data.shape[0]
        self.transforms = transform

    def __getitem__(self, index):
        hdct = self.data[index, :, :]  # 读取每一个npy的数据
        ldct = self.label[index]
        hdct = self.transforms(hdct)
        return hdct, ldct  # 返回数据还有标签

    def __len__(self):
        return self.length  # 返回数据的总个数

labels = [i for i in range(len(dataname_s))]
labelt = [i for i in range(len(dataname_t))]


source_x, source_y = get_files(dataname_s,axis=axis[0],labels=labels)
target_x, target_y = get_files(dataname_t,axis=axis[0],labels=labelt)

target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(
    target_x, target_y,
    test_size=1/4,
    random_state=40,
    shuffle=True)

source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(
    source_x, source_y,
    test_size=1/4,
    random_state=45,
    shuffle=True)

source_dataset = MyDataset(source_x_train,source_y_train,transform=data_transforms("train",normlizetype))
source_loader = DataLoader(source_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

source_dataset_test = MyDataset(source_x_test,source_y_test,transform=data_transforms("test",normlizetype))
source_loader_test = DataLoader(source_dataset_test,batch_size=batch_size,shuffle=False,drop_last=True)

target_dataset = MyDataset(target_x_train,target_y_train,transform=data_transforms("train",normlizetype))
target_loader = DataLoader(target_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

target_dataset_test = MyDataset(target_x_test,target_y_test,transform=data_transforms("test",normlizetype))
target_loader_test = DataLoader(target_dataset_test,batch_size=batch_size,shuffle=False,drop_last=True)



fig = plt.figure(figsize=(20, 10))
plt.xlabel("Time/frame")
plt.ylabel("Amplitude")
plt.subplot(2,2, 1)
plt.plot(np.squeeze(source_x_train[:1,:]))
plt.title("source_train FFT")
plt.subplot(2,2, 2)
plt.plot(np.squeeze(source_x_test[:1,:]))
plt.title("source_test FFT")
plt.subplot(2,2, 3)
plt.plot(np.squeeze(target_x_train[:1,:]))
plt.title("target_train FFT")
plt.subplot(2,2, 4)
plt.plot(np.squeeze(target_x_test[:1,:]))
plt.title("target_test FFT")
# plt.savefig("../picture/cwru_fft.jpg")
plt.show()