import torch
import torch.nn as nn
import torch.nn.functional as F

# 残差层
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 特征提取器
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=64, stride=1, padding=25),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.residualBlock = nn.Sequential(Residual(64, 128, use_1x1conv=True),
                                           Residual(128, 256, use_1x1conv=True),
                                           nn.AdaptiveAvgPool1d((1)),
                                           nn.Flatten())

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.residualBlock(out)
        out = out.view(out.shape[0], -1)
        self.featuremap = out
        return out

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64,10),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.shape[0],-1)
        # x = F.softmax(x, dim=1)
        self.featuremap = x
        return x

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64,1),
            # nn.BatchNorm1d(1),
            # nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.flatten(x, start_dim=1)
        self.featuremap = x
        return x

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=51, stride=1, padding=25),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.residualBlock = nn.Sequential(Residual(64, 128, use_1x1conv=True),
                                           Residual(128, 256, use_1x1conv=True),
                                           nn.AdaptiveAvgPool1d((1)),
                                           nn.Flatten())
        self.dpout = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 10),
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.residualBlock(out)
        out = out.view(out.shape[0], -1)
        x = self.fc1(out)
        x = F.softmax(x, dim=1)
        self.featuremap = x

        return x

class GF(nn.Module):
    def __init__(self):
        super(GF, self).__init__()
        self.fc3 = nn.Sequential(
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.dpout = nn.Dropout(0.5)


    def forward(self,x):
        out = self.fc3(x)
        out = out.view(out.shape[0], -1)
        return out
