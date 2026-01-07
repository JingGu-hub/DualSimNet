import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, series_length=128, num_classes=100, features=128, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv1d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv1d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv1d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv1d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv1d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv1d(256,512,kernel_size=3,stride=1, padding=1)
        self.c8=nn.Conv1d(512,256,kernel_size=3,stride=1, padding=1)
        self.c9=nn.Conv1d(256,128,kernel_size=3,stride=1, padding=1)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(128)
        self.bn4=nn.BatchNorm1d(256)
        self.bn5=nn.BatchNorm1d(256)
        self.bn6=nn.BatchNorm1d(256)
        self.bn7=nn.BatchNorm1d(512)
        self.bn8=nn.BatchNorm1d(256)
        self.bn9=nn.BatchNorm1d(128)

        self.l_c1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, num_classes)
        )

        self.input_channel = input_channel
        self.series_length = series_length
        self.decoder = nn.Sequential(
            nn.Linear(128, features),
            nn.Linear(features, (input_channel * series_length) // 2),
            nn.Linear((input_channel * series_length) // 2, input_channel * series_length)
        )

    def train_freeze(self):
        self.c1.eval()
        self.c2.eval()
        self.c3.eval()
        self.c4.eval()
        self.c5.eval()
        self.c6.eval()
        self.c7.eval()
        self.c8.eval()
        self.c9.eval()

        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
        self.bn4.eval()
        self.bn5.eval()
        self.bn6.eval()
        self.bn7.eval()
        self.bn8.eval()
        self.bn9.eval()

        self.l_c1.train()

    def forward(self, x, task_type='classification'):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout1d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool1d(h, kernel_size=2, stride=2)
        h=F.dropout1d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool1d(h, kernel_size=h.data.shape[2])

        output = h.view(h.size(0), h.size(1))
        if task_type == 'classification':
            features = output
            classification_output = self.l_c1(output)
            return classification_output, features
        elif task_type == 'restruction':
            features = output
            restruction_outputs = self.decoder(features)
            restruction_outputs = restruction_outputs.reshape(restruction_outputs.size(0), self.input_channel, self.series_length)
            return restruction_outputs, features


