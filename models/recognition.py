import torch.nn as nn
import torch.nn.functional as F

class HTRNet(nn.Module):
    def __init__(self,nclasses, vae=True, head='rnn', flattening='maxpool'):
        super(HTRNet, self).__init__()
        cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
        head_cfg = (256,3)
        
        if vae:
            self.features = VAE_CNN(cnn_cfg, flattening=flattening)
        else:
            self.features = CNN(cnn_cfg, flattening=flattening)

        if flattening=='maxpool':
            hidden = cnn_cfg[-1][-1]
        elif flattening=='concat':
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            print('problem!')
        if head=='rnn':
            self.top = CTCtopR(hidden, head_cfg, nclasses)

    def forward(self, x):
        y = self.features(x)
        y = self.top(y)

        return y
    
class CTCtopR(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super(CTCtopR, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

    def forward(self, x):

        y = x.permute(2, 3, 0, 1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y

class VAE_CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(VAE_CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(4, 64, 3, [1, 1], 1),nn.BatchNorm2d(64),nn.ReLU(),
                                      nn.Conv2d(64, 128, 3, [1, 1], 1),nn.BatchNorm2d(128),nn.ReLU(),
                                      nn.Conv2d(128, 256, 3, [1, 1], 1),nn.BatchNorm2d(256),nn.ReLU()]
                                      )

    def forward(self, x):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y


class CNN(nn.Module):
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()

        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
        #self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1

    def forward(self, x):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        if self.flattening=='maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
        elif self.flattening=='concat':
            y = y.view(y.size(0), -1, 1, y.size(3))

        return y
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out