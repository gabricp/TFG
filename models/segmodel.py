from torch import nn
from torch.nn import functional as F

class SegModel(nn.Module):
    def __init__(self,input_dim, hidden_channels, n_convs):
        super(SegModel, self).__init__()
        self.n_convs = n_convs

        self.conv1 = nn.Conv2d(input_dim, hidden_channels, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(n_convs-1):
            self.conv2.append( nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(hidden_channels) )
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(hidden_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.n_convs-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x