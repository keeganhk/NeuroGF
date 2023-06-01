from .general_pkgs import *



class MLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl):
        super(MLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid']
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, num_points, ic]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [batch_size, ic, num_points, 1]
        y = self.conv(x) # [batch_size, oc, num_points, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() 
        return y # [batch_size, num_points, oc]


class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'tanh', 'sigmoid']
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [batch_size, ic]
        y = self.linear(x) # [batch_size, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y # [batch_size, oc]



