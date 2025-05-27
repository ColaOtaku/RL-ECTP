import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from src.base.model import BaseModel


class STGODE_(BaseModel):
    '''
    Reference code: https://github.com/square-coder/STGODE
    '''
    def __init__(self, **args):
        super(STGODE_, self).__init__(**args)
        self.sp_blocks1 = nn.ModuleList(
            [
                STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                            node_num=self.node_num),
                STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                            node_num=self.node_num)
            ])
        self.sp_blocks2 = nn.ModuleList(
            [
                STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                            node_num=self.node_num),
                STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                            node_num=self.node_num)
            ])
        self.sp_blocks3 = nn.ModuleList(
            [
                STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                            node_num=self.node_num),
                STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                            node_num=self.node_num)
            ])

        self.se_blocks1 = nn.ModuleList([
            STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                        node_num=self.node_num),
            STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                        node_num=self.node_num)
        ])
        self.se_blocks2 = nn.ModuleList([
            STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                        node_num=self.node_num),
            STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                        node_num=self.node_num)
        ])
        self.se_blocks3 = nn.ModuleList([
            STGCNBlock_(in_channels=self.input_dim, out_channels=[64, 32, 64],
                        node_num=self.node_num),
            STGCNBlock_(in_channels=64, out_channels=[64, 32, 64],
                        node_num=self.node_num)
        ])

        self.pred = nn.Sequential(
            nn.Linear(self.seq_len * 64, self.horizon * 32),
            nn.ReLU(),
            nn.Linear(self.horizon * 32, self.horizon)
        )


    def forward(self, x, adj, nodes, mask, label=None):  # (b, t, n, f)
        A_sp, A_se = adj[0], adj[1]
        x = x.transpose(1, 2)
        outs = []
        out = self.sp_blocks1[0](x, A_sp, nodes)
        outs.append(self.sp_blocks1[1](out, A_sp, nodes))
        out = self.sp_blocks2[0](x, A_sp, nodes)
        outs.append(self.sp_blocks2[1](out, A_sp, nodes))
        out = self.sp_blocks3[0](x, A_sp, nodes)
        outs.append(self.sp_blocks3[1](out, A_sp, nodes))
        out = self.se_blocks1[0](x, A_se, nodes)
        outs.append(self.se_blocks1[1](out, A_se, nodes))
        out = self.se_blocks2[0](x, A_se, nodes)
        outs.append(self.se_blocks2[1](out, A_se, nodes))
        out = self.se_blocks3[0](x, A_se, nodes)
        outs.append(self.se_blocks3[1](out, A_se, nodes))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x)
        x = x.unsqueeze(-1).transpose(1, 2)
        return x


class STGCNBlock_(nn.Module):
    def __init__(self, in_channels, out_channels, node_num):
        super(STGCNBlock_, self).__init__()
        self.temporal1 = TemporalConvNet_(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.odeg = ODEG_(out_channels[-1], 12, node_num, time=6)
        self.temporal2 = TemporalConvNet_(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels[-1])


    def forward(self, X, adj, nodes):
        t = self.temporal1(X)
        t = self.odeg(t, adj, nodes)
        t = self.temporal2(F.relu(t))
        t = t.permute(0, 3, 1, 2)
        t = self.batch_norm(t)
        t = t.permute(0, 2, 3, 1)
        return t


class TemporalConvNet_(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet_, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d_(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class Chomp1d_(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class ODEG_(nn.Module):
    def __init__(self, feature_dim, temporal_dim, node_num, time):
        super(ODEG_, self).__init__()
        self.odeblock = ODEblock_(ODEFunc_(feature_dim, temporal_dim, node_num), t=torch.tensor([0, time]))


    def forward(self, x, adj, nodes):
        self.odeblock.set_x0(x)
        z = self.odeblock(x, adj, nodes)
        return F.relu(z)


class ODEblock_(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock_, self).__init__()
        self.t = t
        self.odefunc = odefunc


    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()


    def forward(self, x, adj, nodes):
        self.odefunc.set_adj(adj)
        self.odefunc.set_nodes(nodes)
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


class ODEFunc_(nn.Module):
    def __init__(self, feature_dim, temporal_dim, node_num):
        super(ODEFunc_, self).__init__()
        self.node_num = node_num
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(node_num))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)
        self.adj = None
        self.nodes = None

    def set_adj(self, adj):
        self.adj = adj

    def set_nodes(self, nodes):
        self.nodes = nodes

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha[self.nodes]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f