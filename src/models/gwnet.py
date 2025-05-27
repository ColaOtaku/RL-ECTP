import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel


class GWNET_(BaseModel):
    '''
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    '''
    def __init__(self, adp_adj, dropout, residual_channels, dilation_channels, \
                 skip_channels, end_channels, kernel_size=2, blocks=4, layers=2, use_bn=False, use_mask=True, **args):
        super(GWNET_, self).__init__(**args)
        self.use_bn = use_bn
        self.use_mask = use_mask
        self.supports_len = 1 # if use normlap
        self.adp_adj = adp_adj
        if adp_adj:
            self.nodevec1 = nn.Parameter(torch.randn(self.node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.node_num), requires_grad=True)
            self.supports_len += 1

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        if self.use_bn:
            self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1,kernel_size), dilation=new_dilation))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1,1)))
                if self.use_bn:
                    self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(dilation_channels, residual_channels, self.dropout, support_len=self.supports_len))
        self.receptive_field = receptive_field
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.output_dim * self.horizon,
                                    kernel_size=(1,1),
                                    bias=True)

    def forward(self, input, supports, node_idx, mask=None, label=None):  # (b, t, n, f)
        if self.use_mask:
            assert mask is not None
        input = input.transpose(1,3)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        supports = supports if isinstance(supports, list) else [supports]
        if self.adp_adj:
            if self.use_mask:
                if isinstance(mask, torch.Tensor):
                    add_mask = torch.zeros_like(mask, dtype=torch.float)
                    add_mask.masked_fill_(mask, float('-inf'))
                else:
                    add_mask = 0
                adp = F.softmax(add_mask + F.relu(torch.mm(self.nodevec1[node_idx,:], self.nodevec2[:,node_idx])), dim=1)
            else:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1[node_idx, :], self.nodevec2[:, node_idx])), dim=1)
            new_supports = supports + [adp]
        else:
            new_supports = supports

        x = self.start_conv(x)

        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:         
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            
            x = x + residual[:, :, :, -x.size(3):]
            if self.use_bn:
                x = self.bn[i](x)
        
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)


    def forward(self,x):
        return self.mlp(x)

    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order


    def forward(self, x, support):
        out = [x]
        for a in support:
            a = a.float()
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h