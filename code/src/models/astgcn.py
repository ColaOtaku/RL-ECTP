import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel


class ASTGCN_(BaseModel):
    '''
    Reference code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
    '''

    def __init__(self, device, order, nb_block, nb_chev_filter, nb_time_filter, time_stride, **args):
        super(ASTGCN_, self).__init__(**args)

        self.BlockList = nn.ModuleList([ASTGCN_block_(device, self.input_dim, order, nb_chev_filter, nb_time_filter,
                                                     time_stride, self.node_num, self.seq_len)])
        self.BlockList.extend([ASTGCN_block_(device, nb_time_filter, order, nb_chev_filter, nb_time_filter, 1,
                                                     self.node_num, self.seq_len // time_stride) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(self.seq_len / time_stride), self.horizon, kernel_size=(1, nb_time_filter))

    def forward(self, x, cheb_poly, nodes, mask=None, label=None):  # (b, t, n, f)
        x = x.permute(0, 2, 3, 1)  # (b, n, f, t)

        for block in self.BlockList:
            # (b,n,h,t)
            x = block(x, cheb_poly, nodes, mask)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]  # (b, t, n)
        return output.unsqueeze(-1)


class ASTGCN_block_(nn.Module):
    def __init__(self, device, in_channels, order, nb_chev_filter, nb_time_filter, time_strides,
                 node_num, seq_len):
        super(ASTGCN_block_, self).__init__()
        self.TAt = Temporal_Attention_layer_(device, in_channels, node_num, seq_len)
        self.SAt = Spatial_Attention_layer_(device, in_channels, node_num, seq_len)
        self.cheb_conv_SAt = cheb_conv_withSAt_(device, order, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, cheb_poly, nodes, mask):
        bs, node_num, feature_num, seq_len = x.shape

        # (b, t, t)
        temporal_At = self.TAt(x, cheb_poly, nodes, mask)
        # (b, n, f, t)/(b, n, h, t)
        x_TAt = torch.matmul(x.reshape(bs, -1, seq_len), temporal_At).reshape(bs, node_num, feature_num, seq_len)

        # (b, n, n)
        spatial_At = self.SAt(x_TAt, cheb_poly, nodes, mask)
        # (b, n, h, t)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At, cheb_poly, nodes, mask)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))

        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual


class Temporal_Attention_layer_(nn.Module):
    def __init__(self, device, in_channels, node_num, seq_len):
        super(Temporal_Attention_layer_, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(node_num).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, node_num).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, seq_len, seq_len).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(seq_len, seq_len).to(device))

    def forward(self, x, cheb_poly, nodes, mask):
        _, node_num, feature_num, seq_len = x.shape
        # (b, t, n)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1[nodes]), self.U2[:,nodes])
        # (b, n, t)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class Spatial_Attention_layer_(nn.Module):
    def __init__(self, device, in_channels, node_num, seq_len):
        super(Spatial_Attention_layer_, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(seq_len).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, seq_len).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, node_num, node_num).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(node_num, node_num).to(device))

    def forward(self, x, cheb_poly, nodes, mask):
        # (b, n, t)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (b, t, n)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        # (b, n, n)
        product = torch.matmul(lhs, rhs)

        if isinstance(mask, torch.Tensor):
            product_mask = torch.ones_like(mask, dtype=torch.float)
            product_mask[mask] = 0
            add_mask = torch.zeros_like(mask, dtype=torch.float)
            add_mask.masked_fill_(mask, float('-inf'))
        else:
            product_mask = 1.0
            add_mask = 0.0
        # (n, n) * (b, n, n)
        S = torch.matmul(self.Vs[nodes][:, nodes]*product_mask, torch.sigmoid(product + self.bs[:, nodes][:, :, nodes]))
        S_normalized = F.softmax(S+add_mask, dim=1)
        return S_normalized


class cheb_conv_withSAt_(nn.Module):
    def __init__(self, device, order, in_channels, out_channels):
        super(cheb_conv_withSAt_, self).__init__()
        self.device = device
        self.order = order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(order)])

    def forward(self, x, spatial_attention, cheb_poly, nodes, mask):
        bs, node_num, in_channels, seq_len = x.shape
        outputs = []
        for time_step in range(seq_len):
            # (b, n, f)
            graph_signal = x[:, :, :, time_step]

            output = torch.zeros(bs, node_num, self.out_channels).to(self.device)
            for k in range(self.order):
                T_k = cheb_poly[k]
                T_k_with_at = T_k.mul(spatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))