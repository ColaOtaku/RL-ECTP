import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from src.base.model import BaseModel
from collections import namedtuple, deque, defaultdict
import random


class DownSampling(nn.Module):
    def __init__(self, method):
        super(DownSampling, self).__init__()  
        self.method = method

    def forward(self, x, proj_adj):
        '''
            x: b, n, d
            proj_adj: n * m [n>m]
            hyper_adj: m * m
        '''
        if self.method == 'meanpool':
            x = torch.einsum('bnd,nm->bmd',x, proj_adj)/(torch.sum(proj_adj,dim=0).unsqueeze(0).unsqueeze(-1)) #meanpool
        return x


class Upsampling(nn.Module):
    def __init__(self, method, embed_dim, input_size, input_dim=None):
        super(Upsampling, self).__init__()
        self.method = method
        self.fc = nn.Linear(embed_dim*2, embed_dim)
        self.linear_map = nn.Linear(input_dim, embed_dim)

    def forward(self, x, skips, proj_adj):
        '''
            x: b, n, d
            skips: b, m, d
            proj_adj: n * m [n<m] ,tree
        '''
        x = proj_adj.T.unsqueeze(0).matmul(x)

        x = self.linear_map(x)
        x = F.relu(x)

        x = self.fc(torch.cat([x,skips],dim=-1))
        x = F.relu(x)
        return x


class Graph_Unet(nn.Module):
    def __init__(self, hyper_adjs, proj_adjs, input_dim, hidden_dim, num_layers, up_method, down_method, seq_len, device, tree):
        '''
            block_diag
            hyper_adjs: list, [1*1, ... , Na*Na]  # do those belongs to different hyper graphs are connected?
            proj_adjs: list [1*Ni, ... ,Na*N]
        '''
        super(Graph_Unet, self).__init__()
        self.tree = tree
        self.t_enc = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first = True)
        self.reverse_proj_adjs = [adj.T.to(device) for adj in proj_adjs][::-1]
        self.reverse_hyper_adjs = [adj.to(device) for adj in hyper_adjs][::-1]
        self.proj_adjs =  [adj.to(device) for adj in proj_adjs] 

        self.downsampling_layer = nn.ModuleList([DownSampling(method = down_method,
                                                input_dim = hidden_dim,
                                                output_dim = hidden_dim,
                                                input_size = g.shape[0]).to(device) for g in self.reverse_proj_adjs])

        self.readout = nn.Linear(hidden_dim, seq_len)

        self.seq_len = seq_len
        self.device = device

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def downsampling(self, x, h0): 
        '''
            output:
                node_feats: B,N,D
                hn: graph embedding, B,1,D
                skips: multilevel graph embedding, B,N1~Nk-1
                h:memory, n_rnns, N*B, D             
        '''
        b, t, n, d = x.shape
        if h0 is None:
            x = x.permute(0,2,1,3).contiguous().view(b * n, t, d)
            _, h = self.t_enc(x)
        else:
            x = x[:,-1,...].reshape(b*n,1,d) # for incremental update, only update memory with latest observation
            _, h = self.t_enc(x,h0)

        hn = h[-1]
        hn = hn.view(b, n, -1)
        node_feats = hn

        skips = []
        for i, layer in enumerate(self.downsampling_layer):
            hn = layer( hn, self.reverse_proj_adjs[i], self.reverse_hyper_adjs[i])
            skips.append(hn)

        # overall mean
        # skips = []
        # subgs = self.tree.layer_order_traversal()
        # for subg in subgs:
        #     skips.append(torch.mean(node_feats[:, subg.nodes, :], dim=-2, keepdim=True))
        #     # augment for overall mean
        #     # skips.append(torch.mean(node_feats[:, subg.aug_nodes, :], dim=-2, keepdim=True))
        # skips = skips[::-1]

        return node_feats, hn, skips[::-1], h
    
    def upsampling(self, node_feats, hn, skips):
        skips = skips[1:] + [node_feats]

        for i, layer in enumerate(self. upsampling_layer):
            hn = layer(hn, skips[i], self.proj_adjs[i])
        
        return hn
    
    def forward(self, x, h0 = None):
        b, t, n, d = x.shape
        if h0 is None:
            x = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
            _, h = self.t_enc(x)
        else:
            x = x[:, -1, ...].reshape(b * n, 1, d)
            _, h = self.t_enc(x, h0)

        hn = h[-1]
        hn = hn.view(b, n, -1)
        node_feats = hn

        out = self.readout(node_feats)
        return out.transpose(1,2), h

    # get embed for agent using pre-trained encoder
    def get_agent_embed(self, x, h0, method = 'meanpool'):
        node_feats, _, skips, h = self.downsampling(x, h0)
        g_embed = torch.cat(skips,dim =1) # b, \sum N1~NK-1, d
        return g_embed, h  # b, \sum N1~Nk-1, 2*d


class MLP(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 end_dim=1, 
                 use_bn=False, 
                 dropout_rate=0.0,
                 activation=nn.ReLU(),
                 gen_mask=False):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(hidden_dim, hidden_dim//2))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        layers.append(nn.Linear(hidden_dim//2, end_dim))
        if gen_mask:
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class DQN_greedy(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_subgs, n_layers, if_bn, dropout, device):
        super(DQN_greedy, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc_ls = nn.ModuleList([MLP(hidden_dim, end_dim=3, use_bn= if_bn, dropout_rate= dropout) for _ in range(n_layers)])

        self.device = device

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, x, subg_id, layer=False, check=False):
        """
        Args:
            x: Tensor of shape [b, Ng, 2 + 2 * e]
                - First element: availability indicator
                - Second element: layer indicator
                - Next e elements: embedding of the subgraph
                - Last e elements: pooled embedding of the subgraph's children
        Returns:
            Tensor of shape [b, D] where D = action_size
        """
        # for q net updating
        if not layer:
            batch_size = x.shape[0]
            masks, layers, embeds = x[..., 0], x[..., 1], x[..., 2:]

            first_valid_indices = subg_id

            batch_indices = torch.arange(batch_size, device=self.device)
            selected_embeds = embeds[batch_indices, first_valid_indices]
            selected_layers = layers[batch_indices, first_valid_indices].long()

            q_values = torch.zeros(batch_size, 3, device=self.device)
            for layer_idx in range(len(self.fc_ls)):
                mask = (selected_layers == layer_idx)
                if mask.any():
                    q_values[mask] = self.fc_ls[layer_idx](selected_embeds[mask])
        # for fast decision
        else:
            selected_embeds = x[:, subg_id, 2:]
            q_values = self.fc_ls[layer](selected_embeds)

        if not check:
            return q_values
        else:
            mask = torch.ones_like(q_values)
            return q_values, q_values, mask


class Environment:
    def __init__(self, graph, standard_time_cost, skip_time_cost, naive_cost, max_layer, tradeoff, layer_model_time_table, action_size,batch_size, horizon, node_num, modelsets, device, if_aug, thres, unit, chunk_sizes):
        self.multigraph = graph
        self.standard_time_cost = standard_time_cost
        self.skip_time_cost = skip_time_cost
        self.naive_cost = naive_cost
        self.layer_index = graph.layer_ids.to(device)
        self.max_layer = max_layer
        self.device = device
        self.modelsets = modelsets.to(device) # n_layers
        for l in range(max_layer):
            predictor = modelsets[l]
            predictor.eval()
       
        self.batch_size = batch_size
        self.oom_penalty = 1
        self.max_layer_skip_penalty = 0.2
        self.layer_model_time_table = layer_model_time_table # n_layers
        self.tradeoff = tradeoff
        self.action_size= action_size
        self.horizon = horizon
        self.cache = None #b,t,n
        self.graph_embed_cache = None
        self.if_aug = if_aug
        self.last_update_table = torch.zeros(self.max_layer, node_num).to(self.device)  # records total time after last update(action=1)
        self.thres = thres
        self.unit = unit
        self.chunk_sizes = chunk_sizes
        self.temp = []

    def reset(self, embeds, inputs = None):
        # prepare_state
        # embeds: b, \sum N1~Nk-1, d*2
        self.cur_graph_embed_dim = embeds.shape[2]

        if inputs is not None:
            # inputs b, t, n, d
            # initialize with mean
            self.cache = inputs[..., 0].mean(dim=1).unsqueeze(1).unsqueeze(1).expand(-1, self.max_layer, self.horizon, -1)
            self.graph_embed_cache = embeds[:,:,:self.cur_graph_embed_dim].clone()

        assert embeds.shape[0] == 1

        available_bit = torch.zeros_like(self.layer_index).to(embeds.device)
        available_bit[:,0] = 1 # set root graph as available candidate

        init_state = torch.cat([available_bit, self.layer_index, embeds, self.graph_embed_cache], axis = -1) # [b = 1, \sum N1~Nk-1 , 2 + e * 3]

        return init_state #[b = 1, \sum N1~Nk-1, 2+3*e]

    @staticmethod
    def _find_gid(state, subg_id):
        return torch.argmax(state[0,...,0])

    @staticmethod
    def _find_layer(state):
        return state[0,...,1][torch.argmax(state[0,...,0])]

    def _update_table(self, subg):
        '''
        set last_update_table to 0 for the tree node with action=1 and all its subsequent tree nodes
            root: subg with action=1
        '''
        self.last_update_table[subg.layer, subg.nodes] = 0

    def _update_last_time_graph_embed(self, state, subg):
        # only update subg
        self.graph_embed_cache[:, subg.id, :] = state[:, subg.id, 2:2 + self.cur_graph_embed_dim]
     
    def pred(self, sets, inputs, base_model):
        '''
        merge graph of same level before make prediction

            sets [(state, action)]
            action b,
            states b, n, \sum N1~Nk-1, 2+embeddings * 3
        '''
        layer_adj_matrices = defaultdict(list)
        if base_model == 'STGODE':
            layer_adj_matrices_extra = defaultdict(list)
        elif base_model == 'ASTGCN':
            layer_adj_matrices1, layer_adj_matrices2 = defaultdict(list), defaultdict(list)
        layer_nodes = defaultdict(list)
        layer_index_ori = defaultdict(list)

        layer_naive_nodes = defaultdict(list)
        layer_current_cnt = [0]*(self.max_layer+1)

        n = 0
        for (s,a,gid) in sets:
            subg = self.multigraph.layer_order_node[gid]
            if len(subg.children) == 0 and a == 0:
                a = 2
            
            if a == 0:
                continue
            elif a == 1:
                layer = subg.layer

                if self.if_aug:
                    nodes = subg.aug_nodes
                    adj = subg.aug_adj_matrix_norm
                    layer_index_ori[layer].extend(list(range(layer_current_cnt[layer],layer_current_cnt[layer]+len(subg.nodes))))
                    layer_current_cnt[layer] += len(subg.aug_nodes)

                else:
                    nodes = subg.nodes
                    adj = subg.adj_matrix_norm
                n += len(nodes)

                if base_model == 'GWNET':
                    layer_adj_matrices[layer].append(adj)
                elif base_model == 'STGODE':
                    layer_adj_matrices[layer].append(adj[0])
                    layer_adj_matrices_extra[layer].append(adj[1])
                elif base_model == 'ASTGCN':
                    layer_adj_matrices[layer].append(adj[0])
                    layer_adj_matrices1[layer].append(adj[1])
                    layer_adj_matrices2[layer].append(adj[2])
                layer_nodes[layer].append(nodes)
                self._update_last_time_graph_embed(s, subg)
                self._update_table(subg)
            else:
                nodes = subg.nodes
                layer = subg.layer
                layer_naive_nodes[layer].append(nodes)

        preds = torch.zeros_like(self.cache[:,0]).to(self.device)

        # for layer_pred
        # use preds to predictor subgs with action1
        for idx, adjs in layer_adj_matrices.items():
            nodes = layer_nodes[idx]

            if idx == 0 or n<self.thres:
                if base_model == 'GWNET':
                    diag_adjs, diag_masks = self.multigraph.block_diag(adjs) # Ni*Ni
                elif base_model == 'STGODE':
                    diag_adjs_sp, diag_masks_sp = self.multigraph.block_diag(adjs)
                    diag_adjs_se, diag_masks_se = self.multigraph.block_diag(layer_adj_matrices_extra[idx])
                    diag_adjs = [diag_adjs_sp, diag_adjs_se]
                    diag_masks = None
                elif base_model == 'ASTGCN':
                    diag_adjs0, diag_masks0 = self.multigraph.block_diag(adjs)
                    diag_adjs1, diag_masks1 = self.multigraph.block_diag(layer_adj_matrices1[idx])
                    diag_adjs2, diag_masks2 = self.multigraph.block_diag(layer_adj_matrices2[idx])
                    diag_adjs = [diag_adjs0, diag_adjs1, diag_adjs2]
                    diag_masks = diag_masks0
                nodes = torch.cat(nodes)
                layer_inputs = inputs[:,:,nodes] #B,t,Ni,f
                preds[..., nodes] = self.modelsets[idx](layer_inputs, diag_adjs, nodes, diag_masks).squeeze(-1)
            else:
                # chunk_size*subg_size in idx-th layer should be close to self.unit
                chunk_size = self.chunk_sizes[idx]
                for i in range(0, len(adjs), chunk_size):
                    # pred chunk with size chunk_size*subg_size
                    if base_model == 'GWNET':
                        diag_adjs, diag_masks = self.multigraph.block_diag(adjs[i:i+chunk_size])
                    elif base_model == 'STGODE':
                        diag_adjs_sp, diag_masks_sp = self.multigraph.block_diag(adjs[i:i+chunk_size])
                        diag_adjs_se, diag_masks_se = self.multigraph.block_diag(layer_adj_matrices_extra[idx][i:i+chunk_size])
                        diag_adjs = [diag_adjs_sp, diag_adjs_se]
                        diag_masks = None
                    elif base_model == 'ASTGCN':
                        diag_adjs0, diag_masks0 = self.multigraph.block_diag(adjs[i:i + chunk_size])
                        diag_adjs1, diag_masks1 = self.multigraph.block_diag(layer_adj_matrices1[idx][i:i + chunk_size])
                        diag_adjs2, diag_masks2 = self.multigraph.block_diag(layer_adj_matrices2[idx][i:i + chunk_size])
                        diag_adjs = [diag_adjs0, diag_adjs1, diag_adjs2]
                        diag_masks = diag_masks0
                    chunks = torch.cat(nodes[i:i+chunk_size])
                    layer_inputs = inputs[:, :, chunks]
                    preds[..., chunks] = self.modelsets[idx](layer_inputs, diag_adjs, chunks, diag_masks).squeeze(-1)
                nodes = torch.cat(nodes)

            self._update_cache(preds[..., nodes], idx, nodes)

        # take cache for subgs with action2
        for idx in layer_naive_nodes:
            naive_pred_nodes = layer_naive_nodes[idx]
            naive_pred_nodes = torch.cat(naive_pred_nodes)
            preds[..., naive_pred_nodes] = self._naive_pred(idx, naive_pred_nodes)

        return preds

    def _update_cache(self, new_preds, layer, nodes = None):
        self.cache = self.cache.clone()
        if nodes is None:
            self.cache[:, layer] = new_preds
        else:
            self.cache[:, layer, :, nodes] = new_preds

    def _naive_pred(self, layer, nodes):
        return self.cache[:, layer, :, nodes]

    def _check_if_done(self, available_bits):
        # input: b,\sum N1~Nk-1
        return not torch.any(available_bits)

    def eval(self, pred, label, denorm_fun, criterion):
        pred, label = denorm_fun([pred, label])
        error = criterion(pred, label)
        return error

    def pen_func(self, t):
        t = torch.mean(t)
        return 0 if t <= 2 else (t-2)*self.naive_cost

    @torch.no_grad()
    def slim_step(self, state, action, subg_id):
        # for fast state step
        gid = subg_id
        next_state = state
        next_state[0,gid,0] = 0

        if action == 0:
            subg = self.multigraph.layer_order_node[gid]
            if len(subg.children) > 0:
                next_state[0, subg.child_ids, 0] = 1

        done = self._check_if_done(next_state[0, ..., 0])  # b

        return next_state, done

    @torch.no_grad()
    def step(self, state, action, subg_id, raw_input, label, denorm_func, criterion, check=False, cnt=None):
        # states:b = 1, \sum N1~Nk-1, 2 + embedding * 2
        # action: b = 1
        # raw_input: b = 1,t,n,d
        gid = subg_id
        subg = self.multigraph.layer_order_node[gid]
        ratio = len(subg.nodes)/len(self.multigraph.root.nodes)

        next_state = state.clone()

        next_state[0,gid,0] = 0 # set last dim to 0
 
        reward = 0
        # for skip
        if action == 0:
            accuracy_reward = 0
            efficiency_reward = -self.skip_time_cost
            penalty = 0
            if len(subg.children)>0:
                next_state[0, subg.child_ids, 0] = 1
            if subg.layer == self.max_layer-1:
                penalty = -self.max_layer_skip_penalty
            reward += (self.tradeoff * accuracy_reward + efficiency_reward + penalty)

        else:
            if self.if_aug:
                subg_input = raw_input[:,:, subg.aug_nodes]
                subg_adj = torch.tensor(subg.aug_adj_matrix_norm).to(self.device)
                subg_nodes = torch.tensor(subg.aug_nodes)
            else:
                subg_input = raw_input[:,:, subg.nodes]
                subg_adj = subg.adj_matrix_norm
                subg_nodes = subg.nodes
            
            subg_label = label[:,:, subg.nodes]   
            nodes_count = len(subg.nodes)

            if action == 1:         
                with torch.no_grad():
                    try:
                        subg_pred = self._naive_pred(subg.layer, subg.nodes)
                        ref_error = self.eval(subg_pred, subg_label,
                                            denorm_func, criterion)

                        layer_model = self.modelsets[subg.layer]

                        subg_pred = layer_model(subg_input, subg_adj, subg_nodes, 1.0).squeeze(-1)

                        if self.if_aug:
                            subg_pred = subg_pred[:,:,:nodes_count]
                            
                        error = self.eval(subg_pred, subg_label,
                                        denorm_func, criterion)

                        if not check:
                            self._update_cache(subg_pred, subg.layer, subg.nodes)
                            self._update_table(subg)
                            self._update_last_time_graph_embed(state, subg)

                        accuracy_reward = ratio*(ref_error - error)/ref_error
                        accuracy_reward = accuracy_reward.item()
                        efficiency_reward = -(self.skip_time_cost + self.layer_model_time_table[subg.layer])
                        penalty = 0
                        reward += (self.tradeoff * accuracy_reward + efficiency_reward + penalty)
                    except Exception as e:
                        if 'cuda out of memory' in str(e).lower():
                            penalty = -self.oom_penalty
                            reward += penalty
                            print(f"Error : {str(e)}")
                            print(subg.layer, subg_input.shape)
                        else:
                            raise e
            else:   # action = 2
                    accuracy_reward = 0
                    efficiency_reward = -self.skip_time_cost
                    # penalty = -self.pen_func(self.last_update_table[subg.layer, subg.nodes])
                    penalty = 0
                    reward += (self.tradeoff * accuracy_reward + efficiency_reward + penalty)

        done = self._check_if_done(next_state[0,...,0])

        return next_state, reward, done, (action, self.tradeoff*accuracy_reward, efficiency_reward, penalty)

Experience = namedtuple('Experience', ('state', 'action', 'subg_id', 'next_subg_id', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

Experience_enc = namedtuple('Experience', ('input', 'state', 'action', 'subg_id', 'next_subg_id', 'reward', 'next_state', 'done'))

class ReplayBuffer_enc:

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, *args):
        self.memory.append(Experience_enc(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

class Agent:
    def __init__(self, embed_dim, hidden_dim, action_size, num_subgs, max_layer, dropout_rate, if_bn, buffer_size, if_ft, device):
        self.embed_dim = embed_dim 
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size)
        if if_ft:
            self.memory_enc = ReplayBuffer_enc(buffer_size)

        self.q_net = DQN_greedy(embed_dim, hidden_dim, action_size, num_subgs, max_layer,  if_bn, dropout_rate, device)
        self.target_net = DQN_greedy(embed_dim, hidden_dim, action_size, num_subgs, max_layer,  if_bn, dropout_rate, device)

        self.device = device
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.gamma = 0.1

    def memory_step(self, state, action, subg_id, next_subg_id, reward, next_state, done):
        self.memory.add(state, action, subg_id, next_subg_id, reward, next_state, done)

    def memory_enc_step(self, inp, state, action, subg_id, next_subg_id, reward, next_state, done):
        self.memory_enc.add(inp, state, action, subg_id, next_subg_id, reward, next_state, done)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon

    def get_epsilon(self):
        print(self.epsilon)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    @torch.no_grad()  
    def act(self, state, subg_id, layer=False, training=True, check=False):
        # state: b = 1, Na, D
        if (not training) or random.random() > self.epsilon:
            if not check:
                q_values = self.q_net(state, subg_id, layer)
                return torch.argmax(q_values, dim=1)
            else:
                q_values, _values, _mask = self.q_net(state, subg_id, layer, check)
                return torch.argmax(q_values, dim=1), _values, _mask
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if not check:
                return torch.randint(self.action_size,(1,), device=self.device) # b = 1
            else:
                return torch.randint(self.action_size, (1,), device=self.device), torch.tensor([-1,-1,-1]).to(self.device), torch.tensor([-1,-1,-1]).to(self.device)

    @torch.no_grad()
    def act_by_layer(self, state, subg_id, layer=False, training=True, check=False):
        # state: b = 1, Na, D
        if (not training) or random.random() > self.epsilon:
            if not check:
                q_values = self.q_net(state, subg_id, layer)
                return torch.argmax(q_values[:,1:], dim=1)+1
            else:
                q_values, _values, _mask = self.q_net(state, subg_id, layer, check)
                return torch.argmax(q_values[:,1:], dim=1)+1, _values, _mask
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if not check:
                return torch.randint(self.action_size,(1,), device=self.device) # b = 1
            else:
                return torch.randint(self.action_size, (1,), device=self.device), torch.tensor([-1,-1,-1]).to(self.device), torch.tensor([-1,-1,-1]).to(self.device)

