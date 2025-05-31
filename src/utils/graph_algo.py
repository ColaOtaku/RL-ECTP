import torch
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import metis
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque
from src.utils.metrics import compute_all_metrics

def normalize_adj_mx(adj_mx, adj_type, return_type='dense'):
    if adj_type == 'normlap':
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == 'symadj':
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'hyper':
        adj = [hyper_norm(adj_mx)]
    elif adj_type == 'proj':
        adj = [proj_norm(adj_mx)]
        return adj
    else:
        return []

    if return_type == 'dense':
        adj = [a.astype(np.float32).todense() for a in adj]
    elif return_type == 'coo':
        adj = [a.tocoo() for a in adj]
    return adj


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return res


def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adj_mx)
    return res


def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)

def hyper_norm(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res

def proj_norm(adj_mx):
    adj_mx = adj_mx.numpy()
    row_degrees = np.sum(adj_mx, axis=1)
    col_degrees = np.sum(adj_mx, axis=0)

    D_row = np.diag(np.power(row_degrees, -0.5))
    D_col = np.diag(np.power(col_degrees, -0.5))

    normalized_adj_matrix = np.dot(np.dot(D_row, adj_mx), D_col)
    return normalized_adj_matrix

class MultiLevelGraphNode:
    def __init__(self, nodes, adj_matrix, layer=0, dataset='CA', base_model='GWNET'):
        self.nodes = nodes
        self.children = deque()  
        self.layer = layer
        self.id = None
        self.adj_matrix = adj_matrix
        self.hyper_adj_matrix = None # hyper_adj of its children

        self.aug_nodes = None # initialized by augment_graph  
        self.aug_adj_matrix = None # initialized by augment_graph

        self.dataset = dataset
        self.base_model = base_model

    def add_child(self, child):
        self.children.append(child)  

    def init_leave_graph_hyper(self):
        self.hyper_adj_matrix = torch.tensor(self.adj_matrix)

    def make_partition(self):
        degree = self.adj_matrix.sum(axis=0)
        med_degree = np.median(degree)
        # set tree-structure for different datasets
        if self.dataset == 'CA':
            if self.base_model == 'GWNET':
                if self.layer==0:
                    n_part = 4
                elif self.layer==1:
                    n_part = 3
            elif self.base_model == 'STGODE':
                if self.layer == 0:
                    n_part = 2
            elif self.base_model == 'ASTGCN':
                if self.layer == 0:
                    n_part = 4
                elif self.layer == 1:
                    n_part = 3
        elif self.dataset == 'chengdu':
            if self.base_model == 'GWNET':
                if self.layer==0:
                    n_part = 3
                elif self.layer==1:
                    n_part = 3
            elif self.base_model == 'STGODE':
                if self.layer == 0:
                    n_part = 2
                elif self.layer == 1:
                    n_part = 2
            elif self.base_model == 'ASTGCN':
                if self.layer == 0:
                    n_part = 3
                elif self.layer == 1:
                    n_part = 3
        else:
            if self.layer==0:
                n_part = 3
            elif self.layer==1:
                n_part = 2
        _, group_id = metis.part_graph(self.to_list(self.adj_matrix), n_part)
        try:
            max_group = np.max(group_id) + 1
            self.hyper_adj_matrix = torch.zeros([max_group, max_group])
            group_id = np.array(group_id)
            for src in range(max_group):
                for tgt in range(max_group):
                    value = self.adj_matrix[np.ix_(group_id == src, group_id == tgt)].sum()
                    self.hyper_adj_matrix[src, tgt] = value
                    self.hyper_adj_matrix[tgt, src] = value
        except Exception as e:
            print(str(e))

        try:
            for g in range(max_group):
                sub_nodes = self.nodes[np.array(group_id) == g]
                sub_adj = self.adj_matrix[np.ix_(np.array(group_id) == g, np.array(group_id) == g)]
                child = MultiLevelGraphNode(sub_nodes, sub_adj, self.layer + 1, dataset=self.dataset, base_model=self.base_model)
                self.add_child(child)
        except Exception as e:
            print(str(e))
    
    def to_list(self,adjacency_matrix):
        num_nodes = adjacency_matrix.shape[0]
        adj_list = []
        for i in range(num_nodes):
            neighbors = np.nonzero(adjacency_matrix[i])[0].tolist()
            adj_list.append(neighbors)
        return adj_list
    

class MultiLevelGraph:
    def __init__(self, nodes, adj_matrix, adj_matrix_norm, dataset='CA', base_model='GWNET'):
        self.root = MultiLevelGraphNode(nodes, adj_matrix, layer=0, dataset=dataset, base_model=base_model)
        self.root.adj_matrix_norm = adj_matrix_norm
        self.layer_ids = None 
        self.max_layer = None # both initialized after make_multilevel_partition
        self.layer_order_node = None
        self.base_model = base_model

    def augment_graph(self, hop):
        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                node.aug_adj_matrix, node.aug_adj_matrix_norm, node.aug_nodes = self.get_augmented_adj(self.root.adj_matrix, self.root.adj_matrix_norm, node.nodes, hop)
                for child in node.children:
                    queue.append(child)
        
    @staticmethod
    def get_augmented_adj(adj_matrix, adj_matrix_norm, subg_node_list, hop):
        subg_nodes = set(subg_node_list)
        
        def get_hop_neighbors(nodes, hop_count):
            if hop_count == 0:
                return nodes
            
            neighbors = set()
            for node in nodes:
                direct_neighbors = set(np.where(adj_matrix[node] > 0)[0])
                neighbors.update(direct_neighbors)
                
            if hop_count > 1:
                neighbors.update(get_hop_neighbors(neighbors - nodes, hop_count - 1))
                
            return neighbors - nodes  
        
        augmented_nodes = set()
        for h in range(1, hop + 1):
            new_nodes = get_hop_neighbors(subg_nodes, h)
            augmented_nodes.update(new_nodes)
        
        final_nodes = list(subg_nodes) + list(augmented_nodes)
        
        n_total = len(final_nodes)
        augmented_matrix = np.zeros((n_total, n_total))
        augmented_matrix_norm = np.zeros((n_total, n_total))
        
        
        for i, node_i in enumerate(final_nodes):
            for j, node_j in enumerate(final_nodes):
                augmented_matrix[i,j] = adj_matrix[node_i, node_j]
                augmented_matrix_norm[i,j] = adj_matrix_norm[node_i, node_j]
        
        return augmented_matrix, augmented_matrix_norm, final_nodes


    def layer_order_traversal(self):
        result = []
        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                result.append(node)
                for child in node.children:
                    queue.append(child)

        return result


    def get_layer_graph(self, layer_id, if_aug, base_model):

        sub_matrices = []
        sub_matrices_norm = []
        if base_model == 'STGODE':
            sub_matrices_norm_extra = []
        elif base_model == 'ASTGCN':
            sub_matrices_norm0, sub_matrices_norm1, sub_matrices_norm2 = [], [], []
        node_index = []
        if if_aug:
            aug_index = []
            current_index = 0    

        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            
            if node.layer == layer_id:

                if if_aug:
                    indices = node.aug_nodes
                    aug_index.extend(list(range(current_index, current_index+len(node.nodes))))
                    current_index+= len(node.aug_nodes)
                    
                else:
                    indices = node.nodes
                
                if base_model=='GWNET':
                    sub_matrix = torch.tensor(self.root.adj_matrix[np.ix_(indices, indices)])
                    sub_matrix_norm = torch.tensor(self.root.adj_matrix_norm[np.ix_(indices, indices)])
                    sub_matrices.append(sub_matrix)
                    sub_matrices_norm.append(sub_matrix_norm)
                    node_index.extend(indices)
                elif base_model=='STGODE':
                    sub_matrix_norm = torch.tensor(self.root.adj_matrix_norm[0][np.ix_(indices, indices)])
                    sub_matrix_norm_extra = torch.tensor(self.root.adj_matrix_norm[1][np.ix_(indices, indices)])
                    sub_matrices_norm.append(sub_matrix_norm)
                    sub_matrices_norm_extra.append(sub_matrix_norm_extra)
                    node_index.extend(indices)
                elif base_model=='ASTGCN':
                    sub_matrix_norm0 = torch.tensor(self.root.adj_matrix_norm[0][np.ix_(indices, indices)])
                    sub_matrix_norm1 = torch.tensor(self.root.adj_matrix_norm[1][np.ix_(indices, indices)])
                    sub_matrix_norm2 = torch.tensor(self.root.adj_matrix_norm[2][np.ix_(indices, indices)])
                    sub_matrices_norm0.append(sub_matrix_norm0)
                    sub_matrices_norm1.append(sub_matrix_norm1)
                    sub_matrices_norm2.append(sub_matrix_norm2)
                    node_index.extend(indices)
                
            elif node.layer < layer_id:
                queue.extend(node.children)
        
        if base_model=='GWNET':
            return self.block_diag(sub_matrices), self.block_diag(sub_matrices_norm), torch.tensor(node_index, dtype=torch.long), None if (not if_aug) else torch.tensor(aug_index, dtype=torch.long)
        elif base_model=='STGODE':
            return self.block_diag(sub_matrices_norm), self.block_diag(sub_matrices_norm_extra), torch.tensor(node_index, dtype=torch.long), None if (not if_aug) else torch.tensor(aug_index, dtype=torch.long)
        elif base_model=='ASTGCN':
            return self.block_diag(sub_matrices_norm0), self.block_diag(sub_matrices_norm1), self.block_diag(sub_matrices_norm2), torch.tensor(node_index, dtype=torch.long), None if (not if_aug) else torch.tensor(aug_index, dtype=torch.long)

    def get_adjs(self):
        '''
            hyper_adjs: [1 * 1, ..., Na * Na]
            proj_adjs: [1* Ni,... Na * N]
        '''
        
        queue = deque([self.root])
        hyper_adjs, proj_adjs = [], []
        layer, endf  = 0 , False
        temp_hyper_adjs,temp_proj_adjs = [], []
        final_proj_set = [] # for last proj_adj

        first_hyper = torch.tensor([[1]])
        hyper_adjs.append(first_hyper)

        while queue:
            current_node = queue.popleft()

            # all graph within the layer are collected, build block_diag
            if current_node.layer != layer:
                if not endf:
                    hyper_adjs.append(self.block_diag(temp_hyper_adjs)[0])
                proj_adjs.append(self.block_diag(temp_proj_adjs)[0])
                temp_hyper_adjs,temp_proj_adjs = [], []
                layer +=1

            # collect hyper adj of the next layer & proj of the current layer
            if current_node.hyper_adj_matrix is None:
                endf = True
            else:
                temp_hyper_adjs.append(current_node.hyper_adj_matrix)
            
            temp_proj_adjs.append(torch.ones(1,current_node.children.__len__())) # 1*Ni

            for child in current_node.children:
                queue.append(child)

            if current_node.layer == self.max_layer:
                final_proj_set.append(current_node.nodes)

        final_proj = torch.zeros([len(final_proj_set),len(self.root.nodes)])

        for i,nodes in enumerate(final_proj_set):
            final_proj[i,nodes] = 1
        
        proj_adjs.append(final_proj)

        assert len(hyper_adjs) == len(proj_adjs), 'unmatched length'
        for hyper, proj in zip(hyper_adjs, proj_adjs):
            assert hyper.shape[0] == proj.shape[0], 'unmatched dimension, decrease the max_partition_layers'
        return hyper_adjs, proj_adjs
    
    def _build_index(self):
        index = []
        id = 0
        queue = deque([self.root]) 

        while queue:
            current_node = queue.popleft()
            index.append(current_node.layer)  
            current_node.id = id
            id+=1

            for child in current_node.children:
                queue.append(child)
        return index

    def search_subg(self, idx):
        def recursive_search(node, current_index):
            if current_index == idx:
                return node
            
            for child in node.children:
                current_index += 1
                found_node = recursive_search(child, current_index)
                if found_node:
                    return found_node
            
            return None  

        return recursive_search(self.root, current_index=0)

    def display(self):
        queue = deque([self.root]) 
        while queue:
            current_layer_subgraphs = []  
            current_layer = queue[0].layer  

            while queue and queue[0].layer == current_layer:
                current_subgraph = queue.popleft()
                current_layer_subgraphs.append(current_subgraph)

            for index, sub in enumerate(current_layer_subgraphs):
                print(f'Layer {sub.layer}, subgraph {index} with graph id {sub.id} includes {len(sub.nodes)} nodes, {sub.children.__len__()} child graphs')

            for sub in current_layer_subgraphs:
                for child in sub.children:
                    queue.append(child)

    def degrees_hist(self):
        degrees = []
        for node in self.layer_order_node:
            degree = node.adj_matrix.sum(axis=0)
            avg_degree = degree.mean()
            degrees.append(avg_degree)
        plt.hist(degrees)
        plt.show()

    def make_multilevel_partition(self, max_layer):
        self.max_layer = max_layer
        
        def dfs(sub_graph,max_layer):
            if sub_graph.layer >= max_layer or len(sub_graph.adj_matrix) <= 1:
                return
            
            sub_graph.make_partition()
            for child in sub_graph.children:
                dfs(child, max_layer)  
        
        dfs(self.root,max_layer)
        self.layer_ids = torch.tensor(self._build_index()).unsqueeze(0).unsqueeze(-1)
        self.layer_order_node = self.layer_order_traversal()

    def sync_norm_adjs(self):
        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                nodes = node.nodes
                if level_size != 1:
                    if self.base_model == 'STGODE':
                        node.adj_matrix_norm = []
                        node.adj_matrix_norm.append(self.root.adj_matrix_norm[0][np.ix_(nodes, nodes)])
                        node.adj_matrix_norm.append(self.root.adj_matrix_norm[1][np.ix_(nodes, nodes)])
                    elif self.base_model == 'GWNET':
                        node.adj_matrix_norm = self.root.adj_matrix_norm[np.ix_(nodes, nodes)]
                    elif self.base_model == 'ASTGCN':
                        node.adj_matrix_norm = []
                        node.adj_matrix_norm.append(self.root.adj_matrix_norm[0][np.ix_(nodes, nodes)])
                        node.adj_matrix_norm.append(self.root.adj_matrix_norm[1][np.ix_(nodes, nodes)])
                        node.adj_matrix_norm.append(self.root.adj_matrix_norm[2][np.ix_(nodes, nodes)])
                for child in node.children:
                    queue.append(child)

    def get_children_ids(self, args):
        for node in self.layer_order_node:
            child_ids = []
            for child in node.children:
                child_ids.append(child.id)
            node.child_ids = torch.tensor(child_ids).long().to(args.device)

    # to save pred time
    def convert_to_tensor(self, args):
        for subg in self.layer_order_node:
            subg.nodes = torch.from_numpy(subg.nodes).long().to(args.device)
            if args.base_model == 'STGODE':
                subg.adj_matrix_norm[0] = torch.from_numpy(subg.adj_matrix_norm[0]).to(args.device)
                subg.adj_matrix_norm[1] = torch.from_numpy(subg.adj_matrix_norm[1]).to(args.device)
            elif args.base_model == 'GWNET':
                subg.adj_matrix_norm = torch.from_numpy(subg.adj_matrix_norm).to(args.device)
            elif args.base_model == 'ASTGCN':
                subg.adj_matrix_norm[0] = torch.from_numpy(subg.adj_matrix_norm[0]).to(args.device)
                subg.adj_matrix_norm[1] = torch.from_numpy(subg.adj_matrix_norm[1]).to(args.device)
                subg.adj_matrix_norm[2] = torch.from_numpy(subg.adj_matrix_norm[2]).to(args.device)

    def cal_subg_time_cost(self, predictors, dataloader, args):
        assert len(predictors) == (self.max_layer+1)

        time_table = []
        all_nodes, layer_ids = np.array(self.layer_order_node), np.array(self.layer_ids[0,:,0])
        for l in range(self.max_layer+1):
            predictor = predictors[l]
            predictor.to(args.device)
            predictor.eval()

            current_layer_nodes = all_nodes[layer_ids == l]

            print('len current layer nodes {}'.format(len(current_layer_nodes)))
            t1 = time.time()
            with torch.no_grad():
                for node in tqdm(current_layer_nodes):
                    for X, label, *rest in dataloader["test_loader"].get_iterator():
                        X = X.astype(np.float32)
                        X = torch.from_numpy(X).to(args.device)
                        if args.if_aug:
                            adj = torch.from_numpy(node.aug_adj_matrix_norm).to(args.device)
                            nodes = torch.from_numpy(node.aug_nodes).to(args.device)
                        else:
                            adj = node.adj_matrix_norm
                            nodes = node.nodes
                        _ = predictor(X[:,:,nodes], adj, nodes, 1.0)
            t2 = time.time()

            time_table.append((t2-t1)/len(current_layer_nodes))

        assert len(time_table) == (self.max_layer+1)
        return time_table

    def cal_partition_time_cost(self, predictors, dataloader, treegraph, engine, args):
        all_nodes, layer_ids = np.array(self.layer_order_node), np.array(self.layer_ids[0,:,0])
        for l in range(self.max_layer+1):
            if l==0:
                continue
            predictor = predictors[l]
            predictor.to(args.device)
            predictor.eval()

            preds = []
            labels = []

            current_layer_nodes = all_nodes[layer_ids == l]

            print('len current layer nodes {}'.format(len(current_layer_nodes)))
            t1 = time.time()
            with torch.no_grad():
                for X, label, *rest in tqdm(dataloader["test_loader"].get_iterator()):
                    X, label = engine._to_device(engine._to_tensor([X, label]))
                    pred = torch.zeros_like(label, device=args.device)
                    if l != 0:
                        chunk_size = engine.env.chunk_sizes[l]
                        for i in range(0, len(current_layer_nodes), chunk_size):
                            if args.base_model == 'GWNET':
                                diag_adjs, diag_masks = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm for j in range(i, i + chunk_size)])
                            elif args.base_model == 'STGODE':
                                diag_adjs_sp, diag_masks_sp = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm[0] for j in range(i, i + chunk_size)])
                                diag_adjs_se, diag_masks_se = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm[1] for j in range(i, i + chunk_size)])
                                diag_adjs = [diag_adjs_sp, diag_adjs_se]
                                diag_masks = None
                            elif args.base_model == 'ASTGCN':
                                diag_adjs0, diag_masks0 = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm[0] for j in range(i, i + chunk_size)])
                                diag_adjs1, diag_masks1 = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm[1] for j in range(i, i + chunk_size)])
                                diag_adjs2, diag_masks2 = treegraph.block_diag([current_layer_nodes[j].adj_matrix_norm[2] for j in range(i, i + chunk_size)])
                                diag_adjs = [diag_adjs0, diag_adjs1, diag_adjs2]
                                diag_masks = diag_masks0
                            chunks = torch.cat([current_layer_nodes[j].nodes for j in range(i, i + chunk_size)])
                            layer_inputs = X[:, :, chunks]
                            pred[..., chunks] = predictor(layer_inputs, diag_adjs, chunks, diag_masks).squeeze(-1)
                    else:
                        for node in current_layer_nodes:
                            nodes = node.nodes
                            if args.base_model == 'GWNET':
                                diag_adjs, diag_masks = treegraph.block_diag([node.adj_matrix_norm])
                            elif args.base_model == 'STGODE':
                                diag_adjs_sp, diag_masks_sp = treegraph.block_diag([node.adj_matrix_norm[0]])
                                diag_adjs_se, diag_masks_se = treegraph.block_diag([node.adj_matrix_norm[1]])
                                diag_adjs = [diag_adjs_sp, diag_adjs_se]
                                diag_masks = None
                            elif args.base_model == 'ASTGCN':
                                diag_adjs0, diag_masks0 = treegraph.block_diag([node.adj_matrix_norm[0]])
                                diag_adjs1, diag_masks1 = treegraph.block_diag([node.adj_matrix_norm[1]])
                                diag_adjs2, diag_masks2 = treegraph.block_diag([node.adj_matrix_norm[2]])
                                diag_adjs = [diag_adjs0, diag_adjs1, diag_adjs2]
                                diag_masks = diag_masks0
                            pred = predictor(X, diag_adjs, nodes, diag_masks).squeeze(-1)
                    pred, label = engine._inverse_transform([pred, label])
                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

            t2 = time.time()

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)

            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()

            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(engine.env.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                engine._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            engine._logger.info(
                f"{engine.COLORS['cyan']} Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}, Inference TIME: {t2 - t1:.3f} {engine.COLORS['reset']}")


    def cal_time_cost(self, predictors, dataloader, args, logger):

        time_table = []
        predictor = predictors[0]
        predictor.to(args.device)
        predictor.eval()
        graph = self.layer_order_node[0]
        n = graph.adj_matrix.shape[-1]
        print('num of nodes: ', n)
        for size in list(range(200, 8600, 200)) + [8600]:
            selected_indices = torch.randperm(n)[:size]
            if args.base_model == 'STGODE':
                sampled_adj = []
                sampled_adj.append(graph.adj_matrix_norm[0][selected_indices][:, selected_indices])
                sampled_adj.append(graph.adj_matrix_norm[1][selected_indices][:, selected_indices])
            elif args.base_model == 'GWNET':
                sampled_adj = graph.adj_matrix_norm[selected_indices][:, selected_indices]
            elif args.base_model == 'ASTGCN':
                sampled_adj = []
                sampled_adj.append(graph.adj_matrix_norm[0][selected_indices][:, selected_indices])
                sampled_adj.append(graph.adj_matrix_norm[1][selected_indices][:, selected_indices])
                sampled_adj.append(graph.adj_matrix_norm[2][selected_indices][:, selected_indices])
            sampled_nodes = graph.nodes[selected_indices]

            t1 = time.time()
            with torch.no_grad():
                for X, label, *rest in dataloader["test_loader"].get_iterator():
                    X = X.astype(np.float32)
                    X = torch.from_numpy(X).to(args.device)
                    _ = predictor(X[:,:,sampled_nodes], sampled_adj, sampled_nodes, 1.0)
            t2 = time.time()

            logger.info('size {}, time {}'.format(size, t2-t1))

            time_table.append((t2-t1))

        return time_table


    @staticmethod
    def block_diag(input):
        tensors =  [item.clone().detach() for item in input]
        masks = [torch.ones_like(tensor) for tensor in tensors]
        return torch.block_diag(*tensors), (1-torch.block_diag(*masks)).bool()
    
    def norm_hyper_adj(adjs):
        return [torch.sqrt(adj) / torch.sqrt(adj).max for adj in adjs]
