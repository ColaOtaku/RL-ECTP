import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.rl_ectp import Graph_Unet, Agent, Environment
from src.engines.rl_ectp_engine import ENC_Engine,PRED_Engine, AGENT_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_sliding_dataset, load_adj_from_numpy, get_dataset_info,load_dataset
from src.utils.graph_algo import MultiLevelGraph, normalize_adj_mx, calculate_cheb_poly
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger
import torch.nn.functional as F

import torch.nn as nn
from src.models.gwnet import GWNET_
from src.models.stgode import STGODE_
from src.models.astgcn import ASTGCN_
from fastdtw import fastdtw
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_config():
    parser = get_public_config()
    parser.add_argument('--enc_hidden_dim', type=int, default=32)
    parser.add_argument('--enc_gru_nums', type=int, default=3)
    parser.add_argument('--up_method', choices=['copy','degree_weighted','weighted'], default='weighted')
    parser.add_argument('--down_method',choices=['meanpool','maxpool','weighted'], default='meanpool')
    parser.add_argument('--enc_seed', type=int, default=2024)
    parser.add_argument('--if_print', type=bool, default=False)

    parser.add_argument('--base_model', type=str, default='GWNET')
    parser.add_argument('--pred_seed', type=int, default=2024)
    parser.add_argument('--adp_adj', type=int, default=1)
    parser.add_argument('--tpd', type=int, default=96, help='time per day')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--thres', type=float, default=0.6)
    parser.add_argument('--adj_type', type=str, default='normlap')
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--nb_block', type=int, default=2)
    parser.add_argument('--nb_chev_filter', type=int, default=64)
    parser.add_argument('--nb_time_filter', type=int, default=64)
    parser.add_argument('--time_stride', type=int, default=1)
    parser.add_argument('--if_aug', type = bool, default = False)
    parser.add_argument('--aug_hop', type = int, default= 1)
    parser.add_argument('--max_layer', type = int, default= 2)
    parser.add_argument('--learning_rate', type = float, default=0.001)
    parser.add_argument('--dropout_rate', type = float, default= 0.1)

    parser.add_argument('--agent_embed_dim', type = int, default=128) # not used
    parser.add_argument('--agent_hidden_dim', type = int, default=192)
    parser.add_argument('--action_size', type = int, default= 3)
    parser.add_argument('--if_bn', type=bool, default = False)
    parser.add_argument('--buffer_size', type = int , default=1000)
    parser.add_argument('--sample_size', type = int, default= 64, help ='size of memory sampling for each agent training')
    parser.add_argument('--if_ft', type=bool, default=False)

    parser.add_argument('--gamma', type = float, default= 0.9)
    parser.add_argument('--standard_time_cost', type = float, default= 1)
    parser.add_argument('--skip_time_cost', type= float, default=0.01)
    parser.add_argument('--naive_cost', type=float, default=0.01)
    parser.add_argument('--tradeoff', type = float, default= 1)
    parser.add_argument('--layer_model_time_table', type = list, default=[1,1,1,1,1])

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    parser.add_argument('--n_increments', type = int, default=0, help = 'clear memory every n_increments')
    parser.add_argument('--learn_cnts', type = int, default = 20, help = 'train agent every learn_cnts')
    parser.add_argument('--learn_thres', type = int, default = 40, help = 'trigger training after learn_thres')
    parser.add_argument('--learn_times', type =int, default= 5, help = 'train learn_times once training is triggered')
    parser.add_argument('--enc_description', type = str, default='', help='description to specify each run of enc')
    parser.add_argument('--pred_description', type = str, default='', help='description to specify each run of pred')
    parser.add_argument('--agent_description', type = str, default='', help='description to specify each run of agent')

    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    if args.enc_description or args.pred_description or args.agent_description:
        logger = get_logger(log_dir, __name__, 'record_s{}_E{}_P{}_A{}.log'.format(args.seed, args.enc_description,
                                                                                   args.pred_description, args.agent_description))
    else:
        logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def Predictor_Constructor(base_model, node_num, args):
    if base_model == 'GWNET':
        modelsets = nn.ModuleList([
            GWNET_(
                    node_num=node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    adp_adj=args.adp_adj,
                    dropout=args.dropout_rate,
                    residual_channels=32,
                    dilation_channels=32,
                    skip_channels=256,
                    end_channels=512,
            ) for _ in range(args.max_layer+1)
        ])
    elif base_model == 'STGODE':
        modelsets = nn.ModuleList([
            STGODE_(
                    node_num=node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
            ) for _ in range(args.max_layer+1)
        ])
    elif base_model == 'ASTGCN':
        modelsets = nn.ModuleList([
            ASTGCN_(
                    node_num=node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    device=args.device,
                    order=args.order,
                    nb_block=args.nb_block,
                    nb_chev_filter=args.nb_chev_filter,
                    nb_time_filter=args.nb_time_filter,
                    time_stride=args.time_stride
            ).cpu() for _ in range(args.max_layer+1)
        ])
    else:
        raise "NotImplement"
    return modelsets


def normalize_adj_mx_stgode(adj_mx):
    alpha = 0.8
    D = np.array(np.sum(adj_mx, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), adj_mx),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(adj_mx.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def construct_se_matrix(data_path, args):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    data = ptr['data'][..., 0]
    sample_num, node_num = data.shape

    data_mean = np.mean([data[args.tpd * i: args.tpd * (i + 1)] for i in range(sample_num // args.tpd)], axis=0)
    data_mean = data_mean.T

    dist_matrix = np.zeros((node_num, node_num))
    for i in tqdm(range(node_num)):
        for j in range(i, node_num):
            dist_matrix[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]

    for i in tqdm(range(node_num)):
        for j in range(i):
            dist_matrix[i][j] = dist_matrix[j][i]

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    dist_matrix = np.exp(-dist_matrix ** 2 / args.sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres] = 1
    return dtw_matrix

def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    adj = load_adj_from_numpy(adj_path)
    if args.dataset != 'chengdu':
        adj = (adj + adj.T)/2

    # hard filtering
    filter_adj_matrix = adj.copy()
    filter_adj_matrix[adj>0.45] = 1
    filter_adj_matrix[adj<=0.45] = 0

    # normalize adj
    if args.base_model == 'GWNET':
        adj_norm = normalize_adj_mx(adj, args.adj_type)
        adj_norm = adj_norm[0]
    elif args.base_model == 'STGODE':
        adj = adj - np.eye(node_num)
        sp_matrix = adj + np.transpose(adj)
        sp_matrix = normalize_adj_mx_stgode(sp_matrix).numpy()

        se_matrix_path = os.path.join(data_path, 'se_matrix.npy')
        # uncomment the following 2 lines for the first running
        # se_matrix = construct_se_matrix(data_path, args)
        # np.save(se_matrix_path, se_matrix)
        se_matrix = np.load(se_matrix_path)
        print(se_matrix.shape)
        se_matrix = normalize_adj_mx_stgode(se_matrix).numpy()
        adj_norm = [sp_matrix, se_matrix]
    elif args.base_model == 'ASTGCN':
        adj_mx = adj - np.eye(node_num)
        adj = np.zeros((node_num, node_num), dtype=np.float32)
        for n in range(node_num):
            idx = np.nonzero(adj_mx[n])[0]
            adj[n, idx] = 1
        L_tilde = normalize_adj_mx(adj, 'scalap')[0]
        adj_norm = [torch.from_numpy(i).type(torch.FloatTensor).numpy() for i in calculate_cheb_poly(L_tilde, args.order)]
        print(len(adj_norm))

    treegraph = MultiLevelGraph(np.arange(node_num), filter_adj_matrix, adj_norm, dataset=args.dataset, base_model=args.base_model)
    treegraph.make_multilevel_partition(args.max_layer)
    treegraph.sync_norm_adjs()

    if args.if_print:
        treegraph.display()

    hyper_adjs, proj_adjs = treegraph.get_adjs()
    num_subgs = len(treegraph.layer_order_node)
    for i, (hyper, proj) in enumerate(zip(hyper_adjs, proj_adjs)):
        hyper_adjs[i] = torch.from_numpy(normalize_adj_mx(hyper, 'hyper')[0])
        proj_adjs[i] = torch.from_numpy(normalize_adj_mx(proj, 'proj')[0])
        logger.info('norm hyper_adj and proj_adj')

    enc_model = Graph_Unet(   hyper_adjs = hyper_adjs,
                              proj_adjs = proj_adjs,
                              input_dim = args.input_dim,
                              hidden_dim = args.enc_hidden_dim,
                              num_layers = args.enc_gru_nums,
                              up_method= args.up_method,
                              down_method = args.down_method,
                              seq_len= args.seq_len,
                              device = args.device,
                              tree = treegraph
                              )

    predictors = Predictor_Constructor(
        base_model = args.base_model,
        node_num = node_num, 
        args = args
    )

    #-------------Train Encoder---------------#
    if args.mode == 'train_enc':
        dataloader, scaler = load_dataset(data_path, args, logger)

        loss_fn = masked_mae
        optimizer = torch.optim.Adam(enc_model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = None

        enc_engine = ENC_Engine(device=args.device,
                            model=enc_model,
                            dataloader=dataloader,
                            scaler=scaler,
                            sampler=None,
                            loss_fn=loss_fn,
                            lrate=args.learning_rate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=log_dir,
                            logger=logger,
                            seed=args.seed,
                            des=args.enc_description,
                            n_increments=args.n_increments
        )
        import copy
        initial_params = copy.deepcopy(enc_model.state_dict())
        logger.info('get initial params!')

        enc_engine.train()

        trained_params = enc_model.state_dict()
        for param_tensor in initial_params:
            if not torch.allclose(initial_params[param_tensor], trained_params[param_tensor], atol=1e-5):
                print(f'Parameter {param_tensor} has been updated.')
            else:
                print(f'Parameter {param_tensor} not update!!!')

        enc_engine.evaluate('test')

    #------------Train Predictors--------------#
    elif args.mode == 'train_pred':
        loss_fn = masked_mae
        print('loading data...')
        dataloader, scaler = load_dataset(data_path, args, logger)
        print('loading data done!')
        for i, model in enumerate(predictors):
            if i == 0:
                continue
            if args.base_model == 'GWNET':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
                scheduler = None
            elif args.base_model == 'STGODE':
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0)  # 2e-3  0
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
                args.clip_grad_value = 0
            elif args.base_model == 'ASTGCN':
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
                args.clip_grad_value = 5

            if args.base_model == 'GWNET':
                (block_diag_adjs, block_masks), (block_diag_adjs_norm, block_masks_norm), node_orders, ori_index = treegraph.get_layer_graph(layer_id=i, if_aug = args.if_aug, base_model=args.base_model)
                print('===:', block_masks)
                print('----:', block_masks_norm)
            elif args.base_model == 'STGODE':
                (block_diag_adjs_norm_sp, block_masks), (block_diag_adjs_norm_se, block_masks_norm), node_orders, ori_index = treegraph.get_layer_graph(layer_id=i, if_aug=args.if_aug, base_model=args.base_model)
                print('===:', block_diag_adjs_norm_sp)
                print('----:', block_diag_adjs_norm_se)
                block_diag_adjs_norm = [block_diag_adjs_norm_sp, block_diag_adjs_norm_se]
            elif args.base_model == 'ASTGCN':
                (block_diag_adjs_norm0, block_masks_norm0), (block_diag_adjs_norm1, block_masks_norm1), (block_diag_adjs_norm2, block_masks_norm2), node_orders, ori_index = treegraph.get_layer_graph(layer_id=i, if_aug=args.if_aug, base_model=args.base_model)
                print('--0--:', block_diag_adjs_norm0)
                print('--1--:', block_diag_adjs_norm1)
                print('--2--:', block_diag_adjs_norm2)
                block_diag_adjs_norm = [block_diag_adjs_norm0, block_diag_adjs_norm1, block_diag_adjs_norm2]
                block_masks = block_masks_norm0
                print('===', block_masks)

            pred_engine = PRED_Engine(device=args.device,
                                model=model,
                                dataloader=dataloader,
                                scaler=scaler,
                                sampler=None,
                                loss_fn=loss_fn,
                                lrate=args.learning_rate,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                clip_grad_value=args.clip_grad_value,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                log_dir=log_dir,
                                logger=logger,
                                seed=args.seed,
                                des=args.pred_description,
                                block_diag_adjs = block_diag_adjs_norm,
                                block_mask=block_masks,
                                node_orders = node_orders,
                                layer_id = i,
                                if_aug = args.if_aug,
                                ori_index = ori_index,
                                base_model = args.base_model
            )
            logger.info('--------layer:{} predictor training--------'.format(i))
            pred_engine.train()

            logger.info('--------layer:{} predictor testing--------'.format(i))
            args.bs = 1
            sliding_dataloader, sliding_scaler = load_sliding_dataset(data_path, args, logger)
            pred_engine._dataloader = sliding_dataloader
            pred_engine._scaler = sliding_scaler
            pred_engine.evaluate('test')

            torch.cuda.empty_cache()
    else:
        # both enc predictor are trained
        enc_model_name = 'enc_s{}_{}.pt'.format(args.enc_seed, args.enc_description) if args.enc_description else 'enc_s{}.pt'.format(args.enc_seed)
        print(log_dir)
        enc_model.load_state_dict(torch.load(
            os.path.join(log_dir, enc_model_name)))
        enc_model.eval()
        
        for layer_id, predictor in enumerate(predictors):
            if layer_id == 0:
                continue
            predictor_name = 'pred_l{}_s{}_{}.pt'.format(layer_id,args.pred_seed,args.pred_description) if args.pred_description else 'pred_l{}_s{}.pt'.format(layer_id,args.pred_seed)
            predictor.load_state_dict(torch.load(
            os.path.join(log_dir, predictor_name)))
            predictor.eval()
    
    #------------Train Agent--------------#
    args.bs = 1
    treegraph.convert_to_tensor(args)
    treegraph.get_children_ids(args)
    if args.dataset == 'CA':
        if args.base_model == 'GWNET':
            args.layer_model_time_table = [0.4, 0.1, 0.0333]
            thres, unit = 3400, 2200
            chunk_sizes = [None, 1, 3]
        elif args.base_model == 'STGODE':
            args.layer_model_time_table = [0.25, 0.125]
            thres, unit = 4000, 4300
            chunk_sizes = [None, 1]
            args.sample_size = 32
            args.buffer_size = 250
            args.learn_cnts = 5
        elif args.base_model == 'ASTGCN':
            args.layer_model_time_table = [0.5, 0.125, 0.0425]
            thres, unit = 3400, 2200
            chunk_sizes = [None, 1, 3]
    elif args.dataset == 'chengdu':
        if args.base_model == 'GWNET':
            args.layer_model_time_table = [0.1, 0.03333, 0.01111]
            thres, unit = 4000, 2200
            chunk_sizes = [None, 1, 3]
        elif args.base_model == 'STGODE':
            args.layer_model_time_table = [0.04, 0.02, 0.01]
            thres, unit = 3000, 3200
            chunk_sizes = [None, 1, 2]
            args.sample_size = 32
            args.learning_rate = 0.001
            args.buffer_size = 300
            args.learn_cnts = 5
        elif args.base_model == 'ASTGCN':
            args.layer_model_time_table = [0.1, 0.03333, 0.01111]
            thres, unit = 3400, 2200
            chunk_sizes = [None, 1, 3]

    logger.info('Time cost table for each layer: {}'.format(args.layer_model_time_table))

    agent = Agent(  embed_dim = args.agent_embed_dim,
                    hidden_dim = args.agent_hidden_dim,
                    action_size = args.action_size,
                    num_subgs = num_subgs,
                    max_layer = args.max_layer+1,
                    dropout_rate = args.dropout_rate,
                    if_bn = args.if_bn,
                    buffer_size = args.buffer_size,
                    if_ft = args.if_ft,
                    device = args.device 
    )

    env = Environment(  graph = treegraph,
                        standard_time_cost=args.standard_time_cost,
                        skip_time_cost = args.skip_time_cost,
                        naive_cost=args.naive_cost,
                        max_layer = args.max_layer+1,
                        tradeoff = args.tradeoff,
                        layer_model_time_table = args.layer_model_time_table,
                        action_size = args.action_size,
                        batch_size = args.bs,
                        horizon = args.horizon,
                        node_num = node_num,
                        modelsets = predictors,
                        device = args.device,
                        if_aug=args.if_aug,
                        thres = thres,
                        unit = unit,
                        chunk_sizes = chunk_sizes
    )
        
    if args.mode == 'train_agent':
        dataloader, scaler = load_sliding_dataset(data_path, args, logger)
        optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=args.learning_rate, weight_decay=args.wdecay)
        if args.if_ft:
            logger.info('with encoder fine tuning!!!')
            optimizer_enc = torch.optim.Adam(enc_model.parameters(), lr=0.00001, weight_decay=args.wdecay)  # new
        else:
            logger.info('without encoder fine tuning.')
            optimizer_enc = None
        scheduler = None

        agent_engine = AGENT_Engine(device=args.device,
                            enc_model=enc_model,
                            agent = agent,
                            env = env,
                            dataloader=dataloader,
                            scaler=scaler,
                            sampler=None,
                            criterion = F.l1_loss, # for accuracy reward computing
                            gamma = args.gamma,
                            sample_size  = args.sample_size,
                            loss_fn = F.mse_loss,  # F.l1_loss, # for train rl
                            ft_loss_fn = F.mse_loss,
                            lrate = args.learning_rate,
                            optimizer=optimizer,
                            optimizer_enc=optimizer_enc,
                            scheduler=scheduler,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=log_dir,
                            logger=logger,
                            seed=args.seed,
                            des=args.agent_description,
                            n_increments = args.n_increments,
                            learn_cnts = args.learn_cnts, 
                            learn_thres = args.learn_thres,
                            learn_times = args.learn_times,
                            if_ft = args.if_ft,
                            base_model=args.base_model
        )
        agent_engine.train()

    elif args.mode == 'test_agent':
        dataloader, scaler = load_sliding_dataset(data_path, args, logger)

        agent_engine = AGENT_Engine(device=args.device,
                            enc_model=enc_model,
                            agent = agent,
                            env = env,
                            dataloader=dataloader,
                            scaler=scaler,
                            sampler=None,
                            criterion = F.l1_loss, # for accuracy reward computing
                            gamma = args.gamma,
                            sample_size  = args.sample_size,
                            loss_fn=F.mse_loss,# F.l1_loss, # for train rl
                            ft_loss_fn = F.mse_loss,
                            lrate = args.learning_rate,
                            optimizer=None,
                            optimizer_enc=None,
                            scheduler=None,
                            clip_grad_value=args.clip_grad_value,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            log_dir=log_dir,
                            logger=logger,
                            seed=args.seed,
                            des=args.agent_description,
                            n_increments = args.n_increments,
                            learn_cnts = args.learn_cnts,
                            learn_thres = args.learn_thres,
                            learn_times = args.learn_times,
                            if_ft = args.if_ft,
                            base_model = args.base_model
        )

        # epochs = list(range(30, 160, 10))
        # for epoch in epochs:
        #     agent_engine.load_model(agent_engine._save_path, epoch)
        #     logger.info(args.agent_description + '   ' + str(epoch))
        #     agent_engine.evaluate('test')

        agent_engine.load_model(agent_engine._save_path)
        agent_engine.evaluate('test')


if __name__ == "__main__":
    main()