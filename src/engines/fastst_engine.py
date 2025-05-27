import os
import random
import time

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.base.engine import BaseEngine
from src.utils.metrics import masked_mae
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics

class AGENT_Engine(BaseEngine):
    def __init__(self, device, enc_model, agent, env, dataloader, scaler, sampler, criterion, gamma, sample_size, lrate, loss_fn, ft_loss_fn, des,
                 optimizer, optimizer_enc, scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, n_increments, learn_cnts, learn_thres,learn_times, if_ft, base_model):
        self._device = device
        self.enc_model = enc_model
        self.enc_model.to(self._device)
        self.enc_model.eval()
        
        self.agent = agent
        self.agent.q_net.to(self._device)
        self.agent.target_net.to(self._device)
        self.env = env
        self.base_model = base_model

        self._sample_size = sample_size
        self._dataloader = dataloader

        self._scaler = scaler

        self._loss_fn = loss_fn
        self._ft_loss_fn = ft_loss_fn
        self._optimizer = optimizer
        self._optimizer_enc = optimizer_enc  # new
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._criterion = criterion
        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._des = des
        self._gamma = gamma
        self._n_increments = n_increments
        self._lrate = lrate
        self._learn_cnts = learn_cnts
        self._learn_thres = learn_thres
        self._learn_times = learn_times
        self.if_ft = if_ft

        self.COLORS = {
                        "red": "\033[91m",
                        "green": "\033[92m",
                        "yellow": "\033[93m",
                        "blue": "\033[94m",
                        "magenta": "\033[95m",
                        "cyan": "\033[96m",
                        "white": "\033[97m",
                        "reset": "\033[0m",
                    }

    def tradd_off_estimation(self):
        c_layer, ratio = [], []
        for subg in self.env.multigraph.layer_order_node:
            ratio.append(0.2*len(subg.nodes)/len(self.env.multigraph.root.nodes))
            c_layer.append(self.env.layer_model_time_table[subg.layer])
        k, b = np.polyfit(ratio, c_layer, 1)
        x_fit = np.linspace(min(ratio), max(ratio), 100)
        y_fit = k * x_fit + b
        plt.figure()
        plt.scatter(ratio, c_layer)
        plt.plot(x_fit, y_fit, label=f'Fitted line: y={k:.2f}x+{b:.2f}')
        plt.xlabel('0.2*ratio')
        plt.ylabel('c_layer')
        plt.legend()
        plt.savefig(os.path.join(self._save_path, 'figures', 'fit.png'))
        plt.close()

    def save_model(self, save_path, epoch=None):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not epoch:
            filename = 'agent_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'agent_s{}.pt'.format(self._seed)
        else:
            filename = 'agent_e{}_s{}_{}.pt'.format(epoch, self._seed, self._des) if self._des else 'agent_e{}_s{}.pt'.format(epoch, self._seed)
        torch.save({
            'q_net_state_dict': self.agent.q_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
        }, os.path.join(save_path, filename))

        # new, save fine-tuned encoder
        if self.if_ft:
            if not epoch:
                filename = 'enc_ft_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'enc_ft_s{}.pt'.format(self._seed)
            else:
                filename = 'enc_ft_e{}_s{}_{}.pt'.format(epoch, self._seed, self._des) if self._des else 'enc_ft_e{}_s{}.pt'.format(epoch, self._seed)
            torch.save(self.enc_model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path, epoch=None):
        if not epoch:
            filename = 'agent_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'agent_s{}.pt'.format(self._seed)
        else:
            filename = 'agent_e{}_s{}_{}.pt'.format(epoch, self._seed, self._des) if self._des else 'agent_e{}_s{}.pt'.format(epoch, self._seed)
        checkpoint = torch.load(os.path.join(save_path, filename))
        self.agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.agent.q_net.to(self._device)
        self.agent.target_net.to(self._device)

        # new, load fine-tuned encoder
        if self.if_ft:
            if not epoch:
                filename = 'enc_ft_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'enc_ft_s{}.pt'.format(self._seed)
            else:
                filename = 'enc_ft_e{}_s{}_{}.pt'.format(epoch, self._seed, self._des) if self._des else 'enc_ft_e{}_s{}.pt'.format(epoch, self._seed)
            self.enc_model.load_state_dict(torch.load(os.path.join(save_path, filename)))
            self.enc_model.to(self._device)

    def train_batch(self, epoch=None):
        self.agent.q_net.train()
        if self.if_ft:
            self.enc_model.train()
        else:
            self.enc_model.eval()

        memory, memory_cnts = None, self._n_increments
        init_flag = False
        current_sample = 0
        total_rewards_sets = []
        total_accu_rewards_sets, total_eff_rewards_sets, total_pen_rewards_sets = {}, {}, {}
        for action in [0,1,2]:
            total_accu_rewards_sets[action], total_eff_rewards_sets[action], total_pen_rewards_sets[action] = [], [], []
        processed_nodes_set = []
        action_proportion = np.array([0,0,0])
        initial_subg_id = 1
        end_subg_id = 1 + torch.sum(self.env.layer_index==1).item()
        self.env.last_update_table[:,:] = 0
        for cnt, (X, label, sample) in tqdm(enumerate(self._dataloader['train_loader'].get_iterator())):
            # print('-----batch{}------'.format(cnt))
            self.env.last_update_table += 1
            X, label = self._to_device(self._to_tensor([X,label]))

            with torch.no_grad():
                if memory_cnts == self._n_increments:
                    memory, memory_cnts = None, 0 # clear memory
                else:
                    memory_cnts +=1

                embeds, memory = self.enc_model.get_agent_embed(X, memory)

                total_rewards, total_accu_rewards, total_eff_rewards, total_pen = 0, {}, {}, {}
                for action in [0,1,2]:
                    total_accu_rewards[action], total_eff_rewards[action], total_pen[action] = [], [], []

                if not init_flag:
                    state = self.env.reset(embeds,X) # init cache
                    init_flag = True
                else:
                    state = self.env.reset(embeds)
                done = False

                processed_nodes = 0
                action_count = np.array([0, 0, 0])
                subg_id = initial_subg_id
                state[0, :initial_subg_id, 0] = 0
                state[0, initial_subg_id:end_subg_id, 0] = 1
                while not done:
                    processed_nodes += 1

                    action, q_values, gen_masks = self.agent.act(state, subg_id, layer=state[0,subg_id,1].long(), training=True, check=True)

                    action_count[action.item()] = action_count[action.item()]+1
                    next_state, reward, done, (action, adjust_accuracy_reward, efficiency_reward, penalty) = self.env.step(state, action, subg_id, X, label, self._inverse_transform, self._criterion, cnt=cnt)
                    total_rewards+=reward
                    total_accu_rewards[action.item()].append(adjust_accuracy_reward)
                    total_eff_rewards[action.item()].append(efficiency_reward)
                    total_pen[action.item()].append(penalty)

                    next_subg_id = self.env._find_gid(next_state, subg_id).item()

                    self.agent.memory_step(state, action, subg_id, next_subg_id, reward, next_state, done)
                    if self.if_ft:
                        self.agent.memory_enc_step(X, state, action, subg_id, next_subg_id, reward, next_state, done)

                    if not done:
                        state = next_state
                        subg_id = next_subg_id

            action_count = action_count/np.sum(action_count)
            action_proportion = action_proportion + action_count
            processed_nodes_set.append(processed_nodes)

            total_rewards_sets.append(total_rewards)
            for action in [0, 1, 2]:
                # average for every decision
                total_accu_rewards_sets[action].append(torch.mean(torch.tensor(total_accu_rewards[action], dtype=torch.float)))
                total_eff_rewards_sets[action].append(torch.mean(torch.tensor(total_eff_rewards[action], dtype=torch.float)))
                total_pen_rewards_sets[action].append(torch.mean(torch.tensor(total_pen[action], dtype=torch.float)))

            self.count = 0
            if cnt % self._learn_cnts == 0 and cnt > self._learn_thres:
                for i in range(self._learn_times):

                    if len(self.agent.memory) > self._sample_size:
                        experiences = self.agent.memory.sample(self._sample_size)

                        states, actions, subg_ids, next_subg_ids, rewards, next_states, done = zip(*experiences)

                        states = torch.cat(states, dim=0).to(self._device)  # b,Na,d
                        actions = torch.tensor(actions,dtype =torch.long).unsqueeze(-1).to(self._device) #b, 1
                        subg_ids = torch.tensor(subg_ids,dtype =torch.long).to(self._device) #b
                        next_subg_ids = torch.tensor(next_subg_ids, dtype=torch.long).to(self._device) #b
                        next_states = torch.cat(next_states, dim=0).to(self._device) # b,Na,d
                        rewards = torch.tensor(rewards, dtype=torch.float32).to(self._device) #b
                        done = torch.tensor(done, dtype=torch.float32).to(self._device) #b

                        done_mask = done == 1
                        q_target = torch.zeros_like(rewards) #b,

                        # for done
                        q_target[done_mask] = rewards[done_mask] #b0

                        action_value = self.agent.q_net(next_states[~done_mask], next_subg_ids[~done_mask]) #b~0,d
                        target_action = torch.argmax(action_value, dim =1, keepdim=True) # b~0,1
                        q_target_next = self.agent.target_net(next_states[~done_mask], next_subg_ids[~done_mask]).gather(1,target_action).squeeze(-1).detach()
                        q_target[~done_mask] = rewards[~done_mask] + self._gamma * q_target_next

                        q_expect = self.agent.q_net(states, subg_ids).gather(1, actions).squeeze(-1) # b

                        self._optimizer.zero_grad()
                        loss = self._loss_fn(q_expect, q_target)
                        loss.backward()
                        self._optimizer.step()

                        self.agent.soft_update(self.agent.q_net, self.agent.target_net)
                        self.count += 1

                    # new, fine-tune the graph encoder
                    if self.if_ft:
                        if len(self.agent.memory_enc) > 40:
                            experiences = self.agent.memory_enc.sample(40)

                            inputs, states, actions, subg_ids, next_subg_ids, rewards, next_states, done = zip(*experiences)
                            inputs = torch.cat(inputs, dim=0).to(self._device)

                            states = torch.cat(states, dim=0).to(self._device)  # b,Na,d
                            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(self._device)  # b, 1
                            subg_ids = torch.tensor(subg_ids, dtype=torch.long).to(self._device)  # b
                            next_subg_ids = torch.tensor(next_subg_ids, dtype=torch.long).to(self._device)  # b
                            next_states = torch.cat(next_states, dim=0).to(self._device)  # b,Na,d
                            rewards = torch.tensor(rewards, dtype=torch.float32).to(self._device)  # b
                            done = torch.tensor(done, dtype=torch.float32).to(self._device)  # b

                            embeds, _ = self.enc_model.get_agent_embed(inputs, None)
                            states_enc = torch.cat([states[:,:,:2], embeds, states[:,:,2+embeds.shape[2]:]], dim=-1)
                            next_states_enc = torch.cat([next_states[:, :, :2], embeds, next_states[:, :, 2 + embeds.shape[2]:]], dim=-1)

                            done_mask = done == 1
                            q_target = torch.zeros_like(rewards)  # b,

                            # for done
                            q_target[done_mask] = rewards[done_mask]  # b0

                            action_value = self.agent.q_net(next_states_enc[~done_mask], next_subg_ids[~done_mask])  # b~0,d
                            target_action = torch.argmax(action_value, dim=1, keepdim=True)  # b~0,1
                            q_target_next = self.agent.target_net(next_states_enc[~done_mask], next_subg_ids[~done_mask]).gather(1, target_action).squeeze(
                                -1).detach()
                            q_target[~done_mask] = rewards[~done_mask] + self._gamma * q_target_next

                            q_expect = self.agent.q_net(states_enc, subg_ids).gather(1, actions).squeeze(-1)  # b

                            self._optimizer_enc.zero_grad()
                            loss_ft = self._ft_loss_fn(q_expect, q_target)
                            loss_ft.backward()
                            self._optimizer_enc.step()

                            self.agent.memory_enc.clear()

        self._logger.info('train min, max, avg number of processed tree nodes per batch:{:.2f}, {:.2f}, {:.2f}'.format(np.min(processed_nodes_set), np.max(processed_nodes_set), np.mean(processed_nodes_set)))
        self._logger.info('train avg actions proportion {} per batch'.format(list(action_proportion/cnt)))

        for action in [0, 1, 2]:
            # average for every sample(batch=1)
            total_accu_rewards_sets[action] = torch.nanmean(torch.tensor(total_accu_rewards_sets[action], dtype=torch.float))
            total_eff_rewards_sets[action] = torch.nanmean(torch.tensor(total_eff_rewards_sets[action], dtype=torch.float))
            total_pen_rewards_sets[action] = torch.nanmean(torch.tensor(total_pen_rewards_sets[action], dtype=torch.float))
        return torch.mean(torch.tensor(total_rewards_sets, dtype=torch.float)), total_accu_rewards_sets, total_eff_rewards_sets, total_pen_rewards_sets

    def train(self):
        self._logger.info(f"{self.COLORS['red']}Start training!{self.COLORS['reset']}")
        print("devices: enc:{}, pred:{}, agent:{}".format(next(self.enc_model.parameters()).device,
                                                          next(self.env.modelsets.parameters()).device,
                                                          next(self.agent.q_net.parameters()).device))

        wait = 0
        cur_rewards = -np.inf

        import copy
        initial_params = copy.deepcopy(self.enc_model.state_dict())
        self._logger.info('get initial params!')

        initial_params_q = copy.deepcopy(self.agent.q_net.state_dict())
        self._logger.info('get initial params!')

        for epoch in range(self._max_epochs):

            t1 = time.time()
            mtrain_rewards, train_accu_rewards, train_eff_rewards, train_pen_rewards = self.train_batch(epoch)
            t2 = time.time()

            v1 = time.time()
            mvalid_rewards, valid_accu_rewards, valid_eff_rewards, valid_pen_rewards = self.validate()
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = 'Epoch: {:03d}, Train Rewards: {:.3f}, Train time: {:.2f}, Valid Rewards: {:.3f}, Valid time: {:.2f}, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_rewards, t2 - t1, mvalid_rewards, v2 - v1, cur_lr))

            for action in [0, 1, 2]:
                message = 'Action {}:, Train Accuracy Rewards: {:.3f}, Train efficiency Rewards: {:.3f}, Train penalty Rewards: {:.3f}, Valid Accuracy Rewards: {:.3f}, Valid Efficiency Rewards: {:.3f}, Valid Penalty Rewards: {:.3f}'
                self._logger.info(message.format(action, train_accu_rewards[action], train_eff_rewards[action], train_pen_rewards[action], valid_accu_rewards[action], valid_eff_rewards[action], valid_pen_rewards[action]))

            if mvalid_rewards > cur_rewards:
                self.save_model(self._save_path)
                self._logger.info(f"{self.COLORS['blue']}Val rewards increase from {cur_rewards:.4f} to {mvalid_rewards:.4f}{self.COLORS['reset']}")
                cur_rewards = mvalid_rewards
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(f"{self.COLORS['red']}Early stop at epoch {epoch + 1}, loss = {mvalid_rewards:.6f}{self.COLORS['reset']}")
                    break
            
            if epoch % 5 ==0:  # 10
                mvalid_mae, mvalid_mape, mvalid_rmse, infer_t = self.evaluate('val', epoch)
                self._logger.info(f"{self.COLORS['cyan']}Epoch: {epoch + 1:03d}, Valid MAE: {mvalid_mae:.4f}, Valid RMSE: {mvalid_rmse:.4f}, Valid MAPE: {mvalid_mape:.4f}, Valid Time: {infer_t:.4f}s, LR: {cur_lr:.4e}{self.COLORS['reset']}")
                self.evaluate('test')
                self.save_model(self._save_path, epoch)

        self._logger.info(f"{self.COLORS['red']}End training!{self.COLORS['reset']}")

        trained_params = self.enc_model.state_dict()
        trained_params_q = self.agent.q_net.state_dict()

        for param_tensor in initial_params:
            if not torch.allclose(initial_params[param_tensor], trained_params[param_tensor], atol=1e-5):
                print(f'Parameter {param_tensor} has been updated.')
            else:
                print(f'Parameter {param_tensor} not update!!!')

        for param_tensor in initial_params_q:
            if not torch.allclose(initial_params_q[param_tensor], trained_params_q[param_tensor], atol=1e-5):
                print(f'Parameter {param_tensor} has been updated.')
            else:
                print(f'Parameter {param_tensor} not update!!!')

    
    def validate(self):
        # compute rewards only

        self.agent.q_net.eval()
        self.enc_model.eval()  # new

        total_rewards_sets = []
        total_accu_rewards_sets, total_eff_rewards_sets, total_pen_rewards_sets = {}, {}, {}
        for action in [0,1,2]:
            total_accu_rewards_sets[action], total_eff_rewards_sets[action], total_pen_rewards_sets[action] = [], [], []

        processed_nodes_set = []
        action_proportion = np.array([0, 0, 0])
        initial_subg_id = 1
        end_subg_id = 1 + torch.sum(self.env.layer_index == 1).item()
        self.env.last_update_table[:, :] = 0
        with torch.no_grad():
            memory, memory_cnts = None, self._n_increments
            init_flag = False

            for cnt, (X, label, sample) in tqdm(enumerate(self._dataloader['val_loader'].get_iterator())):
                self.env.last_update_table += 1
                X, label = self._to_device(self._to_tensor([X,label])) 
                with torch.no_grad():
                    if memory_cnts == self._n_increments:
                        memory, memory_cnts = None, 0 # clear memory
                    else:
                        memory_cnts +=1

                    embeds, memory = self.enc_model.get_agent_embed(X, memory)

                    total_rewards, total_accu_rewards, total_eff_rewards, total_pen = 0, {}, {}, {}
                    for action in [0, 1, 2]:
                        total_accu_rewards[action], total_eff_rewards[action], total_pen[action] = [], [], []

                    if not init_flag:
                        state = self.env.reset(embeds,X) # init cache
                        init_flag = True
                    else:
                        state = self.env.reset(embeds)
                    done = False

                    processed_nodes = 0
                    action_count = np.array([0, 0, 0])
                    subg_id = initial_subg_id
                    state[0, :initial_subg_id, 0] = 0
                    state[0, initial_subg_id:end_subg_id, 0] = 1
                    while not done:
                        processed_nodes += 1

                        action, q_values, gen_masks = self.agent.act(state, subg_id, layer=state[0,subg_id,1].long(), training=False, check=True)
                        action_count[action.item()] = action_count[action.item()] + 1
                        next_state, reward, done, (action, adjust_accuracy_reward, efficiency_reward, penalty) = self.env.step(state, action, subg_id, X, label, self._inverse_transform, self._criterion)
                        total_rewards += reward
                        total_accu_rewards[action.item()].append(adjust_accuracy_reward)
                        total_eff_rewards[action.item()].append(efficiency_reward)
                        total_pen[action.item()].append(penalty)

                        if not done:
                            state = next_state
                            subg_id = self.env._find_gid(state, subg_id).item()

                    action_count = action_count / np.sum(action_count)
                    action_proportion = action_proportion + action_count
                    processed_nodes_set.append(processed_nodes)

                    total_rewards_sets.append(total_rewards)
                    for action in [0, 1, 2]:
                        # average for every decision
                        total_accu_rewards_sets[action].append(torch.mean(torch.tensor(total_accu_rewards[action], dtype=torch.float)))
                        total_eff_rewards_sets[action].append(torch.mean(torch.tensor(total_eff_rewards[action], dtype=torch.float)))
                        total_pen_rewards_sets[action].append(torch.mean(torch.tensor(total_pen[action], dtype=torch.float)))

        self._logger.info('val min, max, mean number of processed tree nodes per batch:{:.2f}, {:.2f}, {:.2f}'.format(np.min(processed_nodes_set), np.max(processed_nodes_set), np.mean(processed_nodes_set)))
        self._logger.info('val avg actions proportion {} per batch'.format(list(action_proportion / cnt)))

        for action in [0, 1, 2]:
            # average for every sample(batch=1)
            total_accu_rewards_sets[action] = torch.nanmean(torch.tensor(total_accu_rewards_sets[action], dtype=torch.float))
            total_eff_rewards_sets[action] = torch.nanmean(torch.tensor(total_eff_rewards_sets[action], dtype=torch.float))
            total_pen_rewards_sets[action] = torch.nanmean(torch.tensor(total_pen_rewards_sets[action], dtype=torch.float))

        return torch.mean(torch.tensor(total_rewards_sets, dtype=torch.float)), total_accu_rewards_sets, total_eff_rewards_sets, total_pen_rewards_sets

    def evaluate(self, mode, epoch=None):

        self.agent.q_net.eval()
        self.enc_model.eval()  # new

        preds = []
        labels = []
        self.env.last_update_table[:,:] = 0

        with torch.no_grad():
            memory, memory_cnts = None, self._n_increments
            init_flag = False

            processed_nodes_set = []
            last_update_record = {}
            action_proportion = {}
            for i in range(self.env.max_layer):
                last_update_record[i] = []
                action_proportion[i] = np.array([0, 0, 0])
            initial_subg_id = 1
            end_subg_id = 1 + torch.sum(self.env.layer_index == 1).item()
            v1 = time.time()
            for cnt, (X, label, sample) in tqdm(enumerate(self._dataloader[mode + '_loader'].get_iterator())):
                # print('batch{}'.format(cnt))
                self.env.last_update_table += 1
                X, label = self._to_device(self._to_tensor([X,label])) 
                with torch.no_grad():
                    if memory_cnts == self._n_increments:
                        memory, memory_cnts = None, 0 # clear memory
                    else:
                        memory_cnts+=1

                    embeds, memory = self.enc_model.get_agent_embed(X, memory)

                    if not init_flag:
                        state = self.env.reset(embeds,X) # init cache
                        init_flag = True
                    else:
                        state = self.env.reset(embeds)
                    done = False

                    processed_nodes = 0
                    action_count = {}
                    for l in range(self.env.max_layer):
                        action_count[l] = np.array([0, 0, 0])
                    final_acts = []
                    subg_id = initial_subg_id
                    state[0, :initial_subg_id, 0] = 0
                    state[0, initial_subg_id:end_subg_id, 0] = 1
                    while not done:
                        processed_nodes += 1

                        l = int(state[0, subg_id, 1].item())
                        action = self.agent.act(state, subg_id, layer=l, training=False)
                        action_count[l][action.item()] = action_count[l][action.item()] + 1

                        final_acts.append((state, action, subg_id))
                        next_state, done = self.env.slim_step(state, action, subg_id)

                        if not done:
                            state = next_state
                            subg_id = self.env._find_gid(state, subg_id).item()


                    for l in range(self.env.max_layer):
                        action_proportion[l] = action_proportion[l] + action_count[l]
                    processed_nodes_set.append(processed_nodes)

                    pred  = self.env.pred(final_acts, X, self.base_model) # here we pred graphs of same level together
                    pred, label = self._inverse_transform([pred, label])

                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

                for l in range(self.env.max_layer):
                    last_update_record[l].append(torch.mean(self.env.last_update_table[l]).item())

            v2 = time.time()

        self._logger.info('test min, max, avg number of processed tree nodes per batch:{:.2f}, {:.2f}, {:.2f}'.format(np.min(processed_nodes_set), np.max(processed_nodes_set), np.mean(processed_nodes_set)))
        for l in range(self.env.max_layer):
            self._logger.info('layer {} test avg actions proportion {} per batch'.format(l, list(action_proportion[l] / (np.sum(action_proportion[l])+1e-10))))
            plt.figure()
            plt.plot(last_update_record[l])
            if not os.path.exists(os.path.join(self._save_path, 'figures', self._des)):
                os.makedirs(os.path.join(self._save_path, 'figures', self._des))
            if mode=='test':
                plt.savefig(os.path.join(self._save_path, 'figures', self._des, '{}_layer{}.png'.format(mode, l)))
            elif epoch != None:
                plt.savefig(os.path.join(self._save_path, 'figures', self._des, '{}_layer{}.png'.format(epoch, l)))
            plt.close()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse, v2-v1
        
        elif mode =='test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(self.env.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            self._logger.info(f"{self.COLORS['cyan']} Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}, Inference TIME: {v2-v1:.3f} {self.COLORS['reset']}")

    def evaluate_pred(self, mode, layer, diag=True, base_model='GWNET'):
        all_nodes, layer_ids = np.array(self.env.multigraph.layer_order_node), np.array(self.env.multigraph.layer_ids[0, :, 0])
        current_layer_nodes = all_nodes[layer_ids == layer]
        print(len(current_layer_nodes))
        preds = []
        labels = []
        predictor = self.env.modelsets[layer]
        predictor.eval()
        with torch.no_grad():
            v1 = time.time()
            for cnt, (X, label, sample) in tqdm(enumerate(self._dataloader[mode + '_loader'].get_iterator())):
                pred = torch.zeros_like(self.env.cache[:,0]).to(self._device)
                X, label = self._to_device(self._to_tensor([X, label]))
                if diag:
                    if base_model == 'GWNET':
                        adjs, layer_nodes = [], []
                        for node in current_layer_nodes:
                            # adj = torch.tensor(node.adj_matrix_norm).to(self._device)
                            adj = node.adj_matrix_norm
                            nodes = node.nodes
                            adjs.append(adj)
                            layer_nodes.append(nodes)
                        diag_adjs, diag_masks = self.env.multigraph.block_diag(adjs)  # Ni*Ni
                        # nodes = torch.tensor([item for sublist in layer_nodes for item in sublist]).long().to(self._device)
                        nodes = torch.cat(layer_nodes)
                        pred[:,:,nodes] = predictor(X[:, :, nodes], diag_adjs, nodes, diag_masks).squeeze(dim=-1)
                    elif base_model == 'STGODE':
                        adjs_sp, adjs_se, layer_nodes = [], [], []
                        for node in current_layer_nodes:
                            nodes = node.nodes
                            adjs_sp.append(node.adj_matrix_norm[0])
                            adjs_se.append(node.adj_matrix_norm[1])
                            layer_nodes.append(nodes)
                        diag_adjs_sp, diag_masks_sp = self.env.multigraph.block_diag(adjs_sp)  # Ni*Ni
                        diag_adjs_se, diag_masks_se = self.env.multigraph.block_diag(adjs_se)  # Ni*Ni
                        diag_adjs = [diag_adjs_sp, diag_adjs_se]
                        nodes = torch.cat(layer_nodes)
                        pred[:, :, nodes] = predictor(X[:, :, nodes], diag_adjs, nodes, None).squeeze(dim=-1)
                    elif base_model == 'ASTGCN':
                        adjs0, adjs1, adjs2, layer_nodes = [], [], [], []
                        for node in current_layer_nodes:
                            nodes = node.nodes
                            adjs0.append(node.adj_matrix_norm[0])
                            adjs1.append(node.adj_matrix_norm[1])
                            adjs2.append(node.adj_matrix_norm[2])
                            layer_nodes.append(nodes)
                        diag_adjs0, diag_masks0 = self.env.multigraph.block_diag(adjs0)  # Ni*Ni
                        diag_adjs1, diag_masks1 = self.env.multigraph.block_diag(adjs1)  # Ni*Ni
                        diag_adjs2, diag_masks2 = self.env.multigraph.block_diag(adjs2)  # Ni*Ni
                        diag_adjs = [diag_adjs0, diag_adjs1, diag_adjs2]
                        diag_masks = diag_masks0
                        nodes = torch.cat(layer_nodes)
                        pred[:, :, nodes] = predictor(X[:, :, nodes], diag_adjs, nodes, diag_masks).squeeze(dim=-1)
                else:
                    for node in current_layer_nodes:
                        # adj = torch.tensor(node.adj_matrix_norm).to(self._device)
                        adj = node.adj_matrix_norm
                        nodes = node.nodes
                        pred[:, :, nodes] = predictor(X[:, :, nodes], adj, nodes, 1.0).squeeze(dim=-1)
                # if torch.sum(pred==0) != 0:
                #     print('error!!!')
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
            v2 = time.time()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._criterion(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse, v2 - v1

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(self.env.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            self._logger.info(
                f"{self.COLORS['cyan']} Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}, Inference TIME: {v2 - v1:.3f} {self.COLORS['reset']}")

    def evaluate_naive(self, mode, layer, gap, diag=True, base_model='GWNET'):
        all_nodes, layer_ids = np.array(self.env.multigraph.layer_order_node), np.array(self.env.multigraph.layer_ids[0, :, 0])
        current_layer_nodes = all_nodes[layer_ids == layer]
        print(len(current_layer_nodes))
        preds = []
        labels = []
        predictor = self.env.modelsets[layer]
        predictor.eval()

        with torch.no_grad():
            v1 = time.time()
            # self.env.cache[:,:,:,:] = 0
            for cnt, (X, label, sample) in tqdm(enumerate(self._dataloader[mode + '_loader'].get_iterator())):
                X, label = self._to_device(self._to_tensor([X, label]))
                if cnt==0:
                    self.env.cache = X[..., 0].mean(dim=1).unsqueeze(1).unsqueeze(1).expand(-1, self.env.max_layer, self.env.horizon, -1)
                # 取cache一定注意clone!!!
                # pred = self.env.cache[:,layer]
                if cnt==0 or gap==0 or cnt%gap == 0:
                    batch_pred = torch.zeros_like(self.env.cache[:,0]).to(self._device)
                    X, label = self._to_device(self._to_tensor([X, label]))
                    if diag:
                        if base_model == 'GWNET':
                            adjs, layer_nodes = [], []
                            for node in current_layer_nodes:
                                # adj = torch.tensor(node.adj_matrix_norm).to(self._device)
                                adj = node.adj_matrix_norm
                                nodes = node.nodes
                                adjs.append(adj)
                                layer_nodes.append(nodes)
                            diag_adjs, diag_masks = self.env.multigraph.block_diag(adjs)  # Ni*Ni
                            # nodes = torch.tensor([item for sublist in layer_nodes for item in sublist]).long().to(self._device)
                            nodes = torch.cat(layer_nodes)
                            batch_pred[:,:,nodes] = predictor(X[:, :, nodes], diag_adjs, nodes, diag_masks).squeeze(dim=-1)
                        elif base_model == 'STGODE':
                            adjs_sp, adjs_se, layer_nodes = [], [], []
                            for node in current_layer_nodes:
                                nodes = node.nodes
                                adjs_sp.append(node.adj_matrix_norm[0])
                                adjs_se.append(node.adj_matrix_norm[1])
                                layer_nodes.append(nodes)
                            diag_adjs_sp, diag_masks_sp = self.env.multigraph.block_diag(adjs_sp)  # Ni*Ni
                            diag_adjs_se, diag_masks_se = self.env.multigraph.block_diag(adjs_se)  # Ni*Ni
                            diag_adjs = [diag_adjs_sp, diag_adjs_se]
                            nodes = torch.cat(layer_nodes)
                            batch_pred[:, :, nodes] = predictor(X[:, :, nodes], diag_adjs, nodes, None).squeeze(dim=-1)
                    else:
                        for node in current_layer_nodes:
                            # adj = torch.tensor(node.adj_matrix_norm).to(self._device)
                            adj = node.adj_matrix_norm
                            nodes = node.nodes
                            batch_pred[:, :, nodes] = predictor(X[:, :, nodes], adj, nodes, 1.0).squeeze(dim=-1)
                    self.env._update_cache(batch_pred.clone(), layer)
                    # if torch.sum(batch_pred==0) != 0:
                    #     print('error!!!')
                pred = self.env._naive_pred(layer, [i for i in range(X.shape[2])])

                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
            v2 = time.time()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._criterion(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse, v2 - v1

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(self.env.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                # self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            self._logger.info(
                f"{self.COLORS['cyan']} Average Test MAE: {np.mean(test_mae):.4f}, Test RMSE: {np.mean(test_rmse):.4f}, Test MAPE: {np.mean(test_mape):.4f}, Inference TIME: {v2 - v1:.3f} {self.COLORS['reset']}")

class PRED_Engine(BaseEngine):
    def __init__(self, block_diag_adjs, block_mask, node_orders, layer_id, if_aug, ori_index, base_model, **args):
        super(PRED_Engine, self).__init__(**args)
        if base_model == 'ASTGCN':
            for p in self.model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.uniform_(p)

        if base_model == 'STGODE':
            self.block_diag_adjs = block_diag_adjs
            self.block_diag_adjs[0] = self.block_diag_adjs[0].to(self._device)
            self.block_diag_adjs[1] = self.block_diag_adjs[1].to(self._device)
            self.mask = None
        elif base_model == 'GWNET':
            self.block_diag_adjs = block_diag_adjs.to(self._device)
            if self.model.use_mask:
                self.mask = block_mask.to(self._device)
            else:
                self.mask = None
        elif base_model == 'ASTGCN':
            self.block_diag_adjs = block_diag_adjs
            self.block_diag_adjs[0] = self.block_diag_adjs[0].to(self._device)
            self.block_diag_adjs[1] = self.block_diag_adjs[1].to(self._device)
            self.block_diag_adjs[2] = self.block_diag_adjs[2].to(self._device)
            self.mask = block_mask.to(self._device)

        self.node_orders = node_orders.to(self._device)
        self.layer_id = layer_id
        self.if_aug = if_aug
        self.ori_index = ori_index.to(self._device) if if_aug else None

        if self.if_aug:
            if 'without_reindex' in self._des:
                self._logger.info("loss is calculated with augmented data!")
            else:
                self._logger.info("loss is calculated with original data!")

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'pred_l{}_s{}_{}.pt'.format(self.layer_id,self._seed,self._des) if self._des else 'pred_l{}_s{}.pt'.format(self.layer_id,self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        filename = 'pred_l{}_s{}_{}.pt'.format(self.layer_id,self._seed,self._des) if self._des else 'pred_l{}_s{}.pt'.format(self.layer_id,self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        if self._dataloader['train_loader'].sliding:
            self._dataloader['train_loader'].shuffle(sample_only=False)
        else:
            self._dataloader['train_loader'].shuffle()
        for X, label, *rest in tqdm(self._dataloader['train_loader'].get_iterator()):

            self._optimizer.zero_grad()
            # X (b, t, n, f), 
            X,label = self._to_device(self._to_tensor([X,label]))

            X_aug = X.index_select(2, self.node_orders)
            label_aug = label.index_select(2, self.node_orders)

            pred_aug = self.model(X_aug, self.block_diag_adjs, self.node_orders, self.mask)

            pred_aug, label_aug = self._inverse_transform([pred_aug, label_aug])

            if self.if_aug and ('without_reindex' not in self._des):
                # only care original parts
                pred = pred_aug.index_select(2, self.ori_index)
                label = label_aug.index_select(2, self.ori_index)
            else:
                pred = pred_aug
                label = label_aug

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()


            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []

        v1 = time.time()
        with torch.no_grad():
            for X, label, *rest in tqdm(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)

                X,label = self._to_device(self._to_tensor([X,label]))

                X_aug = X.index_select(2, self.node_orders)
                label_aug = label.index_select(2, self.node_orders)

                pred_aug = self.model(X_aug, self.block_diag_adjs, self.node_orders, self.mask)

                pred_aug, label_aug = self._inverse_transform([pred_aug, label_aug])

                if self.if_aug:
                    # only care original parts
                    pred = pred_aug.index_select(2, self.ori_index)
                    label = label_aug.index_select(2, self.ori_index)
                else:
                    pred = pred_aug
                    label = label_aug

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
        v2 = time.time()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Inference TIME:{:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape), v2-v1))

class ENC_Engine(BaseEngine):
    def __init__(self, n_increments, **args):
        super(ENC_Engine, self).__init__(**args)
        self.n_increments = n_increments

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'enc_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'enc_s{}.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        filename = 'enc_s{}_{}.pt'.format(self._seed, self._des) if self._des else 'enc_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        memory, memory_cnts = None, self.n_increments
        if self._dataloader['train_loader'].sliding:
            self._dataloader['train_loader'].shuffle(sample_only=True)
        else:
            self._dataloader['train_loader'].shuffle()
        for cnt, (X, label, *rest) in tqdm(enumerate(self._dataloader['train_loader'].get_iterator())):
            self._optimizer.zero_grad()
            # X (b, t, n, f), 
            X, label = self._to_device(self._to_tensor([X, label]))
            label = label.squeeze(-1)

            if memory_cnts == self.n_increments:
                memory, memory_cnts = None, 0 # clear memory
            else:
                memory_cnts += 1

            pred, memory = self.model(X, memory)
            memory = memory.detach()
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        memory, memory_cnts = None, self.n_increments

        v1 = time.time()
        with torch.no_grad():
            for cnt, (X, label, *rest) in tqdm(enumerate(self._dataloader[mode + '_loader'].get_iterator())):
                # print('-----batch{}-----'.format(cnt))
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                label = label.squeeze(-1)

                if memory_cnts == self.n_increments:
                    memory, memory_cnts = None, 0 # clear memory
                else:
                    memory_cnts+=1

                pred, memory = self.model(X, memory)
                memory = memory.detach()
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
        v2 = time.time()

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)
            for i in range(self.model.seq_len):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Inference TIME: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape), v2-v1))



   