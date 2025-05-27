import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp
from torch.utils.data import Dataset
from threading import Thread
from queue import Queue

class SlidingDataset(Dataset):
    def __init__(self, data, batch_size, window_size, stride, sample_size, logger):
        self.data = data
        self.stride = stride
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.window_size = window_size

        self.num_samples = data.shape[0] // sample_size
        self.windows_per_sample = (sample_size - window_size) // stride + 1
        self.batch_per_sample = self.windows_per_sample - batch_size +1
        self.total_batches = self.num_samples * self.batch_per_sample

        logger.info('Windows per sample: ' + str(self.windows_per_sample - batch_size +1))
        logger.info('Total batches: ' + str(self.total_batches))

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx):
        '''
            e.g
            Batch 0:
            [[1 2 3 4 5]
            [2 3 4 5 6]
            [3 4 5 6 7]]
            Sample Index: 0
            Batch 1:
            [[2 3 4 5 6]
            [3 4 5 6 7]
            [4 5 6 7 8]]
            Sample Index: 0
            Batch 2:
            [[3 4 5 6 7]
            [4 5 6 7 8]
            [5 6 7 8 9]]
            Batch 3:
            [[ 4  5  6  7  8]
            [ 5  6  7  8  9]
            [ 6  7  8  9 10]]
            Sample Index: 0
            Batch 4:
            [[11 12 13 14 15]
            [12 13 14 15 16]
            [13 14 15 16 17]]
            Sample Index: 1
            Batch 5:
            [[12 13 14 15 16]
            [13 14 15 16 17]
            [14 15 16 17 18]]
            Sample Index: 1
        '''
        sample_idx = (idx ) // (self.windows_per_sample - self.batch_size +1)
        window_start_idx = (idx ) % (self.windows_per_sample - self.batch_size +1)

        sample_start_idx = sample_idx * self.sample_size
        sample_data = self.data[sample_start_idx:sample_start_idx + self.sample_size]


        batch_samples = [
            sample_data[window_start_idx + i * self.stride: window_start_idx + i * self.stride + self.window_size]
            for i in range(self.batch_size)
        ]
        
        return np.stack(batch_samples, axis=0), sample_idx

class MyDataLoader:
    def __init__(self, data, idx, batch_size, stride, seq_len, pred_len, sample_size, logger, drop_last=True):
        self.datasets = SlidingDataset(data[idx.min():idx.max()+1], batch_size, seq_len + pred_len,
                                       stride, sample_size, logger)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_batch = len(self.datasets) - 1 if drop_last else len(self.datasets)
        self.sliding = True

        self.indices = []
        all_batch_idx, cur = range(self.n_batch), 0
        for _ in range(self.datasets.num_samples):
            self.indices.append([])
            for _ in range(self.datasets.batch_per_sample):
                if cur >= self.n_batch:
                    break
                self.indices[-1].append(all_batch_idx[cur])
                cur += 1

        self.logger = logger
        logger.info('Batch num: ' + str(self.n_batch))
        
    def get_iterator(self,):
        self.current_ind = 0
        if isinstance(self.indices[0], list):  # not shuffle or shuffle sample only
            indices = [item for sample in self.indices for item in sample]
        else:  # shuffle all data
            indices = self.indices

        def _wrapper():
            while self.current_ind< self.n_batch:
                idx = indices[self.current_ind]
                item, sample_idx = self.datasets.__getitem__(idx)
                yield item[:,:self.seq_len], item[:,self.seq_len:,:,0], sample_idx
                self.current_ind += 1
                
        return _wrapper()

    def shuffle(self, sample_only=False):
        if sample_only:
            np.random.shuffle(self.indices)
            self.logger.info('Sliding Dataset shuffled wrt sample.')
        else:
            if isinstance(self.indices[0], list):  # initial condition
                self.indices = [item for sample in self.indices for item in sample]
            np.random.shuffle(self.indices)
            self.logger.info('Sliding Dataset shuffled.')


class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        self.sliding = False
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        self.logger = logger
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx
        self.logger.info('Dataset shuffled.')


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                if len(idx_ind) > 1:
                    num_threads = len(idx_ind) // 2
                else:
                    num_threads = 1
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_sliding_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))
    
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        print(idx.min(),idx.max(),cat)
        dataloader[cat + '_loader'] = MyDataLoader(ptr['data'][..., :args.input_dim], idx, args.bs, \
                                                 1, args.seq_len, args.horizon, args.dataset_sample_size, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler    


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))
    
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
         'chengdu': [base_dir + 'chengdu', base_dir + 'chengdu/cnt_300_adj.npy', 6402],
        }
    assert dataset in d.keys()
    return d[dataset]