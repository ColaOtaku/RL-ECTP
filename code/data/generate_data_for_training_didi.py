import os
import argparse
import numpy as np
import pandas as pd


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week, sampling_interval='5m'):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df, axis=-1)

    feature_list = [data]

    start_time = np.datetime64('2016-10-01')
    timestamps = start_time + np.arange(num_samples) * np.timedelta64(int(sampling_interval[:-1]), sampling_interval[-1])

    if add_time_of_day:
        # Calculate time of day from timestamps
        time_ind = (timestamps.astype('datetime64[m]') - timestamps.astype('datetime64[D]')) / np.timedelta64(1, 'h') / 24.0
        time_of_day = np.tile(time_ind[:, None], [1, num_nodes])
        feature_list.append(np.expand_dims(time_of_day, axis=-1))

    if add_day_of_week:
        # Calculate day of week from timestamps
        dow = ((timestamps.astype('datetime64[D]').view('int64') -
                timestamps.astype('datetime64[W]').view('int64')) % 7) / 7.0
        day_of_week = np.tile(dow[:, None], [1, num_nodes])
        feature_list.append(np.expand_dims(day_of_week, axis=-1))

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    df = np.load(args.dataset + '/' + 'raw/' + 'cnt_300.npy').T
    print('original data shape:', df.shape)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    print('final data shape:', data.shape, 'idx shape:', idx.shape)

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0]
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = args.dataset + '/' + '2016'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chengdu', help='dataset name')
    # parser.add_argument('--years', type=str, default='2019',
    #                     help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')

    args = parser.parse_args()
    generate_train_val_test(args)
