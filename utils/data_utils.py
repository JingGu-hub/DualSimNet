import os

import numpy as np
import torch
from scipy.io.arff import loadarff
from torch.utils.data import DataLoader
from torch.utils import data

from scipy import stats
import torch.nn.functional as F
from math import inf

from numpy.testing import assert_array_almost_equal

class TimeDataset(data.Dataset):
    def __init__(self, dataset, target, clean_target=None, pred=None, prob=None, mode='train'):
        self.dataset = dataset
        self.target = target
        self.clean_target = clean_target
        self.mode = mode

        if mode == 'labeled':
            self.pred_idx = pred.nonzero().reshape(-1)
            self.probability = [prob[i] for i in self.pred_idx]
            self.dataset = dataset[self.pred_idx]
            self.target = [target[i] for i in self.pred_idx]
        elif mode == 'unlabeled':
            self.pred = pred
            self.pred_idx = (1 - pred).nonzero().reshape(-1)
            self.dataset = dataset[self.pred_idx]
            self.target = [target[i] for i in self.pred_idx]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.dataset[index], self.target[index], self.clean_target[index], index
        elif self.mode == 'test':
            return self.dataset[index], self.target[index], index
        elif self.mode == 'labeled':
            return self.dataset[index], self.target[index], self.probability[index], index
        elif self.mode == 'unlabeled':
            return self.dataset[index], index

    def __len__(self):
        return len(self.target)

def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels

def shuffler_dataset(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train

def build_dataset_uea(args):
    data_path = args.data_dir
    train_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TRAIN.arff'))[0]
    test_data = loadarff(os.path.join(data_path, args.dataset, args.dataset + '_TEST.arff'))[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    train_dataset = train_X.transpose(0, 2, 1)
    train_target = train_y
    test_dataset = test_X.transpose(0, 2, 1)
    test_target = test_y

    num_classes = len(np.unique(train_target))
    train_target = transfer_labels(train_target)
    test_target = transfer_labels(test_target)

    ind = np.where(np.isnan(train_dataset))
    col_mean = np.nanmean(train_dataset, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_dataset[ind] = np.take(col_mean, ind[1])

    ind_test = np.where(np.isnan(test_dataset))
    test_dataset[ind_test] = np.take(col_mean, ind_test[1])

    train_dataset, train_target = shuffler_dataset(train_dataset, train_target)
    test_dataset, test_target = shuffler_dataset(test_dataset, test_target)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def build_dataset_pt(args):
    data_path = args.data_dir + args.dataset
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    if args.dataset == 'HAR':
        val_dataset_dict = torch.load(os.path.join(data_path, "val.pt"))
        val_dataset = val_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
        val_target = transfer_labels(val_dataset_dict["labels"].numpy())

        train_dataset = np.concatenate((train_dataset, val_dataset), axis=0)
        train_target = np.concatenate((train_target, val_target), axis=0)

    if args.dataset == 'FD':
        train_dataset = train_dataset.reshape(len(train_dataset), 1, -1)
        test_dataset = test_dataset.reshape(len(test_dataset), 1, -1)

    return train_dataset, train_target, test_dataset, test_target, num_classes

def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42):
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        t = W[y]
        A = x.reshape(1, -1).mm(W[y]).squeeze(0)

        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy() # if i not in ood_ids:

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != labels.detach().cpu().numpy()).mean()}')

    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        #
    print('****************************************')
    print('following is flip percentage:')

    for i in range(label_num):
        sum_i = sum(record[i])
        for j in range(label_num):
            if i != j:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
            else:
                print(f"{record[i][j] / sum_i: .2f}", end='\t')
        print()

    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            for j in range(label_num):
                print(f"{a[j]:.2f}", end="\t")
            print()
            cnt += 1
        if cnt >= 10:
            break
    print(P)
    return np.array(new_label)

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def build_for_dataset(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def noisify_asymmetric(args, y_train, noise):
    """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
    """
    nb_classes = args.num_classes
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = args.num_classes // 2
    nb_subclasses = 2

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            if i % 2 == 0:
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = build_for_dataset(nb_subclasses, n)

        y_train_noisy = multiclass_noisify(y_train, P=P)
        actual_noise = (y_train_noisy != y_train).mean()
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy

    return y_train, P

def flip_label(args, dataset, target, ratio):
    assert 0 <= ratio < 1

    _, channel, seq_len = dataset.shape
    target = np.array(target).astype(int)
    label = target.copy()
    n_class = len(np.unique(label))

    if args.noise_type == 'instance':
        # Instance
        num_classes = len(np.unique(target, return_counts=True)[0])
        data = torch.from_numpy(dataset).type(torch.FloatTensor)
        targets = torch.from_numpy(target).type(torch.FloatTensor).to(torch.int64)
        dataset_ = zip(data, targets)
        feature_size = dataset.shape[1] * dataset.shape[2]
        label = get_instance_noisy_label(n=ratio, dataset=dataset_, labels=targets, num_classes=num_classes,
                                         feature_size=feature_size, seed=args.random_seed)
    elif args.noise_type == 'asymmetric':
        label, _ = noisify_asymmetric(args, target, ratio)
    else:
        for i in range(label.shape[0]):
            # symmetric noise
            if args.noise_type == 'symmetric':
                p1 = ratio / (n_class - 1) * np.ones(n_class)
                p1[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p1)
            elif args.noise_type == 'pairflip':
                # pairflip
                label[i] = np.random.choice([label[i], (target[i] + 1) % n_class], p=[1 - ratio, ratio])

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])
    clean_ids = np.where(target == label)[0]
    # clean_rate = len(clean_ids) / len(target)
    # print('clean_rate:', clean_rate)

    return dataset, label, mask, clean_ids

def build_dataset(args):
    if args.archive == 'UEA':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_uea(args)
    elif args.archive == 'other':
        train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    input_channel = train_dataset.shape[1]
    seq_len = train_dataset.shape[2]
    args.num_classes = num_classes

    train_noisy_target, clean_ids = train_target.copy(), []
    if args.label_noise_rate > 0:
        train_dataset, train_noisy_target, mask_train_target, clean_ids = flip_label(args=args, dataset=train_dataset, target=train_target.copy(), ratio=args.label_noise_rate)

    # load train_loader
    train_loader = load_loader(args, train_dataset, train_noisy_target, clean_target=train_target, mode='train')
    # load test_loader
    test_loader = load_loader(args, test_dataset, test_target, shuffle=False, mode='test')

    return train_loader, test_loader, input_channel, seq_len, num_classes, clean_ids

def load_loader(args, data, target, clean_target=None, pred=None, prob=None, shuffle=True, mode='train'):
    if mode == 'train':
        dataset = TimeDataset(torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(target).type(torch.LongTensor),
                              clean_target=torch.from_numpy(clean_target).type(torch.LongTensor), mode=mode)
    elif mode == 'test':
        dataset = TimeDataset(torch.from_numpy(data).type(torch.FloatTensor), torch.from_numpy(target).type(torch.LongTensor), mode=mode)
    elif mode == 'labeled':
        dataset = TimeDataset(data, target, pred=torch.from_numpy(pred).type(torch.LongTensor), prob=torch.from_numpy(prob).type(torch.FloatTensor), mode=mode)
    elif mode == 'unlabeled':
        dataset = TimeDataset(data, target, pred=torch.from_numpy(pred).type(torch.LongTensor), prob=torch.from_numpy(prob).type(torch.FloatTensor), mode=mode)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=shuffle)

    return loader



