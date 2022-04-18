import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils
import parsecontigs


class ContigData(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, tasks, startidx=0):
        self.batchsz = batchsz
        self.n_way = n_way  # n-way, 5
        self.k_shot = k_shot  # k-shot, 1 and 15
        self.setsz = self.n_way * self.k_shot  # num of samples per set, 5
        self.path = root

        self.tasks = tasks
        self.tnf = [[None] for i in range(len(tasks))]
        self.rpkm = [[None] for i in range(len(tasks))]
        self.classes = [[None] for i in range(len(tasks))]
        nsample = 0
        for i in range(len(tasks)):
            path = os.path.join(root, tasks[i])
            tnfs = self.calc_tnf(path)
            self.tnf[i] = tnfs
            rpkms = self.calc_rpkm(os.path.join(path, 'abundance.npz'), len(tnfs))
            rpkms.resize(rpkms.shape[0], 10)
            self.rpkm[i] = rpkms
            if rpkms.shape[1] > nsample:
                nsample = rpkms.shape[1]
            self.classes[i] = utils.read_npz(os.path.join(path, 'classes.npz'))
        #for r in self.rpkm:
        #    r.resize(r.shape[0], nsample)
        utils.zscore(self.tnf, axis=0, inplace=True)
        utils.zscore(self.rpkm, axis=0, inplace=True)

        self.create_batch(self.batchsz)
    
    def calc_tnf(self, path):
        begintime = time.time()
        print('\nLoading TNF from' + path)
        tnfs = utils.read_npz(os.path.join(path, 'tnf.npz'))
        elapsed = round(time.time() - begintime, 2)
        print('Processed TNF in {} seconds'.format(elapsed))
        return tnfs

    def calc_rpkm(self, rpkmpath, ncontigs):
        begintime = time.time()
        print('Loading RPKM from' + rpkmpath)
        # If rpkm is given, we load directly from .npz file
        #print('Loading RPKM from npz array {}'.format(rpkmpath))
        rpkms = utils.read_npz(rpkmpath)

        if not rpkms.dtype == np.float32:
            raise ValueError('RPKMs .npz array must be of float32 dtype')

        if len(rpkms) != ncontigs:
            raise ValueError("Length of TNFs and length of RPKM does not match. Verify the inputs")

        elapsed = round(time.time() - begintime, 2)
        print('Processed RPKM in {} seconds'.format(elapsed))

        return rpkms

    def create_batch(self, batchsz):
        # 10000 batches, 5 tasks, 50 classes, 5 shots
        begintime = time.time()
        self.support_tnf = []  # support set batch
        self.support_rpkm = []
        self.support_y = []
        for b in range(batchsz):  # for each batch
            batch_tnf = []
            batch_rpkm = []
            batch_y = []
            class_count = 0
            for t in range(len(self.tasks)):
                selected_cls = np.random.choice(len(self.classes[t]), self.n_way, replace=False)
                #np.random.shuffle(selected_cls)
                for clas in selected_cls:
                    selected_idx = np.random.choice(self.classes[t][clas], self.k_shot)    # replace=True
                    #np.random.shuffle(selected_idx)
                    batch_tnf += self.tnf[t][selected_idx].tolist()
                    batch_rpkm += self.rpkm[t][selected_idx].tolist()
                    batch_y += [class_count]*self.k_shot
                    class_count += 1

            batch_tnf = np.stack(batch_tnf, axis=0)     # 1250 * 103
            batch_rpkm = np.stack(batch_rpkm, axis=0)   # 1250 * 10
            batch_y = np.stack(batch_y, axis=0)         # 1250

            self.support_tnf.append(batch_tnf)
            self.support_rpkm.append(batch_rpkm)
            self.support_y.append(batch_y)
        elapsed = round(time.time() - begintime, 2)
        print('Create batches in {} seconds'.format(elapsed))

    def __getitem__(self, index):
        support_tnf = torch.FloatTensor(self.support_tnf[index])
        support_rpkm = torch.FloatTensor(self.support_rpkm[index])
        support_classes = torch.Tensor(self.support_y[index])

        return support_tnf, support_rpkm, support_classes

    def __len__(self):
        return self.batchsz


if __name__ == '__main__':
    dataset = ContigData('/data/yunxiang/data/', 'train', 10000, 50, 5, ['airways', 'gi', 'oral', 'skin', 'urog'])
    db = DataLoader(dataset, 1, shuffle=True, pin_memory=True)
    for step, (tnf, rpkm, y) in enumerate(db):
        print(tnf.shape)
        print(rpkm.shape)
        print(y.shape)
        print(step)

