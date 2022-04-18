import os
import time
import numpy as np

import torch
from torch.utils.data import Dataset

import utils
import parsecontigs


class ContigData(Dataset):
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, tasks, startidx=0):
        self.batchsz = batchsz
        self.n_way = n_way  # n-way, 5
        self.k_shot = k_shot  # k-shot, 1 and 15
        self.k_query = k_query  # for evaluation, 15
        self.setsz = self.n_way * self.k_shot  # num of samples per set, 5
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.path = root

        self.tasks = tasks
        self.tnf = [None * len(tasks)]
        self.rpkm = [None * len(tasks)]
        for i in range(len(tasks)):
            path = os.path.join(root, tasks[i])
            tnfs, contignames, contiglengths = self.calc_tnf(path, os.path.join(path, 'contigs.fna'))
            self.tnf[i] = tnfs
            rpkms = calc_rpkm(path, len(tnfs))
            self.rpkm[i] = rpkms
    
    def calc_tnf(self, outdir, fastapath, annotated):
        begintime = time.time()
        print('\nLoading TNF from' + fastapath)
        # print('Minimum sequence length: {}'.format(mincontiglength))
        # Parse FASTA files. changed: since it only provides fasta files
        print('Loading data from FASTA file {}'.format(fastapath))
        with utils.Reader(fastapath, 'rb') as tnffile:
            ret = parsecontigs.read_contigs(tnffile)

        tnfs, contignames, contiglengths = ret
        utils.write_npz(os.path.join(outdir, annotated+'tnf.npz'), tnfs)
        utils.write_npz(os.path.join(outdir, annotated+'lengths.npz'), contiglengths)

        elapsed = round(time.time() - begintime, 2)
        ncontigs = len(contiglengths)
        nbases = contiglengths.sum()

        print('Kept {} bases in {} sequences'.format(nbases, ncontigs))
        print('Processed TNF in {} seconds'.format(elapsed))

        return tnfs, contignames, contiglengths

    def calc_rpkm(self, rpkmpath, ncontigs):
        begintime = time.time()
        print('\nLoading RPKM from' + rpkmpath)
        # If rpkm is given, we load directly from .npz file
        print('Loading RPKM from npz array {}'.format(rpkmpath))
        rpkms = utils.read_npz(rpkmpath)

        if not rpkms.dtype == np.float32:
            raise ValueError('RPKMs .npz array must be of float32 dtype')

        if len(rpkms) != ncontigs:
            raise ValueError("Length of TNFs and length of RPKM does not match. Verify the inputs")

        elapsed = round(time.time() - begintime, 2)
        print('Processed RPKM in {} seconds'.format(elapsed))

        return rpkms

    def create_batch(self, mode, batchsz):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    dataset = ContigData('/data/yunxiang/data/', train, 10000, 50, 5, 50, ['gi', 'metahit', 'oral', 'skin', 'urog'])
