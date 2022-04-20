import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy

from genome_net import VAE
from loss import eval_loss, eval_cluster


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, nsamples):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = VAE(nsamples)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, tnf_spt, rpkm_spt, y_spt):
        """

        :param tnf_spt:     [setsz, 103]
        :param rpkm_spt:    [setsz, nsamples]
        :param y_spt:       [setsz, ]
        :return:
        """

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        temp_net = deepcopy(self.net)
        
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

        # 1. run the i-th task and compute loss for k=0
        depths_out, tnf_out, mu, logsigma = self.net(rpkm_spt, tnf_spt)
        loss, _, _, _ = self.net.calc_loss(rpkm_spt, depths_out, tnf_spt, tnf_out, mu, logsigma)
        grad = torch.autograd.grad(loss, self.net.parameters())
        self.update_par(grad, self.update_lr)

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

        for k in range(1, self.update_step):
            depths_out, tnf_out, mu, logsigma = self.net(rpkm_spt, tnf_spt)
            loss, _, _, _ = self.net.calc_loss(rpkm_spt, depths_out, tnf_spt, tnf_out, mu, logsigma)
            grad = torch.autograd.grad(loss, self.net.parameters())
            self.update_par(grad, self.update_lr)

            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

        self.copy_par(self.net, temp_net)
        loss_q = losses_q[-1]

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return losses_q

    def update_par(self, grad, lr):
        for grad, param in zip(grad, self.net.parameters()):
            param.data -= lr * grad

    def copy_par(self, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data = param.data.clone().detach()

    def finetunning(self, tnf_qry, rpkm_qry, y_qry):
        """

        :param tnf_qry:     [setsz, 103]
        :param rpkm_qry:    [setsz, nsamples]
        :param y_qry:       [setsz, ]
        :return:
        """
        losses_q = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

            # eval_cluster(latent, y_qry)

        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

        for k in range(1, self.update_step_test):
            depths_out, tnf_out, mu, logsigma = self.net(rpkm_spt, tnf_spt)
            loss, _, _, _ = self.net.calc_loss(rpkm_spt, depths_out, tnf_spt, tnf_out, mu, logsigma)
            grad = torch.autograd.grad(loss, self.net.parameters())
            self.update_par(grad, self.update_lr)

            latent = self.net.encode(rpkm_spt, tnf_spt)
            loss_q = eval_loss(latent, y_spt)
            losses_q[0] += loss_q

        eval_cluster(latent, y_qry)
        del net

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()

