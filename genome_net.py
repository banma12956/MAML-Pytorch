__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = makeDataLoader(depths, tnf)
>>> vae.trainmodel(dataloader)
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 32)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
import torch as torch
torch.manual_seed(0)

from math import log

from torch import nn
from torch.optim import Adam
from torch.nn.functional import softmax
from torch.utils.data import DataLoader


if torch.__version__ < '0.4':
    raise ImportError('PyTorch version must be 0.4 or newer')

class VAE(nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: List of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively  # TODO: why????
    """

    def __init__(self, nsamples, nhiddens=None, nlatent=32, alpha=None,
                 beta=200, dropout=0.2, cuda=False):
        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0, not {}'.format(nsamples))

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError('Minimum 1 neuron per layer, not {}'.format(min(nhiddens)))

        if beta <= 0:
            raise ValueError('beta must be > 0, not {}'.format(beta))

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1, not {}'.format(alpha))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1, not {}'.format(dropout))

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip([self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor):
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)  # TODO: why the hell you need this tensors???

        # Latent layers
        mu = self.mu(tensor)

        # Note: This softplus constrains logsigma to positive. As reconstruction loss pushes
        # logsigma as low as possible, and KLD pushes it towards 0, the optimizer will
        # always push this to 0, meaning that the logsigma layer will be pushed towards
        # negative infinity. This creates a nasty numerical instability in VAMB. Luckily,
        # the gradient also disappears as it decreases towards negative infinity, avoiding
        # NaN poisoning in most cases. We tried to remove the softplus layer, but this
        # necessitates a new round of hyperparameter optimization, and there is no way in
        # hell I am going to do that at the moment of writing.
        # Also remove needless factor 2 in definition of latent in reparameterize function.
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu, logsigma):
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * torch.exp(logsigma/2)

        return latent

    def _decode(self, tensor):
        # tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            # tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        # If multiple samples, apply softmax    TODO: why???
        if self.nsamples > 1:
            depths_out = softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(self, depths, tnf):
        tensor = torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = - ((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = ce * ce_weight + sse * sse_weight + kld * kld_weight

        return loss, ce, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, batchsteps, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0
        epoch_celoss = 0

        if epoch in batchsteps:
            data_loader = DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size * 2,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=data_loader.num_workers,
                                      pin_memory=data_loader.pin_memory)

        for depths_in, tnf_in in data_loader:
            depths_in.requires_grad = True
            tnf_in.requires_grad = True

            if self.usecuda:
                depths_in = depths_in.cuda()
                tnf_in = tnf_in.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)

            loss, ce, sse, kld = self.calc_loss(depths_in, depths_out, tnf_in,
                                                  tnf_out, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                  epoch + 1,
                  epoch_loss / len(data_loader),
                  epoch_celoss / len(data_loader),
                  epoch_sseloss / len(data_loader),
                  epoch_kldloss / len(data_loader),
                  data_loader.batch_size,
                  ), file=logfile)

            logfile.flush()

        return data_loader

    def encode(self, depths, tnf):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        if self.usecuda:
            depths = depths.cuda()
            tnf = tnf.cuda()

        # Evaluate
        out_depths, out_tnf, mu, logsigma = self(depths, tnf)

        if self.usecuda:
            mu = mu.cpu()

        latent[row: row + len(mu)] = mu
        row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nhiddens': self.nhiddens,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nhiddens = dictionary['nhiddens']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(self, dataloader, nepochs=500, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by makeDataLoader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))

        if batchsteps is None:
            batchsteps_set = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError('All elements of batchsteps must be integers')
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError('Max batchsteps must not equal or exceed nepochs')
            last_batchsize = dataloader.batch_size * 2**len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:
                raise ValueError('Last batch size exceeds dataset length')
            batchsteps_set = set(batchsteps)

        # Get number of features
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape
        optimizer = Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN hidden:', ', '.join(map(str, self.nhiddens)), file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            batchsteps_string = ', '.join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            print('\tBatchsteps:', batchsteps_string, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(dataloader, epoch, optimizer, batchsteps_set, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
