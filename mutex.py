import collections
import math
import pdb
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import collate, eval_format
from src import Decoder, EncDec, MultiIter, RecordLoss, Vocab, batch_seqs, weight_top_p

LossTrack = collections.namedtuple("LossTrack", "nll mlogpyx pointkl")


class Mutex(nn.Module):
    def __init__(
        self,
        vocab_x,
        vocab_y,
        emb,
        dim,
        copy=False,
        temp=1.0,
        max_len_x=8,
        max_len_y=8,
        n_layers=1,
        self_att=True,
        attention=True,
        dropout=0.0,
        bidirectional=True,
        rnntype=nn.LSTM,
        kl_lamda=1.0,
        recorder=RecordLoss(),
        qxy=None,
    ):

        super().__init__()

        self.vocab_x = vocab_x
        self.vocab_y = vocab_y
        self.rnntype = rnntype
        self.bidirectional = bidirectional
        self.dim = dim
        self.n_layers = n_layers
        self.temp = temp
        self.MAXLEN_X = max_len_x
        self.MAXLEN_Y = max_len_y
        self.pyx = EncDec(
            vocab_x,
            vocab_y,
            emb,
            dim,
            copy=copy,
            n_layers=n_layers,
            self_att=self_att,
            source_att=attention,
            dropout=dropout,
            bidirectional=bidirectional,
            rnntype=rnntype,
            MAXLEN=self.MAXLEN_Y,
        )
        if qxy:
            self.qxy = EncDec(
                vocab_y,
                vocab_x,
                emb,
                dim,
                copy=copy,
                n_layers=n_layers,
                self_att=self_att,
                dropout=dropout,
                bidirectional=bidirectional,
                rnntype=rnntype,
                source_att=attention,
                MAXLEN=self.MAXLEN_X,
            )
            # self.qxy = None
        self.recorder = recorder

    def forward(self, inp, out, lens=None, recorder=None):
        return self.pyx(inp, out, lens=lens)

    def print_tokens(self, vocab, tokens):
        return [" ".join(eval_format(vocab, tokens[i])) for i in range(len(tokens))]

    def sample(self, *args, **kwargs):
        return self.pyx.sample(*args, **kwargs)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
